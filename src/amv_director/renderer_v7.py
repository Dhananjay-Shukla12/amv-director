from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg

from amv_director.renderer_v5 import (
    render_clip,
    concat_video,
    mux_audio,
    verify_output,
)

from amv_director.renderer_v6 import (
    resolve_source,
    convert_action,
)


BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_TEMPORAL_PLAN = (
    BASE_DIR
    / "data/outputs/temporal_edit_plan_v2.json"
)

DEFAULT_EFFECT_PLAN = (
    BASE_DIR
    / "data/outputs/effect_director_bridge_v2.json"
)

DEFAULT_CREATIVE_PLAN = (
    BASE_DIR
    / "data/outputs/creative_director_plan_v3.json"
)

DEFAULT_AUDIO = (
    BASE_DIR
    / "data/outputs/reference_audio.wav"
)

DEFAULT_OUTPUT = (
    BASE_DIR
    / "data/outputs/renders/amv_director_v7_preview.mp4"
)


SOURCE_DIR = (
    BASE_DIR
    / "data/clips"
)


MIN_DURATION = 0.04
EPSILON = 0.02


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_parent_lookup(
    creative_plan: dict,
) -> dict[int, dict]:
    lookup = {}

    for shot in creative_plan.get(
        "shot_plan",
        [],
    ):
        try:
            shot_id = int(
                shot["shot_id"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        timeline = shot.get(
            "timeline",
            {},
        )

        lookup[shot_id] = {
            "start": num(
                timeline.get("start")
            ),
            "end": num(
                timeline.get("end")
            ),
            "duration": num(
                timeline.get("duration")
            ),
            "music_role": (
                shot.get(
                    "music",
                    {},
                ).get(
                    "role",
                    "cinematic_zone",
                )
            ),
        }

    return lookup


def build_effect_lookup(
    effect_plan: dict,
) -> dict[int, dict]:
    return {
        int(item["shot_id"]): item
        for item in effect_plan.get(
            "shot_effects",
            [],
        )
        if item.get("shot_id") is not None
    }


def build_parent_effect_segment(
    parent_id: int,
    parent_info: dict,
    effect_lookup: dict[int, dict],
) -> dict:

    source = effect_lookup.get(
        parent_id
    )

    parent_duration = max(
        MIN_DURATION,
        parent_info["duration"],
    )

    if source is None:
        return {
            "editorial_role":
                parent_info["music_role"],

            "camera_profile": {
                "zoom_start": 1.0,
                "zoom_end": 1.0,
            },

            "actions": [],

            "finishing": {
                "contrast": 1.03,
                "saturation": 1.0,
            },
        }

    converted_actions = []

    for raw_action in source.get(
        "actions",
        [],
    ):

        converted = convert_action(
            raw_action,
            parent_duration,
        )

        if converted:
            converted_actions.append(
                converted
            )

    return {
        "editorial_role":
            parent_info["music_role"],

        "camera_profile":
            source.get(
                "camera_profile",
                {
                    "zoom_start": 1.0,
                    "zoom_end": 1.0,
                },
            ),

        "actions":
            converted_actions,

        "finishing":
            source.get(
                "finishing",
                {
                    "contrast": 1.03,
                    "saturation": 1.0,
                },
            ),
    }


def make_render_clip(
    part: dict,
    overlap_start: float,
    overlap_end: float,
    source_path: Path,
    parent_start: float,
) -> dict:

    timeline = part["timeline"]
    source = part["source"]

    part_start = num(
        timeline["start"]
    )

    part_end = num(
        timeline["end"]
    )

    part_duration = max(
        MIN_DURATION,
        num(
            timeline["duration"]
        ),
    )

    source_start = num(
        source["source_start"]
    )

    source_end = num(
        source["source_end"]
    )

    source_span = (
        source_end
        - source_start
    )

    # ------------------------------------------------------------
    # Critical safety check:
    # Temporal V2 promised that source material is never looped.
    # ------------------------------------------------------------

    if source_span + EPSILON < part_duration:
        raise RuntimeError(
            "Temporal plan violates no-loop rule:\n"
            f"parent={part['parent_shot_id']}\n"
            f"source_span={source_span:.4f}s\n"
            f"part_duration={part_duration:.4f}s"
        )

    overlap_duration = (
        overlap_end
        - overlap_start
    )

    if overlap_duration <= MIN_DURATION:
        raise RuntimeError(
            "Invalid render overlap."
        )

    offset_inside_part = (
        overlap_start
        - part_start
    )

    render_source_start = (
        source_start
        + offset_inside_part
    )

    render_source_end = (
        render_source_start
        + overlap_duration
    )

    if render_source_end > source_end + EPSILON:
        raise RuntimeError(
            "Render request exceeds source window:\n"
            f"parent={part['parent_shot_id']}\n"
            f"requested={render_source_end:.4f}\n"
            f"available_end={source_end:.4f}"
        )

    # Effects are still authored relative to the parent shot.
    # The V5 renderer uses effect_offset to localize them.
    effect_offset = (
        part_start
        - parent_start
    )

    return {
        "segment_id":
            int(part["parent_shot_id"]),

        "segment_start":
            parent_start,

        "timeline_start":
            part_start,

        "timeline_end":
            part_end,

        "duration":
            part_duration,

        "source_video":
            source["video"],

        "source_path":
            str(source_path),

        "scene_id":
            source.get("scene_id"),

        "source_start":
            source_start,

        "source_end":
            source_end,

        "render_source_start":
            render_source_start,

        "render_source_end":
            render_source_end,

        "render_duration":
            overlap_duration,

        "effect_offset":
            effect_offset,
    }


def intersect_part(
    part: dict,
    render_start: float,
    render_end: float,
) -> tuple[float, float] | None:

    start = num(
        part["timeline"]["start"]
    )

    end = num(
        part["timeline"]["end"]
    )

    overlap_start = max(
        start,
        render_start,
    )

    overlap_end = min(
        end,
        render_end,
    )

    if overlap_end <= overlap_start + 1e-6:
        return None

    return (
        overlap_start,
        overlap_end,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "AMV Director Renderer V7 — "
            "Temporal V2 + Effect Bridge V2"
        )
    )

    parser.add_argument(
        "--temporal-plan",
        default=str(
            DEFAULT_TEMPORAL_PLAN
        ),
    )

    parser.add_argument(
        "--effect-plan",
        default=str(
            DEFAULT_EFFECT_PLAN
        ),
    )

    parser.add_argument(
        "--creative-plan",
        default=str(
            DEFAULT_CREATIVE_PLAN
        ),
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--audio",
        default=str(
            DEFAULT_AUDIO
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    args = parser.parse_args()

    temporal_plan_path = Path(
        args.temporal_plan
    ).expanduser().resolve()

    effect_plan_path = Path(
        args.effect_plan
    ).expanduser().resolve()

    creative_plan_path = Path(
        args.creative_plan
    ).expanduser().resolve()

    audio_path = Path(
        args.audio
    ).expanduser().resolve()

    output_path = Path(
        args.output
    ).expanduser().resolve()

    temporal_plan = load_json(
        temporal_plan_path
    )

    effect_plan = load_json(
        effect_plan_path
    )

    creative_plan = load_json(
        creative_plan_path
    )

    parent_lookup = build_parent_lookup(
        creative_plan
    )

    effect_lookup = build_effect_lookup(
        effect_plan
    )

    shot_plan = temporal_plan.get(
        "shot_plan",
        [],
    )

    if not shot_plan:
        raise RuntimeError(
            "Temporal plan contains no shot_plan."
        )

    total_duration = max(
        num(
            part["timeline"]["end"]
        )
        for part in shot_plan
    )

    preview_start = max(
        0.0,
        args.start,
    )

    preview_end = min(
        total_duration,
        preview_start
        + max(
            0.1,
            args.duration,
        ),
    )

    actual_duration = (
        preview_end
        - preview_start
    )

    if actual_duration <= 0:
        raise RuntimeError(
            "Invalid preview duration."
        )

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    selected_parts = []

    for part in shot_plan:

        overlap = intersect_part(
            part,
            preview_start,
            preview_end,
        )

        if overlap is None:
            continue

        selected_parts.append(
            (
                part,
                overlap[0],
                overlap[1],
            )
        )

    selected_parts.sort(
        key=lambda item:
            item[0]["timeline"]["start"]
    )

    if not selected_parts:
        raise RuntimeError(
            "No temporal shots overlap preview."
        )

    print()
    print("=" * 90)
    print("🎬 AMV DIRECTOR V7")
    print("=" * 90)
    print()
    print(
        f"Temporal plan : "
        f"{temporal_plan_path.name}"
    )
    print(
        f"Timeline      : "
        f"{preview_start:.3f} → "
        f"{preview_end:.3f}s"
    )
    print(
        f"Duration      : "
        f"{actual_duration:.3f}s"
    )
    print(
        f"Render parts  : "
        f"{len(selected_parts)}"
    )
    print(
        "Source looping: DISABLED"
    )
    print()

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="amv_director_v7_"
        )
    )

    rendered_files = []

    try:

        for render_index, (
            part,
            overlap_start,
            overlap_end,
        ) in enumerate(
            selected_parts,
            start=1,
        ):

            parent_id = int(
                part["parent_shot_id"]
            )

            parent_info = parent_lookup.get(
                parent_id
            )

            if parent_info is None:
                raise RuntimeError(
                    f"Parent shot {parent_id} "
                    "not found in Creative Director plan."
                )

            source_info = part["source"]

            source_video = source_info.get(
                "video"
            )

            if not source_video:
                raise RuntimeError(
                    f"Missing source video "
                    f"for temporal part "
                    f"{render_index}."
                )

            source_path = resolve_source(
                source_video
            )

            clip = make_render_clip(
                part=part,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                source_path=source_path,
                parent_start=parent_info["start"],
            )

            effects = build_parent_effect_segment(
                parent_id=parent_id,
                parent_info=parent_info,
                effect_lookup=effect_lookup,
            )

            output_clip = (
                temp_dir
                / f"rendered_{render_index:04d}.mp4"
            )

            source_duration = (
                num(
                    source_info[
                        "source_end"
                    ]
                )
                -
                num(
                    source_info[
                        "source_start"
                    ]
                )
            )

            print(
                f"[{render_index:02d}] "
                f"Parent {parent_id:02d} "
                f"Part {part['part_index']} | "
                f"{overlap_start:.3f}"
                f"→"
                f"{overlap_end:.3f} | "
                f"source="
                f"{source_video}:"
                f"{source_info.get('scene_id')} | "
                f"source="
                f"{source_duration:.3f}s | "
                f"render="
                f"{overlap_end - overlap_start:.3f}s | "
                f"effects="
                f"{len(effects['actions'])}"
            )

            render_clip(
                ffmpeg,
                clip,
                effects,
                output_clip,
            )

            rendered_files.append(
                output_clip
            )

        if not rendered_files:
            raise RuntimeError(
                "No rendered clips were produced."
            )

        concat_file = (
            temp_dir
            / "concat.txt"
        )

        silent_video = (
            temp_dir
            / "silent.mp4"
        )

        concat_video(
            ffmpeg,
            rendered_files,
            concat_file,
            silent_video,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        mux_audio(
            ffmpeg,
            silent_video,
            audio_path,
            output_path,
            preview_start,
            actual_duration,
        )

        verify_output(
        ffmpeg,
        output_path,
        )

        print()
        print("=" * 90)
        print("✅ RENDER COMPLETE")
        print("=" * 90)
        print(
            f"Output: {output_path}"
        )
        print(
            f"Duration: {actual_duration:.3f}s"
        )
        print(
            "Looping: 0"
        )
        print("=" * 90)

    finally:

        # Keep the final output, but clean temporary render pieces.
        import shutil

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()