from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import imageio_ffmpeg

from amv_director.renderer_v5 import (
    render_clip,
    concat_video,
    mux_audio,
    verify_output,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CREATIVE_PLAN = (
    BASE_DIR
    / "data/outputs/creative_director_plan_v3.json"
)

EFFECT_PLAN = (
    BASE_DIR
    / "data/outputs/effect_director_bridge_v2.json"
)

DEFAULT_AUDIO = (
    BASE_DIR
    / "data/outputs/reference_audio.wav"
)

DEFAULT_OUTPUT = (
    BASE_DIR
    / "data/outputs/renders/amv_director_v6_preview.mp4"
)

SOURCE_DIR = (
    BASE_DIR
    / "data/clips"
)


# ============================================================
# SETTINGS
# ============================================================

FPS = 29.97

CRF = 18
PRESET = "veryfast"

MIN_SOURCE_DURATION = 0.04


# ============================================================
# HELPERS
# ============================================================

def f(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        return float(value)

    except Exception:
        return default


def load_json(
    path: Path,
) -> Any:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as fh:

        return json.load(fh)


def run_ffmpeg(
    ffmpeg: str,
    command: List[str],
    description: str,
) -> None:

    print(
        f"    FFmpeg: {description}"
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:

        print(result.stdout)

        raise RuntimeError(
            f"FFmpeg failed: {description}"
        )


# ============================================================
# SOURCE PATH
# ============================================================

def resolve_source(
    video_name: str,
) -> Path:

    candidate = Path(
        video_name
    ).expanduser()

    if candidate.is_absolute() and candidate.exists():
        return candidate

    candidate = (
        SOURCE_DIR
        / Path(video_name).name
    )

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Source video not found:\n"
        f"{candidate}"
    )


# ============================================================
# EFFECT ADAPTER
# ============================================================

def convert_action(
    action: Dict[str, Any],
    shot_duration: float,
) -> Dict[str, Any]:

    effect = str(
        action.get(
            "effect",
            "",
        )
    )

    at = max(
        0.0,
        f(
            action.get(
                "at",
                0.0,
            )
        ),
    )

    duration = max(
        0.025,
        f(
            action.get(
                "duration",
                0.05,
            )
        ),
    )

    start = min(
        at,
        shot_duration,
    )

    end = min(
        shot_duration,
        start + duration,
    )

    if end <= start:
        return {}

    strength = max(
        0.0,
        min(
            1.0,
            f(
                action.get(
                    "strength",
                    0.5,
                )
            ),
        ),
    )

    converted = {
        "effect": effect,
        "start": round(start, 4),
        "end": round(end, 4),
        "strength": round(
            strength,
            4,
        ),
    }

    # --------------------------------------------------------
    # V5 uses "amount" for these movement effects while the
    # Bridge V2 expresses intensity as "strength".
    # --------------------------------------------------------

    if effect in {
        "micro_push",
        "pre_push",
    }:

        converted["amount"] = round(
            0.018
            + 0.045 * strength,
            5,
        )

    elif effect == "directional_motion":

        converted["amount"] = round(
            0.018
            + 0.065 * strength,
            5,
        )

    elif effect == "camera_drift":

        converted["amount"] = round(
            0.006
            + 0.025 * strength,
            5,
        )

    # Preserve direction when present.
    if "direction" in action:

        converted["direction"] = (
            action["direction"]
        )

    # Preserve blur mode/parameters where available.
    for key in (
        "frequency",
        "mode",
    ):

        if key in action:
            converted[key] = action[key]

    return converted


def build_effect_segment(
    shot: Dict[str, Any],
    effect_plan: Dict[str, Any],
) -> Dict[str, Any]:

    shot_id = int(
        shot["shot_id"]
    )

    duration = max(
        0.05,
        f(
            shot["timeline"]["duration"]
        ),
    )

    effect_lookup = {
        int(item["shot_id"]): item
        for item in effect_plan.get(
            "shot_effects",
            [],
        )
    }

    source_effect = effect_lookup.get(
        shot_id,
        {},
    )

    actions = []

    for raw_action in source_effect.get(
        "actions",
        [],
    ):

        converted = convert_action(
            raw_action,
            duration,
        )

        if converted:
            actions.append(
                converted
            )

    role = (
        shot
        .get("music", {})
        .get(
            "role",
            "cinematic_zone",
        )
    )

    # Keep the renderer's finishing conservative.
    return {
        "editorial_role": role,

        "camera_profile": {
            "zoom_start": 1.0,
            "zoom_end": 1.0,
        },

        "actions": actions,

        "finishing": {
            "contrast": 1.03,
            "saturation": 1.0,
        },
    }


# ============================================================
# LOOP SHORT SOURCE SCENES
# ============================================================

def make_looped_source(
    ffmpeg: str,
    source: Path,
    source_start: float,
    source_duration: float,
    target_duration: float,
    temp_dir: Path,
    index: int,
) -> Path:

    source_duration = max(
        MIN_SOURCE_DURATION,
        source_duration,
    )

    extracted = (
        temp_dir
        / f"source_{index:04d}.mp4"
    )

    looped = (
        temp_dir
        / f"looped_{index:04d}.mp4"
    )

    # --------------------------------------------------------
    # Extract only the selected scene/window.
    # --------------------------------------------------------

    extract_command = [

        ffmpeg,

        "-hide_banner",
        "-loglevel",
        "error",

        "-ss",
        f"{source_start:.6f}",

        "-t",
        f"{source_duration:.6f}",

        "-i",
        str(source),

        "-an",

        "-c:v",
        "libx264",

        "-preset",
        PRESET,

        "-crf",
        str(CRF),

        "-pix_fmt",
        "yuv420p",

        "-r",
        str(FPS),

        "-y",
        str(extracted),
    ]

    run_ffmpeg(
        ffmpeg,
        extract_command,
        f"extract source scene {index}",
    )

    # --------------------------------------------------------
    # Loop the extracted scene until the desired shot
    # duration is reached.
    # --------------------------------------------------------

    loop_command = [

        ffmpeg,

        "-hide_banner",
        "-loglevel",
        "error",

        "-stream_loop",
        "-1",

        "-i",
        str(extracted),

        "-t",
        f"{target_duration:.6f}",

        "-an",

        "-c:v",
        "libx264",

        "-preset",
        PRESET,

        "-crf",
        str(CRF),

        "-pix_fmt",
        "yuv420p",

        "-r",
        str(FPS),

        "-y",
        str(looped),
    ]

    run_ffmpeg(
        ffmpeg,
        loop_command,
        f"loop short source scene {index}",
    )

    return looped


# ============================================================
# BUILD RENDER CLIP
# ============================================================

def build_render_clip(
    shot: Dict[str, Any],
    overlap_start: float,
    overlap_end: float,
    prepared_source: Path,
    original_source_window_start: float,
    source_was_looped: bool,
) -> Dict[str, Any]:

    shot_start = f(
        shot["timeline"]["start"]
    )

    shot_duration = max(
        0.05,
        f(
            shot["timeline"]["duration"]
        ),
    )

    offset_inside_shot = max(
        0.0,
        overlap_start - shot_start,
    )

    render_duration = (
        overlap_end
        - overlap_start
    )

    if render_duration <= 0:
        raise ValueError(
            "Invalid render duration."
        )

    if source_was_looped:

        render_source_start = (
            offset_inside_shot
        )

    else:

        render_source_start = (
            original_source_window_start
            + offset_inside_shot
        )

    return {

        "segment_id":
            int(
                shot["shot_id"]
            ),

        "segment_start":
            shot_start,

        "timeline_start":
            shot_start,

        "timeline_end":
            shot_start
            + shot_duration,

        "duration":
            shot_duration,

        "source_video":
            shot["source"]["video"],

        "source_path":
            str(prepared_source),

        "scene_id":
            shot["source"]["scene_id"],

        "source_start":
            original_source_window_start,

        "source_end":
            original_source_window_start
            + shot["source"][
                "selected_window"
            ]["duration"],

        "preview_start":
            overlap_start,

        "preview_end":
            overlap_end,

        "render_source_start":
            render_source_start,

        "render_duration":
            render_duration,

        "effect_offset":
            offset_inside_shot,

    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AMV Director Renderer V6"
        )
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

    ffmpeg = (
        imageio_ffmpeg
        .get_ffmpeg_exe()
    )

    creative = load_json(
        CREATIVE_PLAN
    )

    effect_plan = load_json(
        EFFECT_PLAN
    )

    shots = creative.get(
        "shot_plan",
        [],
    )

    if not shots:

        raise RuntimeError(
            "No shot_plan found in:\n"
            f"{CREATIVE_PLAN}"
        )

    audio_path = (
        Path(
            args.audio
        )
        .expanduser()
        .resolve()
    )

    output_path = (
        Path(
            args.output
        )
        .expanduser()
        .resolve()
    )

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio not found:\n"
            f"{audio_path}"
        )

    total_duration = max(
        f(
            shot["timeline"]["end"]
        )
        for shot in shots
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

    if preview_end <= preview_start:

        raise RuntimeError(
            "Invalid render window."
        )

    print("=" * 82)
    print("🎬 AMV DIRECTOR V6 RENDERER")
    print("=" * 82)

    print(
        f"\nPlan shots       : "
        f"{len(shots)}"
    )

    print(
        f"Plan duration    : "
        f"{total_duration:.3f}s"
    )

    print(
        f"Render window    : "
        f"{preview_start:.3f}"
        f"→"
        f"{preview_end:.3f}s"
    )

    print()

    # --------------------------------------------------------
    # Validate effect plan
    # --------------------------------------------------------

    effect_shot_ids = {
        int(item["shot_id"])
        for item in effect_plan.get(
            "shot_effects",
            [],
        )
    }

    missing_effects = [

        int(shot["shot_id"])

        for shot
        in shots

        if int(
            shot["shot_id"]
        )
        not in effect_shot_ids

    ]

    if missing_effects:

        raise RuntimeError(
            "Missing effect plans for shots: "
            f"{missing_effects}"
        )

    # --------------------------------------------------------
    # Temporary workspace
    # --------------------------------------------------------

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="amv_director_v6_"
        )
    )

    rendered_files = []

    try:

        print(
            f"Temporary directory: "
            f"{temp_dir}"
        )

        # ====================================================
        # RENDER SELECTED SHOTS
        # ====================================================

        render_index = 0

        for shot in shots:

            shot_start = f(
                shot["timeline"]["start"]
            )

            shot_end = f(
                shot["timeline"]["end"]
            )

            overlap_start = max(
                preview_start,
                shot_start,
            )

            overlap_end = min(
                preview_end,
                shot_end,
            )

            if overlap_end <= overlap_start:
                continue

            render_index += 1

            source_info = shot[
                "source"
            ]

            selected_window = source_info[
                "selected_window"
            ]

            source_window_start = f(
                selected_window.get(
                    "start",
                    source_info[
                        "original_start"
                    ],
                )
            )

            source_window_end = f(
                selected_window.get(
                    "end",
                    source_info[
                        "original_end"
                    ],
                )
            )

            source_duration = max(
                MIN_SOURCE_DURATION,
                source_window_end
                - source_window_start,
            )

            shot_duration = (
                shot_end
                - shot_start
            )

            source = resolve_source(
                source_info["video"]
            )

            source_was_looped = (
                source_duration
                < shot_duration
            )

            if source_was_looped:

                prepared_source = (
                    make_looped_source(
                        ffmpeg,
                        source,
                        source_window_start,
                        source_duration,
                        shot_duration,
                        temp_dir,
                        render_index,
                    )
                )

            else:

                prepared_source = source

            clip = build_render_clip(
                shot,
                overlap_start,
                overlap_end,
                prepared_source,
                source_window_start,
                source_was_looped,
            )

            effects = build_effect_segment(
                shot,
                effect_plan,
            )

            output_clip = (
                temp_dir
                / f"rendered_{render_index:04d}.mp4"
            )

            print(
                f"[{render_index:02d}] "
                f"Shot {shot['shot_id']:02d} | "
                f"{shot_start:.3f}"
                f"→"
                f"{shot_end:.3f} | "
                f"source="
                f"{source_info['video']}:"
                f"{source_info['scene_id']} | "
                f"source_window="
                f"{source_duration:.3f}s | "
                f"target="
                f"{shot_duration:.3f}s | "
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
                "No clips overlapped the render window."
            )

        # ====================================================
        # CONCAT
        # ====================================================

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

        # ====================================================
        # AUDIO
        # ====================================================

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
            preview_end
            - preview_start,
        )

        # ====================================================
        # VERIFY
        # ====================================================

        has_video, has_audio = (
            verify_output(
                ffmpeg,
                output_path,
            )
        )

        print()
        print("=" * 82)
        print("✅ V6 RENDER COMPLETE")
        print("=" * 82)

        print(
            f"\nOutput : "
            f"{output_path}"
        )

        print(
            f"Video  : "
            f"{'OK' if has_video else 'FAILED'}"
        )

        print(
            f"Audio  : "
            f"{'OK' if has_audio else 'FAILED'}"
        )

        print("=" * 82)

        if not has_video or not has_audio:

            raise RuntimeError(
                "Output verification failed."
            )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()