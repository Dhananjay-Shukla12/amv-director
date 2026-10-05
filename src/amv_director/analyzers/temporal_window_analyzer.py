from __future__ import annotations

import json
import math
from pathlib import Path

import cv2


# Reuse the existing frame-quality logic.
from source_quality_analyzer import (
    frame_metrics,
    aggregate_scene,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CLIPS_DIR = (
    PROJECT_ROOT
    / "data"
    / "clips"
)

CLIP_ANALYSIS_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "clip_analysis"
    / "all_clips_analysis.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "clip_analysis"
    / "temporal_window_analysis.json"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_FPS = 5.0

WINDOW_DURATIONS = [
    0.40,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
    3.00,
    4.00,
    5.00,
    7.00,
]

# Step between candidate windows.
WINDOW_STEP = 0.25


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def safe_float(
    value,
    default=0.0,
):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return default


# ============================================================
# LOAD SOURCE SCENES
# ============================================================

def load_source_data():

    data = load_json(
        CLIP_ANALYSIS_FILE
    )

    return data


# ============================================================
# SEQUENTIAL VIDEO SAMPLING
# ============================================================

def collect_scene_frames(
    video_path,
    scenes,
):
    """
    Sequentially decode the complete video and assign each
    sampled frame to its detected scene.

    This avoids the H264 random-seeking problems encountered
    earlier in the project.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open:\n{video_path}"
        )

    fps = safe_float(
        cap.get(
            cv2.CAP_PROP_FPS
        ),
        30.0,
    )

    if fps <= 0:
        fps = 30.0

    frame_step = max(
        1,
        int(
            round(
                fps / SAMPLE_FPS
            )
        ),
    )

    buffers = {
        int(scene["scene_id"]): []
        for scene in scenes
    }

    scene_index = 0
    current_scene = (
        scenes[0]
        if scenes
        else None
    )

    frame_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        timestamp = (
            frame_index /
            fps
        )

        frame_index += 1

        if (
            current_scene is None
            or
            (
                frame_index - 1
            ) % frame_step != 0
        ):
            continue

        # Move to the scene containing this timestamp.
        while (
            current_scene is not None
            and
            timestamp >= float(
                current_scene["end"]
            )
            and
            scene_index <
            len(scenes) - 1
        ):

            scene_index += 1

            current_scene = (
                scenes[scene_index]
            )

        if current_scene is None:
            continue

        start = float(
            current_scene["start"]
        )

        end = float(
            current_scene["end"]
        )

        if not (
            start <= timestamp < end
        ):
            continue

        metrics = frame_metrics(
            frame
        )

        buffers[
            int(
                current_scene[
                    "scene_id"
                ]
            )
        ].append(
            {
                "time": timestamp,
                "metrics": metrics,
            }
        )

    cap.release()

    return buffers


# ============================================================
# WINDOW QUALITY
# ============================================================

def evaluate_window(
    scene,
    frames,
    window_start,
    window_end,
):
    """
    Evaluate a contiguous temporal portion of one scene.
    """

    selected = [
        item["metrics"]
        for item in frames
        if (
            window_start
            <= item["time"]
            <= window_end
        )
    ]

    if not selected:
        return None

    window_scene = {
        "scene_id": scene["scene_id"],
        "start": window_start,
        "end": window_end,
        "duration": (
            window_end -
            window_start
        ),
    }

    result = aggregate_scene(
        window_scene,
        selected,
    )

    return result


# ============================================================
# GENERATE WINDOWS
# ============================================================

def analyze_scene_windows(
    scene,
    frames,
):
    """
    Find the strongest temporal windows of several durations.
    """

    if not frames:
        return {}

    scene_start = float(
        scene["start"]
    )

    scene_end = float(
        scene["end"]
    )

    scene_duration = (
        scene_end -
        scene_start
    )

    best_windows = {}

    for requested_duration in (
        WINDOW_DURATIONS
    ):

        if requested_duration > (
            scene_duration + 0.05
        ):
            continue

        best = None

        current = scene_start

        while (
            current +
            requested_duration
            <= scene_end + 0.001
        ):

            window_start = current

            window_end = min(
                scene_end,
                current +
                requested_duration,
            )

            result = evaluate_window(
                scene,
                frames,
                window_start,
                window_end,
            )

            if result is None:
                current += WINDOW_STEP
                continue

            quality = float(
                result.get(
                    "quality_score",
                    0.0,
                )
            )

            candidate = {
                "start": round(
                    window_start,
                    3,
                ),
                "end": round(
                    window_end,
                    3,
                ),
                "duration": round(
                    window_end -
                    window_start,
                    3,
                ),
                "quality_score": round(
                    quality,
                    4,
                ),
                "usable": bool(
                    result.get(
                        "usable",
                        False,
                    )
                ),
                "reason": result.get(
                    "reason",
                    "",
                ),
                "metrics": result.get(
                    "metrics",
                    {},
                ),
                "subscores": result.get(
                    "subscores",
                    {},
                ),
            }

            if (
                best is None
                or
                candidate[
                    "quality_score"
                ]
                >
                best[
                    "quality_score"
                ]
            ):
                best = candidate

            current += WINDOW_STEP

        if best is not None:
            best_windows[
                f"{requested_duration:.2f}s"
            ] = best

    return best_windows


# ============================================================
# REPORT
# ============================================================

def print_scene_report(
    video_name,
    scene,
    windows,
):
    """
    Print only the most useful temporal windows for the scene.
    """

    print()

    print(
        f"{video_name} "
        f"S{scene['scene_id']:02d} "
        f"{scene['start']:.3f}"
        f"→"
        f"{scene['end']:.3f}"
        f"s "
        f"({scene['duration']:.3f}s)"
    )

    if not windows:
        print(
            "   No temporal windows found."
        )
        return

    # Print a few important durations.
    preferred = [
        "0.50s",
        "0.75s",
        "1.00s",
        "1.50s",
        "2.00s",
        "3.00s",
        "4.00s",
        "5.00s",
        "7.00s",
    ]

    for duration_key in preferred:

        item = windows.get(
            duration_key
        )

        if item is None:
            continue

        print(
            f"   {duration_key:>5} "
            f"{item['start']:.3f}"
            f"→"
            f"{item['end']:.3f} "
            f"quality="
            f"{item['quality_score']:.3f} "
            f"{item['reason']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "🔎 TEMPORAL WINDOW ANALYZER"
    )
    print()

    source_data = load_source_data()

    all_results = []

    total_scenes = 0

    for video_data in source_data:

        video_name = video_data[
            "video"
        ]

        scenes = video_data.get(
            "scenes",
            [],
        )

        video_path = (
            CLIPS_DIR /
            video_name
        )

        print(
            f"🎬 {video_name}"
        )

        if not video_path.exists():

            print(
                f"   ❌ Missing: "
                f"{video_path}"
            )

            continue

        print(
            f"   Sampling {len(scenes)} scenes..."
        )

        buffers = collect_scene_frames(
            video_path,
            scenes,
        )

        video_result = {
            "video": video_name,
            "scenes": [],
        }

        for scene in scenes:

            scene_id = int(
                scene["scene_id"]
            )

            frames = buffers.get(
                scene_id,
                [],
            )

            windows = (
                analyze_scene_windows(
                    scene,
                    frames,
                )
            )

            scene_result = {
                "scene_id": scene_id,
                "start": scene["start"],
                "end": scene["end"],
                "duration": scene["duration"],
                "sample_count": len(
                    frames
                ),
                "best_windows": windows,
            }

            video_result[
                "scenes"
            ].append(
                scene_result
            )

            total_scenes += 1

            # Only print especially long or weak scenes
            # because those are the most likely to contain
            # mixed-quality temporal regions.
            if (
                float(
                    scene["duration"]
                ) >= 3.0
            ):
                print_scene_report(
                    video_name,
                    scene,
                    windows,
                )

        all_results.append(
            video_result
        )

    output = {
        "project": "AMV Director",
        "version": "temporal_window_v1",
        "sample_fps": SAMPLE_FPS,
        "window_durations": (
            WINDOW_DURATIONS
        ),
        "scene_count": total_scenes,
        "videos": all_results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # GLOBAL REPORT
    # --------------------------------------------------------

    print()
    print(
        "=" * 84
    )

    print(
        "🔎 TEMPORAL WINDOW ANALYSIS COMPLETE"
    )

    print(
        "=" * 84
    )

    print(
        f"Scenes analyzed : "
        f"{total_scenes}"
    )

    print(
        "Window durations : "
        f"{', '.join(
            f'{x:.2f}s'
            for x in WINDOW_DURATIONS
        )}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "=" * 84
    )


if __name__ == "__main__":
    main()