from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scenedetect import AdaptiveDetector, detect


def calculate_motion_score(
    video_path: str,
    start_frame: int,
    end_frame: int,
) -> float:
    """
    Estimate visual motion inside a scene.

    The result is later normalized against the video's
    own motion distribution rather than using a fixed
    arbitrary threshold.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = max(1, end_frame - start_frame)
    step = max(1, frame_count // 24)

    previous = None
    raw_scores: list[float] = []

    for frame_number in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = cap.read()

        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))

        if previous is not None:
            difference = cv2.absdiff(previous, gray)
            raw_scores.append(float(np.mean(difference)) / 255.0)

        previous = gray

    cap.release()

    if not raw_scores:
        return 0.0

    return float(np.median(raw_scores))


def extract_representative_frame(
    video_path: str,
    frame_number: int,
    output_path: Path,
) -> None:
    """Save a representative frame for a scene."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()
    cap.release()

    if not success:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)


def normalize_motion_scores(
    scenes: list[dict[str, Any]],
) -> None:
    """
    Convert raw motion scores into percentile-based 0-1 scores.

    This avoids saturation where many unrelated scenes become 1.0.
    """
    if not scenes:
        return

    values = np.array(
        [scene["raw_motion_score"] for scene in scenes],
        dtype=np.float32,
    )

    if len(values) == 1:
        scenes[0]["motion_score"] = 0.5
        return

    sorted_values = np.sort(values)

    for scene in scenes:
        raw = scene["raw_motion_score"]

        rank = np.searchsorted(
            sorted_values,
            raw,
            side="right",
        ) - 1

        percentile = rank / (len(sorted_values) - 1)

        scene["motion_score"] = round(
            float(np.clip(percentile, 0.0, 1.0)),
            3,
        )

        del scene["raw_motion_score"]


def analyze_video(
    video_path: str,
    output_json: str,
    frames_dir: str,
) -> dict[str, Any]:

    video_path = str(Path(video_path))

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        raise RuntimeError("Could not determine video FPS.")

    duration = total_frames / fps

    print("Analyzing video...")
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f}s")
    print("\nRunning AdaptiveDetector...")

    # AdaptiveDetector is intended to handle cases where raw
    # frame differences can be caused by camera motion.
    scenes = detect(
        video_path,
        AdaptiveDetector(
            adaptive_threshold=3.0,
            min_scene_len=0.2,
            window_width=2,
            min_content_val=15.0,
        ),
        show_progress=True,
    )

    print(f"\nDetected {len(scenes)} scenes.")

    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    scene_data: list[dict[str, Any]] = []

    for index, (start_time, end_time) in enumerate(
        scenes,
        start=1,
    ):
        start_frame = start_time.frame_num
        end_frame = end_time.frame_num

        start_seconds = start_time.seconds
        end_seconds = end_time.seconds

        scene_duration = end_seconds - start_seconds

        middle_frame = (start_frame + end_frame) // 2

        raw_motion = calculate_motion_score(
            video_path,
            start_frame,
            end_frame,
        )

        frame_path = frames_dir / f"scene_{index:03d}.jpg"

        extract_representative_frame(
            video_path,
            middle_frame,
            frame_path,
        )

        scene_data.append(
            {
                "scene_id": index,
                "start": round(start_seconds, 3),
                "end": round(end_seconds, 3),
                "duration": round(scene_duration, 3),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "representative_frame": str(frame_path),
                "raw_motion_score": raw_motion,
            }
        )

    # Normalize motion relative to this particular reference video.
    normalize_motion_scores(scene_data)

    durations = [
        scene["duration"]
        for scene in scene_data
    ]

    average_shot_duration = (
        float(np.mean(durations))
        if durations
        else 0.0
    )

    median_shot_duration = (
        float(np.median(durations))
        if durations
        else 0.0
    )

    cut_frequency = (
        len(scene_data) / duration
        if duration > 0
        else 0.0
    )

    short_shots = [
        d for d in durations
        if d < 0.5
    ]

    medium_shots = [
        d for d in durations
        if 0.5 <= d < 1.5
    ]

    long_shots = [
        d for d in durations
        if d >= 1.5
    ]

    high_motion_scenes = [
        scene
        for scene in scene_data
        if scene["motion_score"] >= 0.75
    ]

    extreme_motion_scenes = [
        scene
        for scene in scene_data
        if scene["motion_score"] >= 0.90
    ]

    analysis = {
        "video": {
            "path": video_path,
            "duration_seconds": round(duration, 3),
            "fps": round(fps, 3),
            "total_frames": total_frames,
            "width": width,
            "height": height,
        },

        "detection": {
            "detector": "AdaptiveDetector",
            "adaptive_threshold": 3.0,
            "min_scene_length_seconds": 0.2,
            "window_width": 2,
            "min_content_value": 15.0,
        },

        "style": {
            "scene_count": len(scene_data),
            "average_shot_duration": round(
                average_shot_duration,
                3,
            ),
            "median_shot_duration": round(
                median_shot_duration,
                3,
            ),
            "cut_frequency_per_second": round(
                cut_frequency,
                3,
            ),
            "short_shot_ratio": round(
                len(short_shots) / len(durations),
                3,
            ) if durations else 0.0,
            "medium_shot_ratio": round(
                len(medium_shots) / len(durations),
                3,
            ) if durations else 0.0,
            "long_shot_ratio": round(
                len(long_shots) / len(durations),
                3,
            ) if durations else 0.0,
            "high_motion_scene_ratio": round(
                len(high_motion_scenes) / len(scene_data),
                3,
            ) if scene_data else 0.0,
            "extreme_motion_scene_ratio": round(
                len(extreme_motion_scenes) / len(scene_data),
                3,
            ) if scene_data else 0.0,
        },

        "scenes": scene_data,
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            analysis,
            file,
            indent=2,
        )

    print("\n✅ Reference analysis complete.")
    print(f"JSON: {output_path}")
    print(f"Frames: {frames_dir}")


if __name__ == "__main__":
    analyze_video(
        video_path="data/references/reference.mp4",
        output_json="data/outputs/reference_analysis.json",
        frames_dir="data/outputs/reference_frames",
    )