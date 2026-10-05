from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.signal import find_peaks


VIDEO_PATH = "data/references/reference_full.mp4"
OUTPUT_PATH = "data/outputs/visual_events.json"


def robust_normalize(values: np.ndarray) -> np.ndarray:
    """
    Convert a signal into a smooth 0-1 score.

    Used for ranking event strength, NOT for semantic
    classification.
    """

    if len(values) == 0:
        return values.astype(np.float32)

    median = float(np.median(values))

    mad = float(
        np.median(
            np.abs(values - median)
        )
    )

    scale = max(
        1.4826 * mad,
        1e-6,
    )

    robust_z = (values - median) / scale

    normalized = 1.0 / (
        1.0 + np.exp(-robust_z / 2.0)
    )

    return normalized.astype(np.float32)


def percentile_threshold(
    values: np.ndarray,
    percentile: float,
    minimum: float = 0.0,
) -> float:
    """
    Calculate a threshold relative to this video's own signal.

    This is more robust than hardcoding one threshold for
    every anime/video.
    """

    if len(values) == 0:
        return minimum

    threshold = float(
        np.percentile(
            values,
            percentile,
        )
    )

    return max(
        threshold,
        minimum,
    )


def classify_event(
    raw_motion: float,
    raw_brightness: float,
    raw_edges: float,
    motion_threshold: float,
    brightness_threshold: float,
    edge_threshold: float,
) -> str:
    """
    Objective V0.3 event classification.

    We deliberately avoid claiming that CV alone knows
    whether something is a punch, explosion, camera shake,
    etc. Those semantic labels come later from AI.
    """

    high_motion = (
        raw_motion >= motion_threshold
    )

    high_brightness = (
        raw_brightness >= brightness_threshold
    )

    high_edges = (
        raw_edges >= edge_threshold
    )

    if high_brightness and high_motion:
        return "compound_brightness_motion"

    if high_brightness:
        return "brightness_change"

    if high_motion and high_edges:
        return "high_motion_structural"

    if high_motion:
        return "motion_spike"

    if high_edges:
        return "structural_change"

    return "visual_change"


def analyze_visual_events(
    video_path: str,
    output_path: str,
    sample_fps: float = 15.0,
) -> dict[str, Any]:

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        cap.release()
        raise RuntimeError(
            "Could not determine video FPS."
        )

    duration = frame_count / fps

    print("Analyzing visual micro-events...")
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f}s")
    print(
        f"Analysis sampling: "
        f"{sample_fps:.1f} FPS"
    )

    frame_step = max(
        1,
        int(round(fps / sample_fps)),
    )

    times: list[float] = []

    raw_motion_scores: list[float] = []
    raw_brightness_changes: list[float] = []
    raw_edge_changes: list[float] = []

    previous_gray: np.ndarray | None = None
    previous_brightness: float | None = None
    previous_edge_density: float | None = None

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_index % frame_step != 0:
            frame_index += 1
            continue

        # -----------------------------------------------------
        # Downscale for speed.
        # -----------------------------------------------------

        small = cv2.resize(
            frame,
            (320, 180),
            interpolation=cv2.INTER_AREA,
        )

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2GRAY,
        )

        # -----------------------------------------------------
        # Brightness
        # -----------------------------------------------------

        brightness = float(
            np.mean(gray)
        )

        # -----------------------------------------------------
        # Edge density
        # -----------------------------------------------------

        edges = cv2.Canny(
            gray,
            80,
            160,
        )

        edge_density = float(
            np.mean(edges > 0)
        )

        # -----------------------------------------------------
        # Compare with previous frame.
        # -----------------------------------------------------

        if previous_gray is None:

            motion = 0.0
            brightness_change = 0.0
            edge_change = 0.0

        else:

            frame_difference = cv2.absdiff(
                previous_gray,
                gray,
            )

            motion = float(
                np.mean(frame_difference)
                / 255.0
            )

            brightness_change = abs(
                brightness
                - float(previous_brightness)
            ) / 255.0

            edge_change = abs(
                edge_density
                - float(previous_edge_density)
            )

        time = frame_index / fps

        times.append(time)

        raw_motion_scores.append(
            motion
        )

        raw_brightness_changes.append(
            brightness_change
        )

        raw_edge_changes.append(
            edge_change
        )

        previous_gray = gray
        previous_brightness = brightness
        previous_edge_density = edge_density

        frame_index += 1

    cap.release()

    if not times:
        raise RuntimeError(
            "No frames were analyzed."
        )

    # ---------------------------------------------------------
    # Raw arrays.
    # ---------------------------------------------------------

    raw_motion = np.asarray(
        raw_motion_scores,
        dtype=np.float32,
    )

    raw_brightness = np.asarray(
        raw_brightness_changes,
        dtype=np.float32,
    )

    raw_edges = np.asarray(
        raw_edge_changes,
        dtype=np.float32,
    )

    # ---------------------------------------------------------
    # Normalized scores for ranking.
    # ---------------------------------------------------------

    motion = robust_normalize(
        raw_motion
    )

    brightness_change = robust_normalize(
        raw_brightness
    )

    edge_change = robust_normalize(
        raw_edges
    )

    # ---------------------------------------------------------
    # Event strength.
    # ---------------------------------------------------------

    event_strength = (
        0.50 * motion
        + 0.30 * brightness_change
        + 0.20 * edge_change
    )

    # ---------------------------------------------------------
    # Adaptive semantic thresholds.
    #
    # These are based on this video's own distribution.
    # ---------------------------------------------------------

    motion_threshold = percentile_threshold(
        raw_motion,
        90,
        minimum=0.02,
    )

    brightness_threshold = percentile_threshold(
        raw_brightness,
        95,
        minimum=0.03,
    )

    edge_threshold = percentile_threshold(
        raw_edges,
        90,
        minimum=0.005,
    )

    print("\nAdaptive thresholds:")
    print(
        f"Motion:     {motion_threshold:.4f}"
    )
    print(
        f"Brightness: {brightness_threshold:.4f}"
    )
    print(
        f"Edges:      {edge_threshold:.4f}"
    )

    # ---------------------------------------------------------
    # Find local peaks in visual event strength.
    # ---------------------------------------------------------

    min_distance = max(
        1,
        int(
            round(
                sample_fps * 0.20
            )
        ),
    )

    peaks, _ = find_peaks(
        event_strength,
        distance=min_distance,
        prominence=0.08,
    )

    events: list[dict[str, Any]] = []

    for peak in peaks:

        strength = float(
            event_strength[peak]
        )

        if strength < 0.35:
            continue

        raw_motion_value = float(
            raw_motion[peak]
        )

        raw_brightness_value = float(
            raw_brightness[peak]
        )

        raw_edge_value = float(
            raw_edges[peak]
        )

        event_type = classify_event(
            raw_motion=raw_motion_value,
            raw_brightness=raw_brightness_value,
            raw_edges=raw_edge_value,
            motion_threshold=motion_threshold,
            brightness_threshold=brightness_threshold,
            edge_threshold=edge_threshold,
        )

        events.append(
            {
                "time": round(
                    float(times[peak]),
                    3,
                ),

                "strength": round(
                    strength,
                    3,
                ),

                # Normalized signals for comparison/ranking.
                "motion_score": round(
                    float(motion[peak]),
                    3,
                ),

                "brightness_score": round(
                    float(
                        brightness_change[peak]
                    ),
                    3,
                ),

                "edge_score": round(
                    float(
                        edge_change[peak]
                    ),
                    3,
                ),

                # Raw signals for diagnostics and
                # future model training.
                "raw_motion": round(
                    raw_motion_value,
                    5,
                ),

                "raw_brightness_change": round(
                    raw_brightness_value,
                    5,
                ),

                "raw_edge_change": round(
                    raw_edge_value,
                    5,
                ),

                "event_type": event_type,
            }
        )

    strongest_events = sorted(
        events,
        key=lambda event: event["strength"],
        reverse=True,
    )

    result = {
        "video": {
            "path": video_path,
            "fps": round(
                float(fps),
                3,
            ),
            "duration_seconds": round(
                float(duration),
                3,
            ),
            "sample_fps": sample_fps,
        },

        "thresholds": {
            "motion": round(
                motion_threshold,
                6,
            ),
            "brightness": round(
                brightness_threshold,
                6,
            ),
            "edges": round(
                edge_threshold,
                6,
            ),
        },

        "analysis": {
            "samples": len(times),
            "candidate_event_count": len(events),
        },

        "events": events,

        "strongest_events": (
            strongest_events[:30]
        ),
    }

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        f"\nDetected {len(events)} "
        "candidate visual events."
    )

    print(
        f"JSON: {output}"
    )

    return result


if __name__ == "__main__":
    analyze_visual_events(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
    )