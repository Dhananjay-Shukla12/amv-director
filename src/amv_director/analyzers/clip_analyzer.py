from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CLIPS_DIR = PROJECT_ROOT / "data" / "clips"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "clip_analysis"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def safe_float(value, default=0.0):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return default


def normalize_score(value, low, high):
    if high - low < 1e-9:
        return 0.0

    score = (value - low) / (high - low)
    return float(np.clip(score, 0.0, 1.0))


def calculate_frame_metrics(frame, previous_gray=None):
    """
    Calculate measurable visual properties from one frame.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Brightness
    brightness = float(np.mean(gray))

    # Contrast
    contrast = float(np.std(gray))

    # Blur estimate.
    # Higher variance of Laplacian generally means a sharper frame.
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    # Frame-to-frame motion
    motion = 0.0

    if previous_gray is not None:
        diff = cv2.absdiff(gray, previous_gray)
        motion = float(np.mean(diff)) / 255.0

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": blur_score,
        "edge_density": edge_density,
        "motion": motion,
        "gray": gray,
    }


def calculate_video_metrics(video_path: Path):
    """
    Sample the video and calculate global visual statistics.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    duration = frame_count / fps if frame_count > 0 else 0.0

    # Sample approximately 8 frames/sec.
    sample_fps = min(8.0, fps)
    sample_every = max(1, int(round(fps / sample_fps)))

    brightness_values = []
    contrast_values = []
    sharpness_values = []
    edge_values = []
    motion_values = []

    previous_gray = None
    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_index % sample_every != 0:
            frame_index += 1
            continue

        metrics = calculate_frame_metrics(frame, previous_gray)

        previous_gray = metrics["gray"]

        brightness_values.append(metrics["brightness"])
        contrast_values.append(metrics["contrast"])
        sharpness_values.append(metrics["sharpness"])
        edge_values.append(metrics["edge_density"])
        motion_values.append(metrics["motion"])

        frame_index += 1

    cap.release()

    if not brightness_values:
        raise RuntimeError(f"No frames could be sampled from {video_path}")

    return {
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
        "brightness_mean": float(np.mean(brightness_values)),
        "brightness_std": float(np.std(brightness_values)),
        "contrast_mean": float(np.mean(contrast_values)),
        "sharpness_mean": float(np.mean(sharpness_values)),
        "edge_density_mean": float(np.mean(edge_values)),
        "motion_mean": float(np.mean(motion_values)),
        "motion_max": float(np.max(motion_values)),
        "sample_count": len(brightness_values),
    }


def detect_scenes(video_path: Path):
    """
    Detect shot/scene boundaries using the same general
    philosophy as the reference analysis.
    """

    video = open_video(str(video_path))

    scene_manager = SceneManager()

    detector = AdaptiveDetector(
        adaptive_threshold=3.0,
        min_scene_len=6,
        window_width=2,
        min_content_val=15.0,
    )

    scene_manager.add_detector(detector)
    scene_manager.detect_scenes(video)

    scenes = scene_manager.get_scene_list()

    results = []

    for index, (start, end) in enumerate(scenes, start=1):
        start_seconds = start.get_seconds()
        end_seconds = end.get_seconds()
        duration = end_seconds - start_seconds

        results.append(
            {
                "scene_id": index,
                "start": round(start_seconds, 3),
                "end": round(end_seconds, 3),
                "duration": round(duration, 3),
            }
        )

    return results


def extract_scene_features(video_path: Path, scenes):
    """
    Extract representative frame + visual features for each scene.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 30.0)

    enriched = []

    for scene in scenes:
        start = scene["start"]
        end = scene["end"]

        # Three temporal samples.
        sample_times = [
            start,
            start + (end - start) * 0.5,
            max(start, end - 0.05),
        ]

        brightness = []
        contrast = []
        sharpness = []
        edges = []

        previous_gray = None
        motion = []

        for timestamp in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)

            ok, frame = cap.read()

            if not ok:
                continue

            metrics = calculate_frame_metrics(frame, previous_gray)

            brightness.append(metrics["brightness"])
            contrast.append(metrics["contrast"])
            sharpness.append(metrics["sharpness"])
            edges.append(metrics["edge_density"])

            if previous_gray is not None:
                motion.append(metrics["motion"])

            previous_gray = metrics["gray"]

        if not brightness:
            continue

        scene_copy = dict(scene)

        scene_copy.update(
            {
                "brightness": round(float(np.mean(brightness)), 3),
                "contrast": round(float(np.mean(contrast)), 3),
                "sharpness": round(float(np.mean(sharpness)), 3),
                "edge_density": round(float(np.mean(edges)), 4),
                "motion": round(
                    float(np.mean(motion)) if motion else 0.0,
                    4,
                ),
            }
        )

        enriched.append(scene_copy)

    cap.release()

    return enriched


def classify_scene(scene):
    """
    Convert raw measurements into useful editing-oriented labels.

    These are descriptive labels based on measurable signals,
    not semantic understanding of the anime plot.
    """

    duration = scene["duration"]
    motion = scene["motion"]
    sharpness = scene["sharpness"]
    brightness = scene["brightness"]
    contrast = scene["contrast"]

    labels = []

    # Pacing role
    if duration < 0.5:
        labels.append("very_fast")
    elif duration < 1.2:
        labels.append("fast")
    elif duration < 2.5:
        labels.append("medium")
    elif duration < 5.0:
        labels.append("long")
    else:
        labels.append("very_long")

    # Motion
    if motion >= 0.18:
        labels.append("high_motion")
    elif motion >= 0.08:
        labels.append("medium_motion")
    else:
        labels.append("low_motion")

    # Visual intensity
    if contrast >= 65 and motion >= 0.12:
        labels.append("high_visual_intensity")
    elif contrast >= 45:
        labels.append("medium_visual_intensity")
    else:
        labels.append("low_visual_intensity")

    # Potential calm/establishing material
    if duration >= 3.5 and motion < 0.08:
        labels.append("calm_candidate")

    # Potential impact material
    if motion >= 0.18 and duration <= 3.0:
        labels.append("impact_candidate")

    # Sharpness
    if sharpness < 50:
        labels.append("soft_or_blurred")
    elif sharpness > 250:
        labels.append("sharp")

    # Brightness
    if brightness < 55:
        labels.append("dark")
    elif brightness > 190:
        labels.append("bright")

    return labels


def calculate_global_percentiles(scenes):
    """
    Calculate dataset-level ranges so classifications are
    relative to the user's actual footage.
    """

    if not scenes:
        return {}

    motions = np.array([s["motion"] for s in scenes], dtype=float)
    brightness = np.array([s["brightness"] for s in scenes], dtype=float)
    contrast = np.array([s["contrast"] for s in scenes], dtype=float)
    sharpness = np.array([s["sharpness"] for s in scenes], dtype=float)

    return {
        "motion_p25": float(np.percentile(motions, 25)),
        "motion_p75": float(np.percentile(motions, 75)),
        "brightness_p25": float(np.percentile(brightness, 25)),
        "brightness_p75": float(np.percentile(brightness, 75)),
        "contrast_p25": float(np.percentile(contrast, 25)),
        "contrast_p75": float(np.percentile(contrast, 75)),
        "sharpness_p25": float(np.percentile(sharpness, 25)),
        "sharpness_p75": float(np.percentile(sharpness, 75)),
    }


def assign_relative_scores(scene, ranges):
    """
    Convert raw metrics into 0-1 relative scores.

    Useful later for clip retrieval.
    """

    motion_score = normalize_score(
        scene["motion"],
        ranges["motion_p25"],
        ranges["motion_p75"],
    )

    brightness_score = normalize_score(
        scene["brightness"],
        ranges["brightness_p25"],
        ranges["brightness_p75"],
    )

    contrast_score = normalize_score(
        scene["contrast"],
        ranges["contrast_p25"],
        ranges["contrast_p75"],
    )

    sharpness_score = normalize_score(
        scene["sharpness"],
        ranges["sharpness_p25"],
        ranges["sharpness_p75"],
    )

    scene["scores"] = {
        "motion": round(motion_score, 3),
        "brightness": round(brightness_score, 3),
        "contrast": round(contrast_score, 3),
        "sharpness": round(sharpness_score, 3),
    }

    return scene


def analyze_video(video_path: Path):
    print(f"\n🎬 Analyzing: {video_path.name}")

    metrics = calculate_video_metrics(video_path)

    print(
        f"   Duration: {metrics['duration']:.2f}s | "
        f"FPS: {metrics['fps']:.2f} | "
        f"Frames: {metrics['frame_count']}"
    )

    scenes = detect_scenes(video_path)

    print(f"   Scenes detected: {len(scenes)}")

    scenes = extract_scene_features(video_path, scenes)

    ranges = calculate_global_percentiles(scenes)

    for scene in scenes:
        scene["labels"] = classify_scene(scene)
        assign_relative_scores(scene, ranges)

    result = {
        "video": video_path.name,
        "path": str(video_path),
        "video_metrics": metrics,
        "scene_count": len(scenes),
        "relative_ranges": ranges,
        "scenes": scenes,
    }

    output_file = OUTPUT_DIR / f"{video_path.stem}_analysis.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"   ✅ Saved: {output_file}")

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        [
            p
            for p in CLIPS_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]
    )

    if not videos:
        print(
            "\n❌ No videos found.\n"
            f"Put anime videos inside:\n{CLIPS_DIR}\n"
        )
        sys.exit(1)

    print(f"Found {len(videos)} video(s).")

    all_results = []

    for video in videos:
        try:
            result = analyze_video(video)
            all_results.append(result)
        except Exception as exc:
            print(f"   ❌ Failed: {exc}")

    combined_file = OUTPUT_DIR / "all_clips_analysis.json"

    with combined_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n========================================")
    print("✅ CLIP ANALYSIS COMPLETE")
    print("========================================")
    print(f"Videos analyzed: {len(all_results)}")
    print(f"Combined output: {combined_file}")

    total_scenes = sum(r["scene_count"] for r in all_results)
    print(f"Total candidate scenes: {total_scenes}")


if __name__ == "__main__":
    main()