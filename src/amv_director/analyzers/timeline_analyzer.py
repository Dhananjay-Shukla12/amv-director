from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Any


VIDEO_ANALYSIS = "data/outputs/reference_analysis.json"
MUSIC_ANALYSIS = "data/outputs/music_analysis.json"
OUTPUT_PATH = "data/outputs/reference_timeline.json"


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def nearest_event(
    time: float,
    event_times: list[float],
) -> tuple[float | None, float | None]:
    """
    Return nearest event time and absolute distance.
    """

    if not event_times:
        return None, None

    index = bisect_left(event_times, time)

    candidates = []

    if index < len(event_times):
        candidates.append(event_times[index])

    if index > 0:
        candidates.append(event_times[index - 1])

    nearest = min(
        candidates,
        key=lambda x: abs(x - time),
    )

    return nearest, abs(nearest - time)


def classify_shot(duration: float) -> str:
    if duration < 0.5:
        return "very_fast"

    if duration < 1.0:
        return "fast"

    if duration < 2.0:
        return "medium"

    if duration < 4.0:
        return "long"

    return "very_long"


def alignment_label(distance: float | None) -> str:
    if distance is None:
        return "unknown"

    milliseconds = distance * 1000

    if milliseconds <= 40:
        return "excellent"

    if milliseconds <= 80:
        return "strong"

    if milliseconds <= 150:
        return "moderate"

    if milliseconds <= 250:
        return "weak"

    return "none"


def analyze_timeline(
    video_path: str,
    music_path: str,
    output_path: str,
) -> dict[str, Any]:

    video = load_json(video_path)
    music = load_json(music_path)

    scenes = video["scenes"]
    beats = music["beats"]
    onsets = music["onsets"]
    energy_curve = music["energy_curve"]

    beat_times = [
        float(item["time"])
        for item in beats
    ]

    onset_times = [
        float(item["time"])
        for item in onsets
    ]

    energy_times = [
        float(item["time"])
        for item in energy_curve
    ]

    energy_values = [
        float(item["energy"])
        for item in energy_curve
    ]

    timeline: list[dict[str, Any]] = []

    print("\nAnalyzing visual/music alignment...")

    for scene in scenes:

        start = float(scene["start"])
        end = float(scene["end"])
        duration = float(scene["duration"])

        nearest_beat, beat_distance = nearest_event(
            start,
            beat_times,
        )

        nearest_onset, onset_distance = nearest_event(
            start,
            onset_times,
        )

        # Find approximately matching energy value.
        nearest_energy, energy_distance = nearest_event(
            start,
            energy_times,
        )

        energy = None

        if nearest_energy is not None:
            index = energy_times.index(nearest_energy)
            energy = energy_values[index]

        timeline.append(
            {
                "scene_id": scene["scene_id"],
                "start": start,
                "end": end,
                "duration": duration,
                "shot_type": classify_shot(duration),
                "motion_score": scene["motion_score"],

                "nearest_beat": {
                    "time": nearest_beat,
                    "distance_seconds": (
                        round(beat_distance, 4)
                        if beat_distance is not None
                        else None
                    ),
                    "distance_ms": (
                        round(beat_distance * 1000, 1)
                        if beat_distance is not None
                        else None
                    ),
                    "alignment": alignment_label(
                        beat_distance
                    ),
                },

                "nearest_onset": {
                    "time": nearest_onset,
                    "distance_seconds": (
                        round(onset_distance, 4)
                        if onset_distance is not None
                        else None
                    ),
                    "distance_ms": (
                        round(onset_distance * 1000, 1)
                        if onset_distance is not None
                        else None
                    ),
                    "alignment": alignment_label(
                        onset_distance
                    ),
                },

                "music_energy": (
                    round(energy, 3)
                    if energy is not None
                    else None
                ),
            }
        )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    beat_aligned_count = sum(
        1
        for item in timeline
        if item["nearest_beat"]["distance_ms"] is not None
        and item["nearest_beat"]["distance_ms"] <= 80
    )

    onset_aligned_count = sum(
        1
        for item in timeline
        if item["nearest_onset"]["distance_ms"] is not None
        and item["nearest_onset"]["distance_ms"] <= 80
    )

    total = len(timeline)

    beat_alignment_ratio = (
        beat_aligned_count / total
        if total
        else 0.0
    )

    onset_alignment_ratio = (
        onset_aligned_count / total
        if total
        else 0.0
    )

    short_shots = sum(
        1
        for item in timeline
        if item["shot_type"] in {
            "very_fast",
            "fast",
        }
    )

    high_motion = sum(
        1
        for item in timeline
        if item["motion_score"] >= 0.75
    )

    result = {
        "reference": {
            "video_duration": video["video"]["duration_seconds"],
            "audio_duration": music["audio"]["duration_seconds"],
        },

        "music": {
            "bpm": music["tempo"]["bpm"],
            "beats": len(beats),
            "onsets": len(onsets),
            "energy_peaks": len(
                music["energy_peaks"]
            ),
        },

        "alignment": {
            "visual_events": total,

            "beat_aligned_within_80ms": (
                beat_aligned_count
            ),

            "beat_alignment_ratio": round(
                beat_alignment_ratio,
                3,
            ),

            "onset_aligned_within_80ms": (
                onset_aligned_count
            ),

            "onset_alignment_ratio": round(
                onset_alignment_ratio,
                3,
            ),

            "fast_visual_event_ratio": round(
                short_shots / total,
                3,
            ) if total else 0.0,

            "high_motion_event_ratio": round(
                high_motion / total,
                3,
            ) if total else 0.0,
        },

        "timeline": timeline,
    }

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print("\n✅ Timeline analysis complete.")
    print(f"JSON: {output}")

    return result


if __name__ == "__main__":
    analyze_timeline(
        video_path=VIDEO_ANALYSIS,
        music_path=MUSIC_ANALYSIS,
        output_path=OUTPUT_PATH,
    )