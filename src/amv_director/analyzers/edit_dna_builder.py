from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Any


VIDEO_ANALYSIS = "data/outputs/reference_analysis.json"
MUSIC_ANALYSIS = "data/outputs/music_analysis.json"
VISUAL_EVENTS = "data/outputs/visual_events.json"

OUTPUT_PATH = "data/outputs/edit_dna.json"


def load_json(path: str) -> dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def nearest_value(
    target: float,
    values: list[float],
) -> tuple[float | None, float | None]:
    """
    Find the nearest timestamp and its absolute distance.
    """

    if not values:
        return None, None

    index = bisect_left(values, target)

    candidates: list[float] = []

    if index < len(values):
        candidates.append(values[index])

    if index > 0:
        candidates.append(values[index - 1])

    nearest = min(
        candidates,
        key=lambda value: abs(value - target),
    )

    return nearest, abs(nearest - target)


def find_scene_for_time(
    time: float,
    scenes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the hard-cut scene containing a timestamp."""

    for scene in scenes:

        if (
            scene["start"]
            <= time
            < scene["end"]
        ):
            return scene

    # Handle an event exactly at the video's end.
    if scenes and time >= scenes[-1]["start"]:
        return scenes[-1]

    return None


def classify_music_relationship(
    beat_distance: float | None,
    onset_distance: float | None,
) -> str:
    """
    Describe how closely a visual event aligns with music.

    Thresholds are deliberately conservative.
    """

    beat_ms = (
        beat_distance * 1000
        if beat_distance is not None
        else None
    )

    onset_ms = (
        onset_distance * 1000
        if onset_distance is not None
        else None
    )

    if beat_ms is not None and beat_ms <= 50:
        return "beat_locked"

    if onset_ms is not None and onset_ms <= 50:
        return "onset_locked"

    if beat_ms is not None and beat_ms <= 100:
        return "near_beat"

    if onset_ms is not None and onset_ms <= 100:
        return "near_onset"

    if beat_ms is not None and beat_ms <= 200:
        return "loosely_music_aligned"

    if onset_ms is not None and onset_ms <= 200:
        return "loosely_music_aligned"

    return "not_strongly_music_aligned"


def shot_position(
    event_time: float,
    scene: dict[str, Any],
) -> dict[str, Any]:

    scene_start = float(scene["start"])
    scene_end = float(scene["end"])
    scene_duration = float(scene["duration"])

    relative_time = event_time - scene_start

    normalized_position = (
        relative_time / scene_duration
        if scene_duration > 0
        else 0.0
    )

    if normalized_position < 0.20:
        position = "beginning"

    elif normalized_position < 0.80:
        position = "middle"

    else:
        position = "ending"

    return {
        "relative_time": round(
            relative_time,
            3,
        ),
        "normalized_position": round(
            max(
                0.0,
                min(
                    1.0,
                    normalized_position,
                ),
            ),
            3,
        ),
        "position": position,
    }


def closest_hard_cut(
    event_time: float,
    scenes: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    """
    Find the nearest actual hard-cut boundary.

    Scene starts after 0.0 represent hard visual cuts.
    """

    cut_times = [
        float(scene["start"])
        for scene in scenes
        if float(scene["start"]) > 0.001
    ]

    return nearest_value(
        event_time,
        cut_times,
    )


def build_edit_dna() -> dict[str, Any]:

    video = load_json(
        VIDEO_ANALYSIS
    )

    music = load_json(
        MUSIC_ANALYSIS
    )

    visual = load_json(
        VISUAL_EVENTS
    )

    scenes = video["scenes"]
    visual_events = visual["events"]
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

    edit_events: list[dict[str, Any]] = []

    print("Building Edit DNA...")

    for event in visual_events:

        event_time = float(
            event["time"]
        )

        nearest_beat, beat_distance = nearest_value(
            event_time,
            beat_times,
        )

        nearest_onset, onset_distance = nearest_value(
            event_time,
            onset_times,
        )

        nearest_energy_time, energy_distance = nearest_value(
            event_time,
            energy_times,
        )

        energy = None

        if nearest_energy_time is not None:

            # Find closest energy sample.
            energy_index = min(
                range(len(energy_times)),
                key=lambda i: abs(
                    energy_times[i]
                    - nearest_energy_time
                ),
            )

            energy = energy_values[
                energy_index
            ]

        scene = find_scene_for_time(
            event_time,
            scenes,
        )

        cut_time, cut_distance = closest_hard_cut(
            event_time,
            scenes,
        )

        music_relationship = classify_music_relationship(
            beat_distance,
            onset_distance,
        )

        event_data: dict[str, Any] = {
            "time": round(
                event_time,
                3,
            ),

            "visual": {
                "event_type": event["event_type"],
                "strength": event["strength"],
                "motion_score": event["motion_score"],
                "brightness_score": event["brightness_score"],
                "edge_score": event["edge_score"],
            },

            "music": {
                "nearest_beat": (
                    round(
                        nearest_beat,
                        3,
                    )
                    if nearest_beat is not None
                    else None
                ),

                "beat_distance_ms": (
                    round(
                        beat_distance * 1000,
                        1,
                    )
                    if beat_distance is not None
                    else None
                ),

                "nearest_onset": (
                    round(
                        nearest_onset,
                        3,
                    )
                    if nearest_onset is not None
                    else None
                ),

                "onset_distance_ms": (
                    round(
                        onset_distance * 1000,
                        1,
                    )
                    if onset_distance is not None
                    else None
                ),

                "energy": (
                    round(
                        energy,
                        3,
                    )
                    if energy is not None
                    else None
                ),

                "relationship": music_relationship,
            },
        }

        if scene is not None:

            event_data["shot"] = {
                "scene_id": scene["scene_id"],
                "scene_start": scene["start"],
                "scene_end": scene["end"],
                "scene_duration": scene["duration"],
                "shot_type": (
                    "very_fast"
                    if scene["duration"] < 0.5
                    else "fast"
                    if scene["duration"] < 1.0
                    else "medium"
                    if scene["duration"] < 2.0
                    else "long"
                    if scene["duration"] < 4.0
                    else "very_long"
                ),
                "position": shot_position(
                    event_time,
                    scene,
                ),
            }

        else:

            event_data["shot"] = None

        event_data["nearest_hard_cut"] = {
            "time": (
                round(
                    cut_time,
                    3,
                )
                if cut_time is not None
                else None
            ),

            "distance_ms": (
                round(
                    cut_distance * 1000,
                    1,
                )
                if cut_distance is not None
                else None
            ),
        }

        edit_events.append(
            event_data
        )

    # ---------------------------------------------------------
    # GLOBAL STATISTICS
    # ---------------------------------------------------------

    total_events = len(edit_events)

    beat_locked = sum(
        1
        for event in edit_events
        if event["music"]["relationship"]
        == "beat_locked"
    )

    onset_locked = sum(
        1
        for event in edit_events
        if event["music"]["relationship"]
        == "onset_locked"
    )

    near_beat = sum(
        1
        for event in edit_events
        if event["music"]["relationship"]
        == "near_beat"
    )

    near_onset = sum(
        1
        for event in edit_events
        if event["music"]["relationship"]
        == "near_onset"
    )

    strongly_music_aligned = (
        beat_locked
        + onset_locked
    )

    music_aligned_within_100ms = (
        beat_locked
        + onset_locked
        + near_beat
        + near_onset
    )

    event_types: dict[str, int] = {}

    for event in edit_events:

        event_type = event["visual"][
            "event_type"
        ]

        event_types[event_type] = (
            event_types.get(
                event_type,
                0,
            )
            + 1
        )

    durations = [
        float(scene["duration"])
        for scene in scenes
    ]

    short_shots = sum(
        1
        for duration in durations
        if duration < 0.5
    )

    fast_shots = sum(
        1
        for duration in durations
        if 0.5 <= duration < 1.0
    )

    medium_shots = sum(
        1
        for duration in durations
        if 1.0 <= duration < 2.0
    )

    long_shots = sum(
        1
        for duration in durations
        if duration >= 2.0
    )

    style_profile = {
        "video_duration_seconds": video[
            "video"
        ]["duration_seconds"],

        "audio_duration_seconds": music[
            "audio"
        ]["duration_seconds"],

        "bpm": music[
            "tempo"
        ]["bpm"],

        "hard_scene_count": len(scenes),

        "micro_event_count": total_events,

        "hard_cut_frequency": (
            video["style"][
                "cut_frequency_per_second"
            ]
        ),

        "average_shot_duration": (
            video["style"][
                "average_shot_duration"
            ]
        ),

        "median_shot_duration": (
            video["style"][
                "median_shot_duration"
            ]
        ),

        "shot_distribution": {
            "very_fast_under_0_5s": short_shots,
            "fast_0_5_to_1s": fast_shots,
            "medium_1_to_2s": medium_shots,
            "long_over_2s": long_shots,
        },

        "music_alignment": {
            "beat_locked": beat_locked,
            "onset_locked": onset_locked,
            "beat_locked_ratio": round(
                beat_locked / total_events,
                3,
            ) if total_events else 0.0,
            "onset_locked_ratio": round(
                onset_locked / total_events,
                3,
            ) if total_events else 0.0,
            "strong_music_alignment_ratio": round(
                strongly_music_aligned
                / total_events,
                3,
            ) if total_events else 0.0,
            "within_100ms_ratio": round(
                music_aligned_within_100ms
                / total_events,
                3,
            ) if total_events else 0.0,
        },

        "visual_event_types": event_types,
    }

    result = {
        "project": "AMV Director",

        "version": "edit_dna_v0.1",

        "style_profile": style_profile,

        "event_timeline": edit_events,
    }

    output = Path(
        OUTPUT_PATH
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        "\n✅ Edit DNA generated."
    )

    print(
        f"Events: {total_events}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    return result


if __name__ == "__main__":
    build_edit_dna()