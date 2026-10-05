from __future__ import annotations

import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EDIT_DNA_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "edit_dna.json"
)

REFERENCE_TIMELINE_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "reference_timeline.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "reference_rhythm_map.json"
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def number(value, default=0.0):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return default


def clamp(value, low=0.0, high=1.0):
    return max(
        low,
        min(high, float(value)),
    )


def gaussian_score(
    distance_ms: float,
    scale_ms: float,
):
    """
    Converts timing distance into a smooth 0-1 alignment score.

    0 ms = 1.0
    larger distance = progressively lower score
    """

    distance_ms = abs(distance_ms)

    return math.exp(
        -(
            distance_ms ** 2
        ) /
        (
            2 *
            scale_ms ** 2
        )
    )


# ============================================================
# LOAD REFERENCE
# ============================================================

def load_reference():

    data = load_json(
        REFERENCE_TIMELINE_FILE
    )

    timeline = data.get(
        "timeline"
    )

    if not isinstance(
        timeline,
        list,
    ):
        raise ValueError(
            "reference_timeline.json does not contain timeline[]."
        )

    events = []

    for item in timeline:

        start = number(
            item.get("start")
        )

        end = number(
            item.get("end")
        )

        if end <= start:
            continue

        events.append(
            {
                "scene_id": item.get(
                    "scene_id"
                ),
                "start": start,
                "end": end,
                "duration": end - start,
                "shot_type": item.get(
                    "shot_type",
                    "",
                ),
            }
        )

    events.sort(
        key=lambda x: x["start"]
    )

    duration = max(
        event["end"]
        for event in events
    )

    return events, duration


# ============================================================
# LOAD EDIT DNA EVENTS
# ============================================================

def load_edit_events():

    data = load_json(
        EDIT_DNA_FILE
    )

    events = data.get(
        "event_timeline"
    )

    if not isinstance(
        events,
        list,
    ):
        raise ValueError(
            "edit_dna.json does not contain event_timeline[]."
        )

    return events


# ============================================================
# EVENT FEATURES
# ============================================================

def event_features(event):

    visual = event.get(
        "visual",
        {}
    )

    music = event.get(
        "music",
        {}
    )

    shot = event.get(
        "shot",
        {}
    )

    event_type = str(
        visual.get(
            "event_type",
            ""
        )
    )

    strength = number(
        visual.get(
            "strength"
        )
    )

    motion = number(
        visual.get(
            "motion_score"
        )
    )

    brightness = number(
        visual.get(
            "brightness_score"
        )
    )

    edge = number(
        visual.get(
            "edge_score"
        )
    )

    energy = number(
        music.get(
            "energy"
        )
    )

    beat_distance = number(
        music.get(
            "beat_distance_ms",
            9999,
        ),
        9999,
    )

    onset_distance = number(
        music.get(
            "onset_distance_ms",
            9999,
        ),
        9999,
    )

    beat_score = gaussian_score(
        beat_distance,
        120.0,
    )

    onset_score = gaussian_score(
        onset_distance,
        90.0,
    )

    # Visual event strength.
    visual_score = (
        strength * 0.45
        + motion * 0.20
        + brightness * 0.20
        + edge * 0.15
    )

    # Music alignment.
    music_score = (
        onset_score * 0.55
        + beat_score * 0.25
        + energy * 0.20
    )

    # Structural events deserve additional weight
    # when deciding possible cut points.
    structural_bonus = 0.0

    if event_type in {
        "structural_change",
        "high_motion_structural",
    }:
        structural_bonus = 0.20

    if event_type in {
        "compound_brightness_motion",
    }:
        structural_bonus = 0.10

    anchor_score = clamp(
        visual_score * 0.55
        + music_score * 0.30
        + structural_bonus * 0.15
    )

    return {
        "event_type": event_type,
        "strength": strength,
        "motion": motion,
        "brightness": brightness,
        "edge": edge,
        "energy": energy,
        "beat_distance_ms": beat_distance,
        "onset_distance_ms": onset_distance,
        "beat_score": beat_score,
        "onset_score": onset_score,
        "visual_score": visual_score,
        "music_score": music_score,
        "anchor_score": anchor_score,
        "scene_id": shot.get(
            "scene_id"
        ),
    }


# ============================================================
# ANCHOR CLASSIFICATION
# ============================================================

def classify_anchor(features):

    event_type = features[
        "event_type"
    ]

    strength = features[
        "strength"
    ]

    energy = features[
        "energy"
    ]

    onset_distance = features[
        "onset_distance_ms"
    ]

    beat_distance = features[
        "beat_distance_ms"
    ]

    if (
        event_type in {
            "structural_change",
            "high_motion_structural",
        }
        and
        strength >= 0.70
    ):
        return "cut_candidate"

    if (
        energy >= 0.65
        and
        strength >= 0.70
    ):
        return "impact_anchor"

    if (
        onset_distance <= 80
        and
        strength >= 0.65
    ):
        return "onset_anchor"

    if (
        beat_distance <= 80
        and
        strength >= 0.65
    ):
        return "beat_anchor"

    if (
        event_type ==
        "compound_brightness_motion"
        and
        strength >= 0.70
    ):
        return "visual_impact"

    return "visual_anchor"


# ============================================================
# BUILD CANDIDATES
# ============================================================

def build_candidates(events):

    candidates = []

    for event in events:

        time = number(
            event.get("time")
        )

        features = event_features(
            event
        )

        anchor_type = classify_anchor(
            features
        )

        candidate = {
            "time": round(
                time,
                3,
            ),
            "anchor_type": anchor_type,
            **{
                key: (
                    round(value, 4)
                    if isinstance(
                        value,
                        (float, int),
                    )
                    else value
                )
                for key, value
                in features.items()
            },
        }

        candidates.append(
            candidate
        )

    candidates.sort(
        key=lambda x: x["anchor_score"],
        reverse=True,
    )

    return candidates


# ============================================================
# NON-MAXIMUM SUPPRESSION
# ============================================================

def select_anchors(
    candidates,
    min_gap=0.35,
    max_anchors=32,
):
    """
    Avoid selecting several nearly identical events
    clustered around the same frame transition.
    """

    selected = []

    for candidate in candidates:

        time = candidate["time"]

        too_close = any(
            abs(
                time -
                existing["time"]
            ) < min_gap
            for existing in selected
        )

        if too_close:
            continue

        selected.append(
            candidate
        )

        if len(selected) >= max_anchors:
            break

    selected.sort(
        key=lambda x: x["time"]
    )

    return selected


# ============================================================
# HARD CUTS
# ============================================================

def build_hard_boundaries(
    reference_scenes,
):

    boundaries = []

    for index, scene in enumerate(
        reference_scenes
    ):

        boundaries.append(
            {
                "time": round(
                    scene["start"],
                    3,
                ),
                "type": "hard_boundary_start",
                "scene_id": scene[
                    "scene_id"
                ],
            }
        )

    if reference_scenes:

        boundaries.append(
            {
                "time": round(
                    reference_scenes[-1]["end"],
                    3,
                ),
                "type": "hard_boundary_end",
                "scene_id": reference_scenes[-1][
                    "scene_id"
                ],
            }
        )

    return boundaries


# ============================================================
# SUGGESTED CUTS
# ============================================================

def select_cut_candidates(
    anchors,
    min_gap=0.65,
):
    """
    Only a subset of visual events become suggested cuts.

    This intentionally avoids interpreting every visual event
    as a hard edit.
    """

    candidates = [
        item
        for item in anchors
        if item["anchor_type"]
        in {
            "cut_candidate",
            "impact_anchor",
        }
    ]

    candidates.sort(
        key=lambda x: x["anchor_score"],
        reverse=True,
    )

    selected = []

    for candidate in candidates:

        if any(
            abs(
                candidate["time"] -
                item["time"]
            ) < min_gap
            for item in selected
        ):
            continue

        selected.append(
            candidate
        )

    selected.sort(
        key=lambda x: x["time"]
    )

    return selected


# ============================================================
# SCENE ANCHORS
# ============================================================

def attach_anchors_to_scenes(
    reference_scenes,
    anchors,
):

    result = []

    for scene in reference_scenes:

        inside = [
            anchor
            for anchor in anchors
            if (
                scene["start"]
                <
                anchor["time"]
                <
                scene["end"]
            )
        ]

        inside.sort(
            key=lambda x: x["time"]
        )

        result.append(
            {
                "scene_id": scene[
                    "scene_id"
                ],
                "start": scene[
                    "start"
                ],
                "end": scene[
                    "end"
                ],
                "duration": scene[
                    "duration"
                ],
                "shot_type": scene[
                    "shot_type"
                ],
                "anchor_count": len(
                    inside
                ),
                "anchors": inside,
            }
        )

    return result


# ============================================================
# RHYTHM CHARACTER
# ============================================================

def characterize_slot(
    duration,
    anchors,
    start,
    end,
):

    events_in_slot = [
        anchor
        for anchor in anchors
        if start <= anchor["time"] < end
    ]

    density = (
        len(events_in_slot) /
        max(duration, 0.001)
    )

    max_energy = max(
        (
            anchor["energy"]
            for anchor in events_in_slot
        ),
        default=0.0,
    )

    max_strength = max(
        (
            anchor["strength"]
            for anchor in events_in_slot
        ),
        default=0.0,
    )

    if density >= 2.0:
        pacing = "very_fast"

    elif density >= 1.0:
        pacing = "fast"

    elif density >= 0.45:
        pacing = "medium"

    else:
        pacing = "slow"

    return {
        "pacing": pacing,
        "event_density": round(
            density,
            3,
        ),
        "max_energy": round(
            max_energy,
            3,
        ),
        "max_visual_strength": round(
            max_strength,
            3,
        ),
    }


# ============================================================
# BUILD RHYTHM SLOTS
# ============================================================

def build_rhythm_slots(
    duration,
    cut_candidates,
    anchors,
):

    cut_times = [
        candidate["time"]
        for candidate in cut_candidates
        if (
            candidate["time"] > 0.05
            and
            candidate["time"]
            < duration - 0.05
        )
    ]

    cut_times = sorted(
        set(cut_times)
    )

    points = [
        0.0,
        *cut_times,
        duration,
    ]

    points = sorted(
        set(
            round(
                point,
                3,
            )
            for point in points
        )
    )

    slots = []

    for index in range(
        len(points) - 1
    ):

        start = points[index]
        end = points[index + 1]

        if end <= start:
            continue

        slot_duration = end - start

        character = characterize_slot(
            slot_duration,
            anchors,
            start,
            end,
        )

        slots.append(
            {
                "slot_id": index + 1,
                "start": round(
                    start,
                    3,
                ),
                "end": round(
                    end,
                    3,
                ),
                "duration": round(
                    slot_duration,
                    3,
                ),
                **character,
                "anchors": [
                    anchor
                    for anchor in anchors
                    if (
                        start
                        <=
                        anchor["time"]
                        <
                        end
                    )
                ],
            }
        )

    return slots


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("🎵 REFERENCE RHYTHM BUILDER")
    print()

    reference_scenes, duration = (
        load_reference()
    )

    edit_events = load_edit_events()

    print(
        f"Reference duration : "
        f"{duration:.3f}s"
    )

    print(
        f"Hard scenes        : "
        f"{len(reference_scenes)}"
    )

    print(
        f"Visual events      : "
        f"{len(edit_events)}"
    )

    candidates = build_candidates(
        edit_events
    )

    anchors = select_anchors(
        candidates,
        min_gap=0.35,
        max_anchors=32,
    )

    cut_candidates = select_cut_candidates(
        anchors,
        min_gap=0.65,
    )

    hard_boundaries = (
        build_hard_boundaries(
            reference_scenes
        )
    )

    scene_map = attach_anchors_to_scenes(
        reference_scenes,
        anchors,
    )

    rhythm_slots = build_rhythm_slots(
        duration,
        cut_candidates,
        anchors,
    )

    result = {

        "project": "AMV Director",

        "version": "reference_rhythm_v1",

        "duration": round(
            duration,
            3,
        ),

        "source_counts": {
            "hard_scenes": len(
                reference_scenes
            ),
            "visual_events": len(
                edit_events
            ),
            "selected_anchors": len(
                anchors
            ),
            "suggested_cuts": len(
                cut_candidates
            ),
        },

        "hard_boundaries": (
            hard_boundaries
        ),

        "editorial_anchors": anchors,

        "suggested_cuts": cut_candidates,

        "scene_map": scene_map,

        "rhythm_slots": rhythm_slots,
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
            result,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("🎵 REFERENCE RHYTHM MAP")
    print("=" * 76)

    print(
        f"Selected anchors : "
        f"{len(anchors)}"
    )

    print(
        f"Suggested cuts   : "
        f"{len(cut_candidates)}"
    )

    print(
        f"Rhythm slots     : "
        f"{len(rhythm_slots)}"
    )

    print()
    print("Top editorial anchors:")

    ranked = sorted(
        anchors,
        key=lambda x: x["anchor_score"],
        reverse=True,
    )

    for index, anchor in enumerate(
        ranked[:15],
        start=1,
    ):

        print(
            f"  #{index:02d} "
            f"{anchor['time']:.3f}s | "
            f"{anchor['anchor_type']:<16} | "
            f"score={anchor['anchor_score']:.3f} | "
            f"strength={anchor['strength']:.3f} | "
            f"energy={anchor['energy']:.3f} | "
            f"onset={anchor['onset_distance_ms']:.0f}ms"
        )

    print()
    print("Suggested cut points:")

    for cut in cut_candidates:

        print(
            f"  {cut['time']:.3f}s | "
            f"{cut['anchor_type']} | "
            f"score={cut['anchor_score']:.3f}"
        )

    print()
    print("Rhythm slots:")

    for slot in rhythm_slots:

        print(
            f"  [{slot['start']:.3f} → "
            f"{slot['end']:.3f}] "
            f"{slot['duration']:.3f}s | "
            f"{slot['pacing']:<10} | "
            f"events={slot['event_density']:.2f}/s"
        )

    print()
    print("=" * 76)
    print("✅ Saved:")
    print(OUTPUT_FILE)
    print("=" * 76)


if __name__ == "__main__":
    main()