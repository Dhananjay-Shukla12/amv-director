from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "reference_rhythm_map.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "reference_edit_schedule.json"
)


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

# Don't allow artificial cuts closer than this.
MIN_EDIT_GAP = 0.45

# If an anchor is extremely close to a real hard cut,
# treat it as evidence for that same cut rather than
# creating a second cut.
HARD_CUT_MERGE_DISTANCE = 0.18


ANCHOR_PRIORITY = {
    "impact_anchor": 1.00,
    "cut_candidate": 0.95,
    "onset_anchor": 0.82,
    "beat_anchor": 0.78,
    "visual_impact": 0.72,
    "visual_anchor": 0.45,
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clamp(value, low=0.0, high=1.0):
    return max(
        low,
        min(high, float(value)),
    )


def anchor_priority(anchor):
    anchor_type = anchor.get(
        "anchor_type",
        "visual_anchor",
    )

    return ANCHOR_PRIORITY.get(
        anchor_type,
        0.40,
    )


# ------------------------------------------------------------
# Build hard edit points
# ------------------------------------------------------------

def build_hard_points(data):

    points = []

    for item in data.get(
        "hard_boundaries",
        [],
    ):

        time = float(
            item["time"]
        )

        point_type = item.get(
            "type",
            "hard_boundary",
        )

        # Ignore 0 and final duration.
        if time <= 0:
            continue

        points.append(
            {
                "time": time,
                "kind": "hard_cut",
                "strength": 1.0,
                "source": "reference_scene_boundary",
                "scene_id": item.get(
                    "scene_id"
                ),
                "anchor_type": None,
            }
        )

    return points


# ------------------------------------------------------------
# Select useful anchors
# ------------------------------------------------------------

def select_anchor_points(
    anchors,
    hard_points,
):
    """
    Select strong musical/visual anchors while avoiding
    excessive cuts.

    Hard reference boundaries are protected first.
    """

    candidates = []

    for anchor in anchors:

        time = float(
            anchor["time"]
        )

        # Ignore near the beginning/end.
        if time <= 0.05:
            continue

        priority = anchor_priority(
            anchor
        )

        effective_score = (
            float(
                anchor.get(
                    "anchor_score",
                    0.0,
                )
            )
            *
            priority
        )

        candidate = {
            **anchor,
            "_priority_score": effective_score,
        }

        candidates.append(
            candidate
        )

    candidates.sort(
        key=lambda x: x["_priority_score"],
        reverse=True,
    )

    selected = []

    for candidate in candidates:

        time = candidate["time"]

        # Don't add an anchor immediately next to a real
        # hard cut.
        close_to_hard = any(
            abs(
                time -
                hard["time"]
            )
            < HARD_CUT_MERGE_DISTANCE
            for hard in hard_points
        )

        if close_to_hard:
            continue

        # Prevent excessive edit density.
        too_close = any(
            abs(
                time -
                existing["time"]
            )
            < MIN_EDIT_GAP
            for existing in (
                hard_points +
                selected
            )
        )

        if too_close:
            continue

        selected.append(
            {
                "time": time,
                "kind": "anchor_cut",
                "strength": float(
                    candidate.get(
                        "anchor_score",
                        0.0,
                    )
                ),
                "source": "editorial_anchor",
                "anchor_type": candidate.get(
                    "anchor_type"
                ),
                "event_type": candidate.get(
                    "event_type"
                ),
                "energy": float(
                    candidate.get(
                        "energy",
                        0.0,
                    )
                ),
                "motion": float(
                    candidate.get(
                        "motion",
                        0.0,
                    )
                ),
                "onset_distance_ms": float(
                    candidate.get(
                        "onset_distance_ms",
                        9999,
                    )
                ),
            }
        )

    return selected


# ------------------------------------------------------------
# Merge all edit points
# ------------------------------------------------------------

def build_edit_points(
    duration,
    hard_points,
    anchor_points,
):
    """
    Hard cuts + selected editorial anchors.

    If an anchor and hard cut occupy essentially the same
    location, the hard cut remains the actual edit point.
    """

    combined = [
        *hard_points,
        *anchor_points,
    ]

    combined.sort(
        key=lambda x: x["time"]
    )

    merged = []

    for point in combined:

        time = point["time"]

        if time <= 0.05:
            continue

        if time >= duration - 0.05:
            continue

        if not merged:
            merged.append(point)
            continue

        previous = merged[-1]

        if (
            abs(
                time -
                previous["time"]
            )
            <
            HARD_CUT_MERGE_DISTANCE
        ):

            # Hard cut takes precedence.
            if point["kind"] == "hard_cut":

                merged[-1] = point

            continue

        merged.append(point)

    return merged


# ------------------------------------------------------------
# Classify interval
# ------------------------------------------------------------

def classify_interval(
    start,
    end,
    points,
    anchors,
):
    duration = end - start

    anchors_inside = [
        anchor
        for anchor in anchors
        if (
            start
            <=
            anchor["time"]
            <
            end
        )
    ]

    event_density = (
        len(anchors_inside)
        /
        max(duration, 0.001)
    )

    max_energy = max(
        (
            float(
                anchor.get(
                    "energy",
                    0.0,
                )
            )
            for anchor in anchors_inside
        ),
        default=0.0,
    )

    max_strength = max(
        (
            float(
                anchor.get(
                    "strength",
                    0.0,
                )
            )
            for anchor in anchors_inside
        ),
        default=0.0,
    )

    # Determine pacing based on actual reference density.
    if duration <= 0.50:

        pacing = "impact"

    elif event_density >= 1.50:

        pacing = "very_fast"

    elif event_density >= 0.80:

        pacing = "fast"

    elif event_density >= 0.40:

        pacing = "medium"

    elif duration >= 6.0:

        pacing = "slow"

    else:

        pacing = "slow_medium"

    # Determine visual/musical role.
    if max_energy >= 0.65:

        role = "impact"

    elif max_energy >= 0.50 and (
        event_density >= 0.70
    ):

        role = "action"

    elif event_density >= 0.90:

        role = "action"

    elif duration >= 5.0:

        role = "cinematic"

    else:

        role = "cinematic"

    return {
        "duration": round(
            duration,
            3,
        ),
        "pacing": pacing,
        "role": role,
        "event_density": round(
            event_density,
            3,
        ),
        "max_energy": round(
            max_energy,
            3,
        ),
        "max_anchor_strength": round(
            max_strength,
            3,
        ),
        "anchor_count": len(
            anchors_inside
        ),
        "anchors": anchors_inside,
    }


# ------------------------------------------------------------
# Build intervals
# ------------------------------------------------------------

def build_intervals(
    duration,
    edit_points,
    anchors,
):

    times = [
        0.0,
        *[
            point["time"]
            for point in edit_points
        ],
        duration,
    ]

    times = sorted(
        set(
            round(
                time,
                3,
            )
            for time in times
        )
    )

    intervals = []

    for index in range(
        len(times) - 1
    ):

        start = times[index]
        end = times[index + 1]

        if end <= start:
            continue

        features = classify_interval(
            start,
            end,
            edit_points,
            anchors,
        )

        intervals.append(
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
                **features,
            }
        )

    return intervals


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print()
    print("🎬 REFERENCE EDIT SCHEDULE BUILDER")
    print()

    data = load_json(
        INPUT_FILE
    )

    duration = float(
        data["duration"]
    )

    anchors = data.get(
        "editorial_anchors",
        [],
    )

    hard_points = build_hard_points(
        data
    )

    anchor_points = select_anchor_points(
        anchors,
        hard_points,
    )

    edit_points = build_edit_points(
        duration,
        hard_points,
        anchor_points,
    )

    intervals = build_intervals(
        duration,
        edit_points,
        anchors,
    )

    result = {

        "project": "AMV Director",

        "version": "reference_edit_schedule_v1",

        "duration": round(
            duration,
            3,
        ),

        "counts": {
            "editorial_anchors": len(
                anchors
            ),
            "hard_cuts": len(
                hard_points
            ),
            "selected_anchor_cuts": len(
                anchor_points
            ),
            "edit_points": len(
                edit_points
            ),
            "intervals": len(
                intervals
            ),
        },

        "edit_points": edit_points,

        "intervals": intervals,
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

    print("=" * 84)
    print("🎬 REFERENCE EDIT SCHEDULE")
    print("=" * 84)

    print(
        f"Duration            : "
        f"{duration:.3f}s"
    )

    print(
        f"Original anchors    : "
        f"{len(anchors)}"
    )

    print(
        f"Hard cuts           : "
        f"{len(hard_points)}"
    )

    print(
        f"Anchor cuts         : "
        f"{len(anchor_points)}"
    )

    print(
        f"Total edit points   : "
        f"{len(edit_points)}"
    )

    print(
        f"Final intervals     : "
        f"{len(intervals)}"
    )

    print()
    print("EDIT POINTS:")

    for point in edit_points:

        print(
            f"  {point['time']:7.3f}s  "
            f"{point['kind']:<12} "
            f"{point.get('anchor_type') or ''}"
        )

    print()
    print("INTERVALS:")

    for interval in intervals:

        print(
            f"  [{interval['start']:7.3f} → "
            f"{interval['end']:7.3f}] "
            f"{interval['duration']:6.3f}s | "
            f"{interval['pacing']:<11} | "
            f"{interval['role']:<10} | "
            f"anchors={interval['anchor_count']}"
        )

    print()
    print("=" * 84)
    print("✅ Saved:")
    print(OUTPUT_FILE)
    print("=" * 84)


if __name__ == "__main__":
    main()