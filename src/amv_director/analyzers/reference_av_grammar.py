from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any


BASE_DIR = Path(__file__).resolve().parents[3]

INPUT_JSON = (
    BASE_DIR
    / "data/outputs/reference_av_diagnostics_v4.json"
)

OUTPUT_JSON = (
    BASE_DIR
    / "data/outputs/reference_av_grammar_v1.json"
)


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

CLUSTER_GAP = 0.65

MIN_EVENT_SCORE = 0.32


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def f(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def strongest(values):
    return max(values) if values else 0.0


# ------------------------------------------------------------
# EVENT ROLE INFERENCE
# ------------------------------------------------------------

def infer_role(item: Dict[str, Any]) -> Dict[str, Any]:

    audio = item.get("audio", {})
    visual = item.get("visual", {})
    rel = item.get("relationship", {})

    weapon = (
        0.35 * max(
            f(audio.get("gunshot")),
            f(audio.get("pistol")),
        )
        + 0.30 * f(visual.get("gun_weapon"))
        + 0.35 * f(visual.get("gun_fire"))
    )

    sword = (
        0.30 * f(audio.get("sword"))
        + 0.45 * f(visual.get("sword_weapon"))
        + 0.25 * f(visual.get("combat"))
    )

    physical = (
        0.40 * f(audio.get("impact"))
        + 0.35 * f(visual.get("physical_hit"))
        + 0.25 * f(visual.get("impact_visual"))
    )

    explosion = (
        0.45 * f(audio.get("explosion"))
        + 0.40 * f(visual.get("explosion"))
        + 0.15 * f(audio.get("impact"))
    )

    whoosh = (
        0.45 * f(audio.get("whoosh"))
        + 0.35 * f(visual.get("fast_motion"))
        + 0.20 * f(visual.get("combat"))
    )

    scream = (
        0.55 * f(audio.get("scream"))
        + 0.45 * f(visual.get("scream"))
    )

    glass = (
        0.55 * f(audio.get("glass"))
        + 0.45 * f(visual.get("glass_break"))
    )

    candidates = [
        ("WEAPON_ACTION", weapon),
        ("SWORD_ACTION", sword),
        ("PHYSICAL_IMPACT", physical),
        ("EXPLOSION", explosion),
        ("WHOOSH_ACTION", whoosh),
        ("SCREAM", scream),
        ("GLASS_BREAK", glass),
    ]

    candidates.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    role, score = candidates[0]

    # --------------------------------------------------------
    # Editorial role modifiers
    # --------------------------------------------------------

    if role in {
        "WEAPON_ACTION",
        "SWORD_ACTION",
        "PHYSICAL_IMPACT",
        "EXPLOSION",
    }:
        phase = "impact"

    elif role == "WHOOSH_ACTION":
        phase = "motion"

    elif role in {
        "SCREAM",
        "GLASS_BREAK",
    }:
        phase = "accent"

    else:
        phase = "accent"

    return {
        "role": role,
        "score": round(score, 4),
        "phase": phase,

        "evidence": {
            "weapon": round(weapon, 4),
            "sword": round(sword, 4),
            "physical": round(physical, 4),
            "explosion": round(explosion, 4),
            "whoosh": round(whoosh, 4),
            "scream": round(scream, 4),
            "glass": round(glass, 4),
        },
    }


# ------------------------------------------------------------
# CLUSTER NEARBY AUDIO-VISUAL WINDOWS
# ------------------------------------------------------------

def cluster_windows(
    windows: List[Dict[str, Any]]
) -> List[List[Dict[str, Any]]]:

    if not windows:
        return []

    ordered = sorted(
        windows,
        key=lambda x: f(x.get("timestamp")),
    )

    clusters = []
    current = [ordered[0]]

    for item in ordered[1:]:

        previous_time = f(
            current[-1].get("timestamp")
        )

        current_time = f(
            item.get("timestamp")
        )

        if (
            current_time - previous_time
            <= CLUSTER_GAP
        ):
            current.append(item)
        else:
            clusters.append(current)
            current = [item]

    clusters.append(current)

    return clusters


# ------------------------------------------------------------
# BUILD EDITORIAL EVENT
# ------------------------------------------------------------

def build_event(
    cluster: List[Dict[str, Any]],
    index: int,
) -> Dict[str, Any]:

    analyzed = []

    for item in cluster:

        role_data = infer_role(item)

        if role_data["score"] >= MIN_EVENT_SCORE:
            analyzed.append(
                (item, role_data)
            )

    if not analyzed:
        return None

    # Strongest evidence point in this cluster.
    anchor_item, anchor_role = max(
        analyzed,
        key=lambda x: x[1]["score"],
    )

    start = f(
        cluster[0].get("timestamp")
    )

    end = f(
        cluster[-1].get("timestamp")
    )

    duration = max(
        0.05,
        end - start,
    )

    # --------------------------------------------------------
    # Collect role evidence
    # --------------------------------------------------------

    role_scores = {}

    for _, role_data in analyzed:

        role = role_data["role"]

        role_scores[role] = max(
            role_scores.get(role, 0.0),
            role_data["score"],
        )

    # --------------------------------------------------------
    # Detect build → impact patterns
    # --------------------------------------------------------

    motion_score = max(
        f(x[0].get("relationship", {}).get("whoosh_action"))
        for x in analyzed
    )

    # Use direct visual/audio evidence instead
    motion_score = max(
        f(item.get("relationship", {}).get("whoosh", 0.0))
        for item, _ in analyzed
    ) if analyzed else 0.0

    impact_score = max(
        f(item.get("relationship", {}).get("physical_impact", 0.0))
        for item, _ in analyzed
    ) if analyzed else 0.0

    has_motion = motion_score >= 0.30
    has_impact = impact_score >= 0.35

    if has_motion and has_impact:
        sequence_type = "BUILD_TO_IMPACT"

    elif has_impact:
        sequence_type = "DIRECT_IMPACT"

    elif has_motion:
        sequence_type = "MOTION_ACCENT"

    else:
        sequence_type = "ACCENT"

    # --------------------------------------------------------
    # Editorial recommendations
    # --------------------------------------------------------

    if sequence_type == "BUILD_TO_IMPACT":

        edit_behavior = [
            "allow_short_buildup",
            "directional_motion",
            "pre_impact_zoom",
            "impact_snap",
            "short_shake_burst",
            "optional_flash",
            "cut_or_reset",
        ]

    elif sequence_type == "DIRECT_IMPACT":

        edit_behavior = [
            "hold_or_prepare_frame",
            "impact_snap",
            "short_shake_burst",
            "optional_flash",
            "cut_or_reset",
        ]

    elif sequence_type == "MOTION_ACCENT":

        edit_behavior = [
            "directional_motion",
            "motion_blur",
            "short_hold",
            "cut_on_resolution",
        ]

    else:

        edit_behavior = [
            "controlled_accent",
            "short_hold",
        ]

    return {
        "event_id": index,

        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration, 3),

        "anchor": {
            "timestamp": round(
                f(anchor_item.get("timestamp")),
                3,
            ),

            "role": anchor_role["role"],
            "score": anchor_role["score"],
        },

        "role_scores": {
            key: round(value, 4)
            for key, value in sorted(
                role_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        },

        "sequence_type": sequence_type,

        "event_count": len(cluster),

        "edit_behavior": edit_behavior,

        "source_windows": [
            round(
                f(item.get("timestamp")),
                3,
            )
            for item in cluster
        ],
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 72)
    print("🎬 REFERENCE AUDIO-VISUAL GRAMMAR BUILDER")
    print("=" * 72)

    print(
        f"\nInput : {INPUT_JSON}"
    )

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    windows = data.get("windows", [])

    print(
        f"Diagnostic windows : {len(windows)}"
    )

    clusters = cluster_windows(
        windows
    )

    print(
        f"Temporal clusters  : {len(clusters)}"
    )

    events = []

    for idx, cluster in enumerate(
        clusters,
        start=1,
    ):

        event = build_event(
            cluster,
            idx,
        )

        if event is not None:
            events.append(event)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    role_counts = {}
    sequence_counts = {}

    for event in events:

        role = event["anchor"]["role"]

        role_counts[role] = (
            role_counts.get(role, 0)
            + 1
        )

        sequence = event["sequence_type"]

        sequence_counts[sequence] = (
            sequence_counts.get(sequence, 0)
            + 1
        )

    result = {

        "meta": {
            "version": "v1",

            "purpose": (
                "Converts independent audio/visual evidence "
                "into reusable editorial audiovisual events."
            ),

            "source": str(
                INPUT_JSON
            ),

            "cluster_gap": CLUSTER_GAP,
        },

        "summary": {
            "diagnostic_windows": len(windows),
            "temporal_clusters": len(clusters),
            "editorial_events": len(events),

            "event_roles": role_counts,

            "sequence_types": sequence_counts,
        },

        "events": events,
    }

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSON,
        "w",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("✅ REFERENCE AV GRAMMAR COMPLETE")
    print("=" * 72)

    print(
        f"\nDiagnostic windows : {len(windows)}"
    )

    print(
        f"Temporal clusters  : {len(clusters)}"
    )

    print(
        f"Editorial events   : {len(events)}"
    )

    print("\nEvent roles:")

    for key, value in sorted(
        role_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"  {key:<24}: {value}"
        )

    print("\nSequence types:")

    for key, value in sorted(
        sequence_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"  {key:<24}: {value}"
        )

    print("\nFirst 15 editorial events:")

    for event in events[:15]:

        print(
            f"  "
            f"{event['start']:6.3f}"
            f" → "
            f"{event['end']:6.3f} | "
            f"{event['anchor']['role']:<20} | "
            f"{event['sequence_type']}"
        )

    print(
        f"\nOutput: {OUTPUT_JSON}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()