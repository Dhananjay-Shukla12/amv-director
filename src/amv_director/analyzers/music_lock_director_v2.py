from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

MUSIC_PLAN = BASE_DIR / "data/outputs/music_director_plan.json"
AV_GRAMMAR = BASE_DIR / "data/outputs/reference_av_grammar_v1.json"
SHOT_PLAN = BASE_DIR / "data/outputs/reference_conditioned_shot_plan.json"

OUTPUT = BASE_DIR / "data/outputs/music_locked_edit_plan_v2.json"


# ============================================================
# TARGETS
# ============================================================

TARGET_CUTS = 52
MIN_CUT_GAP = 0.16

MAX_CUT_GAP_LOW = 2.8
MAX_CUT_GAP_MEDIUM = 1.9
MAX_CUT_GAP_HIGH = 1.15

STRONG_ONSET_KEEP = 0.22
BEAT_KEEP = 0.10

AV_EVENT_BOOST = 0.35
PHRASE_BOOST = 0.30
ROLE_BOOST = 0.20


# ============================================================
# HELPERS
# ============================================================

def f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as fh:
        return json.load(fh)


def normalize_list(value: Any) -> List[float]:
    if not isinstance(value, list):
        return []

    result = []

    for x in value:
        if isinstance(x, (int, float)):
            result.append(float(x))

        elif isinstance(x, dict):
            for key in (
                "time",
                "timestamp",
                "start",
                "position",
            ):
                if key in x:
                    result.append(f(x[key]))
                    break

    return result


def recursive_find_arrays(
    obj: Any,
    names: set[str],
) -> Dict[str, List[Any]]:

    found = {}

    def walk(x: Any):

        if isinstance(x, dict):

            for key, value in x.items():

                if (
                    key.lower() in names
                    and isinstance(value, list)
                ):
                    found.setdefault(
                        key.lower(),
                        value,
                    )

                walk(value)

        elif isinstance(x, list):

            for item in x:
                walk(item)

    walk(obj)

    return found


def nearest_distance(
    timestamp: float,
    values: List[float],
) -> float:

    if not values:
        return 999.0

    return min(
        abs(timestamp - x)
        for x in values
    )


def nearest_value(
    timestamp: float,
    values: List[float],
) -> float | None:

    if not values:
        return None

    return min(
        values,
        key=lambda x: abs(timestamp - x),
    )


# ============================================================
# MUSIC SEGMENTS
# ============================================================

def extract_music_segments(
    music_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:

    segments = music_plan.get(
        "segments",
        [],
    )

    if not isinstance(segments, list):
        return []

    return segments


def segment_for_time(
    timestamp: float,
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:

    for segment in segments:

        start = f(segment.get("start"))
        end = f(segment.get("end"))

        if start <= timestamp <= end:
            return segment

    if segments:

        return min(
            segments,
            key=lambda s: abs(
                timestamp - f(s.get("start"))
            ),
        )

    return {}


# ============================================================
# ROLE MODEL
# ============================================================

ROLE_WEIGHTS = {
    "cinematic_zone": 0.15,
    "build_zone": 0.30,
    "impact_zone": 0.85,
    "accent_zone": 0.65,
    "action_zone": 0.80,
}


def role_weight(
    role: str,
) -> float:

    return ROLE_WEIGHTS.get(
        role,
        0.35,
    )


# ============================================================
# AV EVENTS
# ============================================================

def prepare_av_events(
    grammar: Dict[str, Any],
) -> List[Dict[str, Any]]:

    events = grammar.get(
        "events",
        [],
    )

    result = []

    for event in events:

        anchor = event.get(
            "anchor",
            {},
        )

        timestamp = f(
            anchor.get("timestamp")
        )

        score = f(
            anchor.get("score")
        )

        result.append({
            "timestamp": timestamp,
            "role": anchor.get(
                "role",
                "UNKNOWN",
            ),
            "score": score,
            "sequence_type": event.get(
                "sequence_type",
                "ACCENT",
            ),
            "start": f(
                event.get("start")
            ),
            "end": f(
                event.get("end")
            ),
        })

    return result


# ============================================================
# AV EVENT SCORE
# ============================================================

def av_event_score(
    timestamp: float,
    av_events: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any] | None]:

    if not av_events:
        return 0.0, None

    nearest = min(
        av_events,
        key=lambda x: abs(
            timestamp - x["timestamp"]
        ),
    )

    distance = abs(
        timestamp - nearest["timestamp"]
    )

    if distance > 0.75:
        return 0.0, None

    proximity = max(
        0.0,
        1.0 - distance / 0.75,
    )

    score = (
        nearest["score"]
        * proximity
    )

    return score, nearest


# ============================================================
# CANDIDATE EXTRACTION
# ============================================================

def build_candidates(
    music_plan: Dict[str, Any],
    av_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    arrays = recursive_find_arrays(
        music_plan,
        {
            "beats",
            "strong_onsets",
            "onsets",
            "accents",
            "phrase_starts",
            "phrase_ends",
            "drops",
        },
    )

    candidates = []

    # --------------------------------------------------------
    # Strong onsets
    # --------------------------------------------------------

    for timestamp in normalize_list(
        arrays.get(
            "strong_onsets",
            [],
        )
    ):

        candidates.append({
            "timestamp": timestamp,
            "source": "strong_onset",
            "base_score": 0.72,
        })

    # --------------------------------------------------------
    # Beats
    # --------------------------------------------------------

    for timestamp in normalize_list(
        arrays.get(
            "beats",
            [],
        )
    ):

        candidates.append({
            "timestamp": timestamp,
            "source": "beat",
            "base_score": 0.42,
        })

    # --------------------------------------------------------
    # Explicit accents
    # --------------------------------------------------------

    for key in (
        "accents",
        "drops",
        "phrase_starts",
        "phrase_ends",
    ):

        for timestamp in normalize_list(
            arrays.get(key, [])
        ):

            candidates.append({
                "timestamp": timestamp,
                "source": key,
                "base_score": 0.82,
            })

    # --------------------------------------------------------
    # AV events become candidates too.
    # --------------------------------------------------------

    for event in av_events:

        candidates.append({
            "timestamp": event["timestamp"],
            "source": "av_event",
            "base_score": 0.60,
        })

    return candidates


# ============================================================
# CANDIDATE SCORING
# ============================================================

def score_candidate(
    candidate: Dict[str, Any],
    segments: List[Dict[str, Any]],
    strong_onsets: List[float],
    beats: List[float],
    av_events: List[Dict[str, Any]],
) -> Dict[str, Any]:

    timestamp = candidate["timestamp"]

    segment = segment_for_time(
        timestamp,
        segments,
    )

    role = segment.get(
        "editorial_role",
        segment.get(
            "role",
            "unknown",
        ),
    )

    role_score = role_weight(
        role
    )

    intensity = f(
        segment.get(
            "editorial_intensity",
            segment.get(
                "music_energy",
                0.0,
            ),
        )
    )

    trend = f(
        segment.get(
            "energy_trend",
            0.0,
        )
    )

    av_score, av_event = av_event_score(
        timestamp,
        av_events,
    )

    strong_distance = nearest_distance(
        timestamp,
        strong_onsets,
    )

    beat_distance = nearest_distance(
        timestamp,
        beats,
    )

    strong_bonus = (
        0.30
        if strong_distance <= 0.08
        else 0.0
    )

    beat_bonus = (
        0.14
        if beat_distance <= 0.06
        else 0.0
    )

    phrase_bonus = (
        0.22
        if candidate["source"]
        in {
            "phrase_starts",
            "phrase_ends",
            "drops",
        }
        else 0.0
    )

    score = (
        candidate["base_score"]
        + strong_bonus
        + beat_bonus
        + phrase_bonus
        + ROLE_BOOST * role_score
        + AV_EVENT_BOOST * av_score
        + 0.15 * intensity
        + 0.10 * max(trend, 0.0)
    )

    return {
        **candidate,

        "score": round(
            score,
            4,
        ),

        "segment": {
            "segment": segment.get(
                "segment"
            ),
            "role": role,
            "intensity": intensity,
            "energy_trend": trend,
        },

        "music_alignment": {
            "strong_distance": round(
                strong_distance,
                4,
            ),
            "beat_distance": round(
                beat_distance,
                4,
            ),
        },

        "av_alignment": (
            {
                "distance": round(
                    abs(
                        timestamp
                        - av_event["timestamp"]
                    ),
                    4,
                ),
                "role": av_event["role"],
                "score": av_event["score"],
                "sequence_type": av_event[
                    "sequence_type"
                ],
            }
            if av_event
            else None
        ),
    }


# ============================================================
# NON-MAXIMUM SUPPRESSION
# ============================================================

def suppress_close_candidates(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    ordered = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True,
    )

    selected = []

    for candidate in ordered:

        t = candidate["timestamp"]

        too_close = any(
            abs(
                t
                - selected_item["timestamp"]
            )
            < MIN_CUT_GAP
            for selected_item in selected
        )

        if not too_close:
            selected.append(
                candidate
            )

    return sorted(
        selected,
        key=lambda x: x["timestamp"],
    )


# ============================================================
# TARGET CUT SELECTION
# ============================================================

def choose_cut_events(
    candidates: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    candidates = suppress_close_candidates(
        candidates
    )

    # Prefer high-scoring candidates.
    ranked = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True,
    )

    selected = []

    # --------------------------------------------------------
    # First pass: strongest editorial candidates.
    # --------------------------------------------------------

    for candidate in ranked:

        if len(selected) >= TARGET_CUTS:
            break

        t = candidate["timestamp"]

        if any(
            abs(
                t
                - item["timestamp"]
            ) < MIN_CUT_GAP
            for item in selected
        ):
            continue

        selected.append(
            candidate
        )

    # --------------------------------------------------------
    # Second pass: fill large visual gaps.
    # --------------------------------------------------------

    selected = sorted(
        selected,
        key=lambda x: x["timestamp"],
    )

    duration = (
        segments[-1].get("end", 59.0)
        if segments
        else 59.0
    )

    changed = True

    while (
        len(selected) < TARGET_CUTS
        and changed
    ):

        changed = False

        gaps = []

        boundaries = [
            0.0
        ]

        boundaries.extend(
            item["timestamp"]
            for item in selected
        )

        boundaries.append(
            f(duration)
        )

        for a, b in zip(
            boundaries[:-1],
            boundaries[1:],
        ):

            gap = b - a

            midpoint = (
                a + gap / 2.0
            )

            segment = segment_for_time(
                midpoint,
                segments,
            )

            role = segment.get(
                "editorial_role",
                "unknown",
            )

            if role == "cinematic_zone":
                max_gap = MAX_CUT_GAP_LOW

            elif role in {
                "build_zone",
                "accent_zone",
            }:
                max_gap = MAX_CUT_GAP_MEDIUM

            else:
                max_gap = MAX_CUT_GAP_HIGH

            if gap > max_gap:
                gaps.append(
                    (
                        gap,
                        midpoint,
                    )
                )

        if not gaps:
            break

        _, midpoint = max(
            gaps,
            key=lambda x: x[0],
        )

        local = min(
            candidates,
            key=lambda x: abs(
                x["timestamp"]
                - midpoint
            ),
        )

        if all(
            abs(
                local["timestamp"]
                - item["timestamp"]
            ) >= MIN_CUT_GAP
            for item in selected
        ):

            selected.append(
                local
            )

            selected = sorted(
                selected,
                key=lambda x: x["timestamp"],
            )

            changed = True

    return selected


# ============================================================
# ACTION CLASSIFICATION
# ============================================================

def classify_action(
    candidate: Dict[str, Any],
) -> str:

    source = candidate["source"]

    av = candidate.get(
        "av_alignment"
    )

    segment = candidate.get(
        "segment",
        {},
    )

    role = segment.get(
        "role",
        "unknown",
    )

    if av:

        sequence = av["sequence_type"]

        if sequence == "BUILD_TO_IMPACT":
            return "IMPACT"

        if sequence == "DIRECT_IMPACT":
            return "IMPACT"

        if sequence == "MOTION_ACCENT":
            return "ACCENT"

        if sequence == "ACCENT":
            return "ACCENT"

    if source in {
        "drops",
        "phrase_starts",
        "phrase_ends",
    }:
        return "CUT"

    if role == "impact_zone":
        return "IMPACT"

    if role == "action_zone":
        return "CUT"

    if role == "accent_zone":
        return "ACCENT"

    return "CUT"


# ============================================================
# BUILD FINAL DECISIONS
# ============================================================

def build_decisions(
    selected_cuts: List[Dict[str, Any]],
    all_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    decisions = []

    # --------------------------------------------------------
    # CUT backbone
    # --------------------------------------------------------

    for item in selected_cuts:

        decisions.append({
            "timestamp": round(
                item["timestamp"],
                3,
            ),

            "action": "CUT",

            "reason": {
                "source": item["source"],
                "score": item["score"],
                "segment": item["segment"],
                "av_alignment": item[
                    "av_alignment"
                ],
            },
        })

    # --------------------------------------------------------
    # Add high-confidence impact/accent events
    # without making them cuts.
    # --------------------------------------------------------

    for item in all_candidates:

        action = classify_action(
            item
        )

        if action == "CUT":
            continue

        timestamp = item["timestamp"]

        too_close = any(
            abs(
                timestamp
                - d["timestamp"]
            ) < 0.12
            for d in decisions
        )

        if too_close:
            continue

        threshold = (
            0.72
            if action == "IMPACT"
            else 0.78
        )

        if item["score"] < threshold:
            continue

        decisions.append({
            "timestamp": round(
                timestamp,
                3,
            ),

            "action": action,

            "reason": {
                "source": item["source"],
                "score": item["score"],
                "segment": item["segment"],
                "av_alignment": item[
                    "av_alignment"
                ],
            },
        })

    return sorted(
        decisions,
        key=lambda x: x["timestamp"],
    )


# ============================================================
# SHOT WINDOWS
# ============================================================

def build_shot_windows(
    cuts: List[Dict[str, Any]],
    duration: float,
) -> List[Dict[str, Any]]:

    points = [0.0]

    points.extend(
        x["timestamp"]
        for x in cuts
        if 0.0 < x["timestamp"] < duration
    )

    points.append(duration)

    points = sorted(
        set(
            round(x, 3)
            for x in points
        )
    )

    windows = []

    for i, (a, b) in enumerate(
        zip(
            points[:-1],
            points[1:],
        ),
        start=1,
    ):

        windows.append({
            "shot_id": i,
            "start": a,
            "end": b,
            "duration": round(
                b - a,
                3,
            ),
        })

    return windows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("🎵 MUSIC LOCK DIRECTOR V2")
    print("=" * 72)

    music_plan = load_json(
        MUSIC_PLAN
    )

    av_grammar = load_json(
        AV_GRAMMAR
    )

    try:
        shot_plan = load_json(
            SHOT_PLAN
        )
    except FileNotFoundError:
        shot_plan = {}

    segments = extract_music_segments(
        music_plan
    )

    av_events = prepare_av_events(
        av_grammar
    )

    arrays = recursive_find_arrays(
        music_plan,
        {
            "beats",
            "strong_onsets",
        },
    )

    beats = normalize_list(
        arrays.get(
            "beats",
            [],
        )
    )

    strong_onsets = normalize_list(
        arrays.get(
            "strong_onsets",
            [],
        )
    )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = 59.072

    if segments:

        duration = max(
            f(x.get("end"))
            for x in segments
        )

    print(
        f"\nMusic duration   : {duration:.3f}s"
    )

    print(
        f"Music segments   : {len(segments)}"
    )

    print(
        f"Strong onsets    : {len(strong_onsets)}"
    )

    print(
        f"Beats            : {len(beats)}"
    )

    print(
        f"AV events        : {len(av_events)}"
    )

    # --------------------------------------------------------
    # Candidates
    # --------------------------------------------------------

    candidates = build_candidates(
        music_plan,
        av_events,
    )

    print(
        f"\nRaw candidates   : {len(candidates)}"
    )

    scored = []

    for candidate in candidates:

        scored.append(
            score_candidate(
                candidate,
                segments,
                strong_onsets,
                beats,
                av_events,
            )
        )

    # --------------------------------------------------------
    # Remove near duplicates
    # --------------------------------------------------------

    scored = suppress_close_candidates(
        scored
    )

    print(
        f"After suppression: {len(scored)}"
    )

    # --------------------------------------------------------
    # CUT BACKBONE
    # --------------------------------------------------------

    selected_cuts = choose_cut_events(
        scored,
        segments,
    )

    print(
        f"Selected cuts   : {len(selected_cuts)}"
    )

    # --------------------------------------------------------
    # DECISIONS
    # --------------------------------------------------------

    decisions = build_decisions(
        selected_cuts,
        scored,
    )

    shots = build_shot_windows(
        selected_cuts,
        duration,
    )

    # --------------------------------------------------------
    # ACTION COUNTS
    # --------------------------------------------------------

    counts = {}

    for decision in decisions:

        action = decision["action"]

        counts[action] = (
            counts.get(action, 0)
            + 1
        )

    # --------------------------------------------------------
    # ROLE COUNTS
    # --------------------------------------------------------

    role_counts = {}

    for decision in decisions:

        role = (
            decision["reason"]
            .get("segment", {})
            .get("role", "unknown")
        )

        role_counts[role] = (
            role_counts.get(role, 0)
            + 1
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output = {

        "meta": {
            "version": "v2",

            "purpose": (
                "Music-locked editorial backbone using "
                "hierarchical music salience, reference "
                "AV grammar and controlled cut density."
            ),

            "duration": duration,
        },

        "music": {
            "segments": len(segments),
            "beats": len(beats),
            "strong_onsets": len(
                strong_onsets
            ),
        },

        "summary": {
            "raw_candidates": len(candidates),
            "usable_candidates": len(scored),
            "selected_cuts": len(
                selected_cuts
            ),
            "total_decisions": len(
                decisions
            ),
            "actions": counts,
            "roles": role_counts,
        },

        "decisions": decisions,

        "shot_windows": shots,

        "selected_cut_candidates": selected_cuts,

        "candidate_pool": scored,
    }

    with open(
        OUTPUT,
        "w",
    ) as fh:

        json.dump(
            output,
            fh,
            indent=2,
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("✅ MUSIC LOCK V2 COMPLETE")
    print("=" * 72)

    print(
        f"\nSelected cuts     : "
        f"{len(selected_cuts)}"
    )

    print(
        f"Total decisions   : "
        f"{len(decisions)}"
    )

    print("\nAction distribution:")

    for key, value in sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"  {key:<16}: {value}"
        )

    print("\nRole distribution:")

    for key, value in sorted(
        role_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"  {key:<24}: {value}"
        )

    print("\nFirst 20 decisions:")

    for decision in decisions[:20]:

        reason = decision["reason"]

        role = (
            reason
            .get("segment", {})
            .get("role", "unknown")
        )

        print(
            f"  "
            f"{decision['timestamp']:7.3f}s | "
            f"{decision['action']:<8} | "
            f"{role:<18} | "
            f"{reason['source']}"
        )

    print(
        f"\nOutput: {OUTPUT}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()