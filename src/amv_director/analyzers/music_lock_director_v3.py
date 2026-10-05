from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

MUSIC_PLAN = (
    BASE_DIR
    / "data/outputs/music_director_plan.json"
)

AV_GRAMMAR = (
    BASE_DIR
    / "data/outputs/reference_av_grammar_v1.json"
)

OUTPUT = (
    BASE_DIR
    / "data/outputs/music_locked_edit_plan_v3.json"
)


# ============================================================
# REFERENCE-BASED TARGETS
# ============================================================

# Reference has 44 editorial shots.
# We allow a little creative freedom.
TARGET_CUTS = 46

CUT_MIN_GAP = 0.20


# ============================================================
# ROLE SETTINGS
# ============================================================

ROLE_SETTINGS = {

    "cinematic_zone": {
        "min_gap": 0.55,
        "preferred_gap": 1.85,
        "max_gap": 3.20,
        "cut_weight": 0.55,
    },

    "build_zone": {
        "min_gap": 0.35,
        "preferred_gap": 1.15,
        "max_gap": 1.90,
        "cut_weight": 0.80,
    },

    "impact_zone": {
        "min_gap": 0.22,
        "preferred_gap": 0.62,
        "max_gap": 1.15,
        "cut_weight": 1.00,
    },

    "accent_zone": {
        "min_gap": 0.30,
        "preferred_gap": 0.85,
        "max_gap": 1.50,
        "cut_weight": 0.85,
    },

    "action_zone": {
        "min_gap": 0.22,
        "preferred_gap": 0.58,
        "max_gap": 1.05,
        "cut_weight": 1.00,
    },

}


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


def recursive_find_arrays(
    obj: Any,
    names: set[str],
) -> Dict[str, List[Any]]:

    found: Dict[str, List[Any]] = {}

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


def normalize_times(
    values: Any,
) -> List[float]:

    if not isinstance(values, list):
        return []

    output = []

    for item in values:

        if isinstance(item, (int, float)):

            output.append(
                float(item)
            )

        elif isinstance(item, dict):

            for key in (
                "timestamp",
                "time",
                "start",
                "position",
            ):

                if key in item:

                    output.append(
                        f(item[key])
                    )

                    break

    return sorted(
        set(
            round(x, 4)
            for x in output
        )
    )


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


def nearest_time(
    timestamp: float,
    values: List[float],
) -> float | None:

    if not values:
        return None

    return min(
        values,
        key=lambda x: abs(
            timestamp - x
        ),
    )


# ============================================================
# MUSIC SEGMENTS
# ============================================================

def get_segments(
    plan: Dict[str, Any],
) -> List[Dict[str, Any]]:

    segments = plan.get(
        "segments",
        [],
    )

    if not isinstance(segments, list):
        return []

    return segments


def segment_at(
    timestamp: float,
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:

    for segment in segments:

        start = f(
            segment.get("start")
        )

        end = f(
            segment.get("end")
        )

        if start <= timestamp <= end:

            return segment

    if not segments:
        return {}

    return min(
        segments,
        key=lambda s: abs(
            timestamp - f(
                s.get("start")
            )
        ),
    )


def segment_role(
    timestamp: float,
    segments: List[Dict[str, Any]],
) -> str:

    segment = segment_at(
        timestamp,
        segments,
    )

    return segment.get(
        "editorial_role",
        segment.get(
            "role",
            "cinematic_zone",
        ),
    )


# ============================================================
# CUT CANDIDATES
# ============================================================

def build_cut_candidates(
    music_plan: Dict[str, Any],
    av_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    arrays = recursive_find_arrays(
        music_plan,
        {
            "beats",
            "strong_onsets",
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

    for t in normalize_times(
        arrays.get(
            "strong_onsets",
            [],
        )
    ):

        candidates.append({
            "timestamp": t,
            "source": "strong_onset",
            "base_score": 0.62,
        })

    # --------------------------------------------------------
    # Beats
    # --------------------------------------------------------

    for t in normalize_times(
        arrays.get(
            "beats",
            [],
        )
    ):

        candidates.append({
            "timestamp": t,
            "source": "beat",
            "base_score": 0.40,
        })

    # --------------------------------------------------------
    # Phrase / structural events
    # --------------------------------------------------------

    for name in (
        "phrase_starts",
        "phrase_ends",
        "drops",
        "accents",
    ):

        for t in normalize_times(
            arrays.get(
                name,
                [],
            )
        ):

            candidates.append({
                "timestamp": t,
                "source": name,
                "base_score": 0.82,
            })

    # --------------------------------------------------------
    # AV events can suggest a cut location.
    #
    # IMPORTANT:
    # They do NOT automatically become effects here.
    # --------------------------------------------------------

    for event in av_events:

        candidates.append({
            "timestamp": event["timestamp"],
            "source": "av_event",
            "base_score": 0.70,
        })

    return candidates


# ============================================================
# SCORE CUT CANDIDATE
# ============================================================

def score_cut_candidate(
    candidate: Dict[str, Any],
    segments: List[Dict[str, Any]],
    strong_onsets: List[float],
    beats: List[float],
    av_events: List[Dict[str, Any]],
) -> Dict[str, Any]:

    t = candidate["timestamp"]

    segment = segment_at(
        t,
        segments,
    )

    role = segment.get(
        "editorial_role",
        "cinematic_zone",
    )

    settings = ROLE_SETTINGS.get(
        role,
        ROLE_SETTINGS["cinematic_zone"],
    )

    score = candidate["base_score"]

    # --------------------------------------------------------
    # Music alignment
    # --------------------------------------------------------

    d_strong = nearest_distance(
        t,
        strong_onsets,
    )

    d_beat = nearest_distance(
        t,
        beats,
    )

    if d_strong <= 0.06:

        score += 0.28

    elif d_strong <= 0.12:

        score += 0.16

    if d_beat <= 0.05:

        score += 0.12

    # --------------------------------------------------------
    # Role weighting
    # --------------------------------------------------------

    score += (
        0.22
        * settings["cut_weight"]
    )

    # --------------------------------------------------------
    # Intensity
    # --------------------------------------------------------

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

    score += (
        0.16
        * intensity
    )

    score += (
        0.10
        * max(trend, 0.0)
    )

    # --------------------------------------------------------
    # AV proximity
    #
    # Small boost only.
    # This prevents AV events from hijacking the cut lane.
    # --------------------------------------------------------

    nearest_av = None

    if av_events:

        nearest_av = min(
            av_events,
            key=lambda event: abs(
                t
                - event["timestamp"]
            ),
        )

        av_distance = abs(
            t
            - nearest_av["timestamp"]
        )

        if av_distance <= 0.18:

            score += (
                0.14
                * nearest_av["score"]
            )

        else:

            nearest_av = None

    return {

        "timestamp": round(
            t,
            4,
        ),

        "source": candidate["source"],

        "score": round(
            score,
            4,
        ),

        "role": role,

        "preferred_gap": settings[
            "preferred_gap"
        ],

        "min_gap": settings[
            "min_gap"
        ],

        "max_gap": settings[
            "max_gap"
        ],

        "music_alignment": {

            "strong_onset_distance": round(
                d_strong,
                4,
            ),

            "beat_distance": round(
                d_beat,
                4,
            ),

        },

        "av_anchor": (

            {
                "timestamp": nearest_av[
                    "timestamp"
                ],

                "role": nearest_av[
                    "role"
                ],

                "score": nearest_av[
                    "score"
                ],

                "sequence_type": nearest_av[
                    "sequence_type"
                ],

            }

            if nearest_av
            else None
        ),
    }


# ============================================================
# CLEAN CANDIDATES
# ============================================================

def deduplicate(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    ordered = sorted(
        candidates,
        key=lambda x: (
            x["timestamp"],
            -x["score"],
        ),
    )

    result = []

    for candidate in ordered:

        if not result:

            result.append(
                candidate
            )

            continue

        previous = result[-1]

        if (
            candidate["timestamp"]
            - previous["timestamp"]
            < 0.05
        ):

            if (
                candidate["score"]
                > previous["score"]
            ):

                result[-1] = candidate

        else:

            result.append(
                candidate
            )

    return result


# ============================================================
# DISTRIBUTED CUT SELECTION
# ============================================================

def select_cuts(
    candidates: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    duration: float,
) -> List[Dict[str, Any]]:

    candidates = deduplicate(
        candidates
    )

    selected = []

    current_time = 0.0

    # --------------------------------------------------------
    # Walk through the timeline rather than taking the
    # globally highest 46 candidates.
    #
    # This prevents the dense onset regions from consuming
    # the entire cut budget.
    # --------------------------------------------------------

    safety = 0

    while (
        current_time
        < duration - 0.15
        and safety < 200
    ):

        safety += 1

        role = segment_role(
            current_time + 0.08,
            segments,
        )

        settings = ROLE_SETTINGS.get(
            role,
            ROLE_SETTINGS["cinematic_zone"],
        )

        min_gap = max(
            CUT_MIN_GAP,
            settings["min_gap"],
        )

        preferred_gap = (
            settings["preferred_gap"]
        )

        max_gap = settings["max_gap"]

        search_start = (
            current_time
            + min_gap
        )

        search_end = min(
            duration,
            current_time
            + max_gap,
        )

        pool = [
            candidate
            for candidate in candidates
            if (
                search_start
                <= candidate["timestamp"]
                <= search_end
            )
        ]

        if not pool:

            # Look slightly beyond the normal window.
            extended_end = min(
                duration,
                current_time
                + max_gap
                + 0.45,
            )

            pool = [
                candidate
                for candidate in candidates
                if (
                    search_start
                    <= candidate["timestamp"]
                    <= extended_end
                )
            ]

        if not pool:
            break

        # ----------------------------------------------------
        # Prefer candidates close to the desired shot length,
        # but still respect editorial score.
        # ----------------------------------------------------

        def local_score(
            candidate: Dict[str, Any],
        ) -> float:

            distance = abs(
                (
                    candidate["timestamp"]
                    - current_time
                )
                - preferred_gap
            )

            timing_score = max(
                0.0,
                1.0
                - distance
                / max(
                    preferred_gap,
                    0.1,
                ),
            )

            return (
                0.72
                * candidate["score"]
                + 0.28
                * timing_score
            )

        chosen = max(
            pool,
            key=local_score,
        )

        selected.append(
            chosen
        )

        current_time = (
            chosen["timestamp"]
        )

    # --------------------------------------------------------
    # Remove accidental duplicates.
    # --------------------------------------------------------

    cleaned = []

    for item in selected:

        if not cleaned:

            cleaned.append(item)
            continue

        if (
            item["timestamp"]
            - cleaned[-1]["timestamp"]
            >= CUT_MIN_GAP
        ):

            cleaned.append(item)

    # --------------------------------------------------------
    # If distributed pass produced too few cuts,
    # fill from strongest candidates.
    # --------------------------------------------------------

    if len(cleaned) < TARGET_CUTS:

        ranked = sorted(
            candidates,
            key=lambda x: x["score"],
            reverse=True,
        )

        for candidate in ranked:

            if len(cleaned) >= TARGET_CUTS:
                break

            t = candidate["timestamp"]

            if all(
                abs(
                    t
                    - x["timestamp"]
                )
                >= CUT_MIN_GAP
                for x in cleaned
            ):

                cleaned.append(
                    candidate
                )

    # --------------------------------------------------------
    # If more than target, trim lowest-quality cuts while
    # preserving timeline spacing.
    # --------------------------------------------------------

    if len(cleaned) > TARGET_CUTS:

        ranked = sorted(
            cleaned,
            key=lambda x: x["score"],
            reverse=True,
        )

        keep = []

        for candidate in ranked:

            if len(keep) >= TARGET_CUTS:
                break

            t = candidate["timestamp"]

            if all(
                abs(
                    t
                    - x["timestamp"]
                )
                >= CUT_MIN_GAP
                for x in keep
            ):

                keep.append(
                    candidate
                )

        cleaned = keep

    return sorted(
        cleaned,
        key=lambda x: x["timestamp"],
    )


# ============================================================
# EFFECT LANE
# ============================================================

def effect_priority(
    event: Dict[str, Any],
) -> float:

    score = f(
        event["anchor"].get(
            "score",
            0.0,
        )
    )

    sequence = event.get(
        "sequence_type",
        "ACCENT",
    )

    role_bonus = {

        "BUILD_TO_IMPACT": 0.12,

        "DIRECT_IMPACT": 0.10,

        "MOTION_ACCENT": 0.05,

        "ACCENT": 0.02,

    }.get(
        sequence,
        0.0,
    )

    return score + role_bonus


def build_effect_sequence(
    event: Dict[str, Any],
) -> List[str]:

    sequence = event.get(
        "sequence_type",
        "ACCENT",
    )

    if sequence == "BUILD_TO_IMPACT":

        return [
            "pre_zoom",
            "directional_motion",
            "impact_snap",
            "short_shake_burst",
            "optional_flash",
            "cut_or_reset",
        ]

    if sequence == "DIRECT_IMPACT":

        return [
            "impact_snap",
            "short_shake_burst",
            "optional_flash",
            "cut_or_reset",
        ]

    if sequence == "MOTION_ACCENT":

        return [
            "directional_motion",
            "motion_blur",
            "resolve_cut",
        ]

    return [
        "controlled_accent",
        "short_hold",
    ]


def select_effect_events(
    av_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    # --------------------------------------------------------
    # Reference grammar events are the ONLY source of effect
    # events in V3.
    # --------------------------------------------------------

    ranked = sorted(
        av_events,
        key=effect_priority,
        reverse=True,
    )

    selected = []

    for event in ranked:

        timestamp = event["timestamp"]

        # Avoid multiple effect events being stacked
        # almost on top of one another.
        too_close = any(
            abs(
                timestamp
                - existing["timestamp"]
            )
            < 0.18
            for existing in selected
        )

        if too_close:
            continue

        if (
            event["anchor"]["score"]
            < 0.34
        ):
            continue

        selected.append(
            event
        )

    return sorted(
        selected,
        key=lambda x: x["timestamp"],
    )


# ============================================================
# LOCK EFFECTS TO MUSIC
# ============================================================

def lock_effect_to_music(
    event: Dict[str, Any],
    strong_onsets: List[float],
    beats: List[float],
) -> Dict[str, Any]:

    timestamp = event["timestamp"]

    nearest_strong = nearest_time(
        timestamp,
        strong_onsets,
    )

    nearest_beat = nearest_time(
        timestamp,
        beats,
    )

    strong_distance = (
        abs(
            timestamp
            - nearest_strong
        )
        if nearest_strong is not None
        else 999.0
    )

    beat_distance = (
        abs(
            timestamp
            - nearest_beat
        )
        if nearest_beat is not None
        else 999.0
    )

    # Strong onset is preferred if very close.
    if strong_distance <= 0.12:

        locked_time = nearest_strong
        lock_type = "strong_onset"

    elif beat_distance <= 0.08:

        locked_time = nearest_beat
        lock_type = "beat"

    else:

        locked_time = timestamp
        lock_type = "event_anchor"

    return {

        "timestamp": round(
            float(locked_time),
            4,
        ),

        "reference_timestamp": round(
            timestamp,
            4,
        ),

        "lock_type": lock_type,

        "distance": round(
            min(
                strong_distance,
                beat_distance,
            ),
            4,
        ),

        "event_role": event[
            "anchor"
        ]["role"],

        "sequence_type": event[
            "sequence_type"
        ],

        "event_score": event[
            "anchor"
        ]["score"],

        "edit_sequence": build_effect_sequence(
            event
        ),

    }


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
    )

    points.append(
        duration
    )

    points = sorted(
        set(
            round(
                x,
                4,
            )
            for x in points
        )
    )

    windows = []

    for index, (a, b) in enumerate(
        zip(
            points[:-1],
            points[1:],
        ),
        start=1,
    ):

        windows.append({

            "shot_id": index,

            "start": a,

            "end": b,

            "duration": round(
                b - a,
                4,
            ),

        })

    return windows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("🎵 MUSIC LOCK DIRECTOR V3")
    print("=" * 72)

    music = load_json(
        MUSIC_PLAN
    )

    grammar = load_json(
        AV_GRAMMAR
    )

    segments = get_segments(
        music
    )

    av_events_raw = grammar.get(
        "events",
        [],
    )

    av_events = []

    for event in av_events_raw:
        anchor = event.get("anchor", {})
        av_events.append({
            **event,
            "timestamp": float(anchor.get("timestamp", 0.0)),
            "score": float(anchor.get("score", 0.0)),
            "role": anchor.get("role", "UNKNOWN"),
        })

    arrays = recursive_find_arrays(
        music,
        {
            "beats",
            "strong_onsets",
        },
    )

    beats = normalize_times(
        arrays.get(
            "beats",
            [],
        )
    )

    strong_onsets = normalize_times(
        arrays.get(
            "strong_onsets",
            [],
        )
    )

    if segments:

        duration = max(
            f(
                segment.get(
                    "end"
                )
            )
            for segment in segments
        )

    else:

        duration = 59.072

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
        f"AV grammar events: {len(av_events)}"
    )

    # ========================================================
    # CUT LANE
    # ========================================================

    raw_candidates = build_cut_candidates(
        music,
        av_events,
    )

    scored_candidates = [

        score_cut_candidate(
            candidate,
            segments,
            strong_onsets,
            beats,
            av_events,
        )

        for candidate
        in raw_candidates
    ]

    cuts = select_cuts(
        scored_candidates,
        segments,
        duration,
    )

    # ========================================================
    # EFFECT LANE
    # ========================================================

    effects = select_effect_events(
        av_events
    )

    locked_effects = [

        lock_effect_to_music(
            event,
            strong_onsets,
            beats,
        )

        for event
        in effects
    ]

    # ========================================================
    # SHOT WINDOWS
    # ========================================================

    shots = build_shot_windows(
        cuts,
        duration,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    action_counts = {

        "CUT": len(cuts),

        "REFERENCE_AV_EFFECT":
            len(locked_effects),

    }

    sequence_counts = {}

    for effect in locked_effects:

        sequence = effect[
            "sequence_type"
        ]

        sequence_counts[sequence] = (
            sequence_counts.get(
                sequence,
                0,
            )
            + 1
        )

    role_counts = {}

    for effect in locked_effects:

        role = effect[
            "event_role"
        ]

        role_counts[role] = (
            role_counts.get(
                role,
                0,
            )
            + 1
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    output = {

        "meta": {

            "version": "v3",

            "purpose": (
                "Separates the music cut lane from the "
                "reference audiovisual effect lane."
            ),

            "duration": duration,

            "reference_editorial_shots": 44,

            "target_cuts": TARGET_CUTS,

        },

        "music": {

            "segments": len(
                segments
            ),

            "beats": len(
                beats
            ),

            "strong_onsets": len(
                strong_onsets
            ),

        },

        "summary": {

            "raw_cut_candidates":
                len(raw_candidates),

            "selected_cuts":
                len(cuts),

            "reference_effect_events":
                len(locked_effects),

            "shots":
                len(shots),

            "action_counts":
                action_counts,

            "effect_sequence_counts":
                sequence_counts,

            "effect_role_counts":
                role_counts,

        },

        "cut_lane": {

            "cuts": cuts,

        },

        "effect_lane": {

            "events": locked_effects,

        },

        "shot_windows": shots,

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

    # ========================================================
    # REPORT
    # ========================================================

    print("\n" + "=" * 72)
    print("✅ MUSIC LOCK V3 COMPLETE")
    print("=" * 72)

    print(
        f"\nSelected cuts           : "
        f"{len(cuts)}"
    )

    print(
        f"Reference AV effects   : "
        f"{len(locked_effects)}"
    )

    print(
        f"Final shot windows     : "
        f"{len(shots)}"
    )

    print("\nCut / effect structure:")

    print(
        f"  CUT                  : "
        f"{len(cuts)}"
    )

    print(
        f"  AV EFFECT EVENTS     : "
        f"{len(locked_effects)}"
    )

    print("\nEffect sequence distribution:")

    for key, value in sorted(
        sequence_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"  {key:<24}: {value}"
        )

    print("\nEffect role distribution:")

    for key, value in sorted(
        role_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"  {key:<24}: {value}"
        )

    print("\nFirst 20 cuts:")

    for cut in cuts[:20]:

        print(
            f"  "
            f"{cut['timestamp']:7.3f}s | "
            f"{cut['role']:<20} | "
            f"{cut['source']:<16}"
        )

    print("\nReference AV effect events:")

    for effect in locked_effects:

        print(
            f"  "
            f"{effect['timestamp']:7.3f}s | "
            f"{effect['sequence_type']:<20} | "
            f"{effect['event_role']:<20} | "
            f"lock={effect['lock_type']}"
        )

    print(
        f"\nOutput: {OUTPUT}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()