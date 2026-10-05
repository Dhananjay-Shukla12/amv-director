from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

MUSIC_PLAN = (
    BASE_DIR
    / "data/outputs/music_director_plan.json"
)

SHOT_MATCH_PLAN = (
    BASE_DIR
    / "data/outputs/reference_conditioned_shot_plan.json"
)

REFERENCE_SHOT_MAP = (
    BASE_DIR
    / "data/outputs/reference_shot_map.json"
)

REFERENCE_RHYTHM = (
    BASE_DIR
    / "data/outputs/reference_rhythm_map.json"
)

OUTPUT_JSON = (
    BASE_DIR
    / "data/outputs/music_locked_edit_plan.json"
)


# ============================================================
# SETTINGS
# ============================================================

EPSILON = 0.025

# Musical event windows.
# These are deliberately small so edits stay close to the
# actual musical event rather than drifting across the beat.
BEAT_WINDOW = 0.055
STRONG_EVENT_WINDOW = 0.090

# Prevent absurdly dense cuts even when many audio events exist.
MIN_CUT_GAP = 0.12

# A very short event can still be used as an accent/flash,
# but should not automatically become a new source shot.
ACCENT_MIN_DURATION = 0.10

# Maximum number of micro-edits inside a single broad segment.
MAX_MICRO_EDITS_PER_SEGMENT = 4


# ============================================================
# HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:
        return default


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:

    return max(
        low,
        min(high, value),
    )


def load_json(path: Path) -> Any:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def normalize_event_time(
    event: Any,
) -> Optional[float]:

    if isinstance(event, (int, float)):
        return safe_float(event)

    if not isinstance(event, dict):
        return None

    for key in (
        "time",
        "timestamp",
        "start",
        "position",
        "onset",
        "beat_time",
    ):

        if key in event:

            value = event[key]

            if isinstance(
                value,
                (int, float),
            ):

                return safe_float(value)

    return None


# ============================================================
# RECURSIVE EVENT EXTRACTION
# ============================================================

def collect_time_events(
    obj: Any,
    target_keys: Tuple[str, ...],
) -> List[float]:

    found: List[float] = []

    def walk(value: Any):

        if isinstance(value, dict):

            for key, child in value.items():

                key_lower = str(key).lower()

                if key_lower in target_keys:

                    if isinstance(
                        child,
                        list,
                    ):

                        for item in child:

                            timestamp = (
                                normalize_event_time(
                                    item
                                )
                            )

                            if timestamp is not None:
                                found.append(
                                    timestamp
                                )

                    else:

                        timestamp = (
                            normalize_event_time(
                                child
                            )
                        )

                        if timestamp is not None:
                            found.append(
                                timestamp
                            )

                walk(child)

        elif isinstance(value, list):

            for item in value:
                walk(item)

    walk(obj)

    return sorted(
        set(
            round(x, 4)
            for x in found
            if x >= 0
        )
    )


# ============================================================
# EXTRACT DIRECT MUSIC EVENTS
# ============================================================

def extract_music_events(
    music_data: Dict,
    rhythm_data: Dict,
) -> Dict[str, List[float]]:

    # --------------------------------------------------------
    # Strong / editorial events
    # --------------------------------------------------------

    strong_candidates = []

    for key in (
        "strong_onsets",
        "strong_events",
        "strong_beats",
        "accent_times",
        "impact_times",
    ):

        strong_candidates.extend(
            collect_time_events(
                music_data,
                (key,),
            )
        )

        strong_candidates.extend(
            collect_time_events(
                rhythm_data,
                (key,),
            )
        )

    # --------------------------------------------------------
    # Beat events
    # --------------------------------------------------------

    beat_candidates = []

    for key in (
        "beats",
        "beat_times",
        "beat_timestamps",
        "beat_positions",
    ):

        beat_candidates.extend(
            collect_time_events(
                music_data,
                (key,),
            )
        )

        beat_candidates.extend(
            collect_time_events(
                rhythm_data,
                (key,),
            )
        )

    # --------------------------------------------------------
    # General onsets
    # --------------------------------------------------------

    onset_candidates = []

    for key in (
        "onsets",
        "onset_times",
        "audio_onsets",
        "events",
    ):

        onset_candidates.extend(
            collect_time_events(
                music_data,
                (key,),
            )
        )

        onset_candidates.extend(
            collect_time_events(
                rhythm_data,
                (key,),
            )
        )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    strong = sorted(
        set(
            round(x, 4)
            for x in strong_candidates
            if x >= 0
        )
    )

    beats = sorted(
        set(
            round(x, 4)
            for x in beat_candidates
            if x >= 0
        )
    )

    onsets = sorted(
        set(
            round(x, 4)
            for x in onset_candidates
            if x >= 0
        )
    )

    return {
        "strong_events": strong,
        "beats": beats,
        "onsets": onsets,
    }


# ============================================================
# ROLE HELPERS
# ============================================================

def editorial_role(
    segment: Dict,
) -> str:

    return str(
        segment.get(
            "editorial_role",
            "free",
        )
    )


def sync_type(
    segment: Dict,
) -> str:

    return str(
        segment.get(
            "sync_type",
            "free",
        )
    )


def sync_priority(
    segment: Dict,
) -> str:

    return str(
        segment.get(
            "sync_priority",
            "low",
        )
    )


# ============================================================
# EVENT LOOKUP
# ============================================================

def events_inside(
    events: List[float],
    start: float,
    end: float,
) -> List[float]:

    return [
        event
        for event in events
        if (
            event >= start - EPSILON
            and event <= end + EPSILON
        )
    ]


def nearest_event(
    events: List[float],
    timestamp: float,
    window: float,
) -> Optional[float]:

    if not events:
        return None

    candidates = [
        event
        for event in events
        if abs(event - timestamp) <= window
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda event: abs(
            event - timestamp
        ),
    )


# ============================================================
# REFERENCE STRUCTURE
# ============================================================

def reference_density_for_interval(
    reference_shots: List[Dict],
    start: float,
    end: float,
) -> Dict[str, Any]:

    shots = []

    for shot in reference_shots:

        shot_start = safe_float(
            shot.get("start")
        )

        shot_end = safe_float(
            shot.get("end")
        )

        overlap = max(
            0.0,
            min(
                end,
                shot_end,
            )
            - max(
                start,
                shot_start,
            ),
        )

        if overlap > 0:
            shots.append(
                shot
            )

    if not shots:

        return {
            "shot_count": 0,
            "mean_duration": 0.0,
            "density": 0.0,
        }

    durations = [
        safe_float(
            shot.get(
                "duration",
                safe_float(
                    shot.get("end")
                )
                - safe_float(
                    shot.get("start")
                ),
            )
        )
        for shot in shots
    ]

    duration = max(
        0.001,
        end - start,
    )

    return {
        "shot_count": len(shots),
        "mean_duration": (
            sum(durations)
            / len(durations)
        ),
        "density": (
            len(shots)
            / duration
        ),
    }


# ============================================================
# DECISION LOGIC
# ============================================================

def choose_action(
    segment: Dict,
    event_kind: str,
    relative_strength: float,
    reference_density: Dict[str, Any],
) -> str:

    role = editorial_role(
        segment
    ).lower()

    priority = sync_priority(
        segment
    ).lower()

    segment_sync = sync_type(
        segment
    ).lower()

    # --------------------------------------------------------
    # Strongest musical moments
    # --------------------------------------------------------

    if event_kind == "strong":

        if role in (
            "impact_zone",
            "action_zone",
        ):

            return "IMPACT"

        if priority == "high":

            return "CUT"

        return "ACCENT"

    # --------------------------------------------------------
    # Beat
    # --------------------------------------------------------

    if event_kind == "beat":

        if role == "cinematic_zone":
            return "HOLD"

        if role == "build_zone":
            return "ACCENT"

        if role == "impact_zone":
            return "CUT"

        if role == "action_zone":
            return "CUT"

        if role == "accent_zone":
            return "ACCENT"

        if segment_sync == "strong_onset":

            return "CUT"

        return "ACCENT"

    # --------------------------------------------------------
    # General onset
    # --------------------------------------------------------

    if event_kind == "onset":

        if role == "cinematic_zone":
            return "HOLD"

        if role == "build_zone":
            return "ACCENT"

        if role == "impact_zone":

            if relative_strength > 0.70:
                return "CUT"

            return "ACCENT"

        if role == "action_zone":

            return "ACCENT"

        return "HOLD"

    return "HOLD"


# ============================================================
# EVENT PRIORITY
# ============================================================

def event_strength(
    event_kind: str,
    segment: Dict,
) -> float:

    editorial_intensity = safe_float(
        segment.get(
            "editorial_intensity",
            0.0,
        )
    )

    music_energy = safe_float(
        segment.get(
            "music_energy",
            0.0,
        )
    )

    priority = (
        sync_priority(
            segment
        ).lower()
    )

    score = (
        editorial_intensity * 0.50
        + music_energy * 0.35
    )

    if priority == "high":
        score += 0.20

    elif priority == "medium":
        score += 0.10

    return clamp(
        score
    )


# ============================================================
# CUT FILTER
# ============================================================

def enforce_cut_spacing(
    decisions: List[Dict],
) -> List[Dict]:

    decisions.sort(
        key=lambda item: safe_float(
            item["time"]
        )
    )

    result = []

    last_cut_time: Optional[float] = None

    for decision in decisions:

        if decision[
            "action"
        ] not in (
            "CUT",
            "IMPACT",
        ):

            result.append(
                decision
            )

            continue

        time = safe_float(
            decision["time"]
        )

        if last_cut_time is None:

            result.append(
                decision
            )

            last_cut_time = time

            continue

        gap = (
            time
            - last_cut_time
        )

        if gap >= MIN_CUT_GAP:

            result.append(
                decision
            )

            last_cut_time = time

        else:

            # Prefer IMPACT over CUT.
            existing = None

            for previous in reversed(
                result
            ):

                previous_time = safe_float(
                    previous["time"]
                )

                if abs(
                    previous_time
                    - time
                ) < MIN_CUT_GAP:

                    existing = previous
                    break

            if existing is not None:

                if (
                    decision["action"]
                    == "IMPACT"
                ):

                    existing.update(
                        decision
                    )

    return result


# ============================================================
# BUILD SEGMENT DECISIONS
# ============================================================

def build_segment_decisions(
    segment: Dict,
    matched: Optional[Dict],
    music_events: Dict[str, List[float]],
    reference_density: Dict[str, Any],
) -> Dict[str, Any]:

    start = safe_float(
        segment.get("start")
    )

    end = safe_float(
        segment.get("end")
    )

    duration = max(
        0.001,
        end - start,
    )

    role = editorial_role(
        segment
    )

    strength = event_strength(
        "segment",
        segment,
    )

    local_decisions: List[Dict] = []

    # --------------------------------------------------------
    # Find musical events in this segment
    # --------------------------------------------------------

    strong_events = events_inside(
        music_events[
            "strong_events"
        ],
        start,
        end,
    )

    beat_events = events_inside(
        music_events[
            "beats"
        ],
        start,
        end,
    )

    onset_events = events_inside(
        music_events[
            "onsets"
        ],
        start,
        end,
    )

    # --------------------------------------------------------
    # Strong events get highest precedence
    # --------------------------------------------------------

    used_times = set()

    for event in strong_events:

        action = choose_action(
            segment,
            "strong",
            strength,
            reference_density,
        )

        local_decisions.append(
            {
                "time": round(
                    event,
                    4,
                ),
                "action": action,
                "event": "strong_onset",
                "role": role,
                "sync_priority": sync_priority(
                    segment
                ),
            }
        )

        used_times.add(
            round(event, 4)
        )

    # --------------------------------------------------------
    # Beats
    # --------------------------------------------------------

    for event in beat_events:

        rounded = round(
            event,
            4,
        )

        if rounded in used_times:
            continue

        action = choose_action(
            segment,
            "beat",
            strength,
            reference_density,
        )

        local_decisions.append(
            {
                "time": rounded,
                "action": action,
                "event": "beat",
                "role": role,
                "sync_priority": sync_priority(
                    segment
                ),
            }
        )

        used_times.add(
            rounded
        )

    # --------------------------------------------------------
    # General onsets
    # --------------------------------------------------------

    for event in onset_events:

        rounded = round(
            event,
            4,
        )

        if rounded in used_times:
            continue

        action = choose_action(
            segment,
            "onset",
            strength,
            reference_density,
        )

        local_decisions.append(
            {
                "time": rounded,
                "action": action,
                "event": "onset",
                "role": role,
                "sync_priority": sync_priority(
                    segment
                ),
            }
        )

        used_times.add(
            rounded
        )

    # --------------------------------------------------------
    # No event = preserve the reference-derived hold
    # --------------------------------------------------------

    if not local_decisions:

        if role == "cinematic_zone":

            fallback_action = "HOLD"

        elif role == "build_zone":

            fallback_action = "BUILD"

        elif role in (
            "impact_zone",
            "action_zone",
        ):

            fallback_action = "HOLD"

        else:

            fallback_action = "HOLD"

        local_decisions.append(
            {
                "time": round(
                    start,
                    4,
                ),
                "action": fallback_action,
                "event": "segment_start",
                "role": role,
                "sync_priority": sync_priority(
                    segment
                ),
            }
        )

    # --------------------------------------------------------
    # Limit micro-decisions
    # --------------------------------------------------------

    local_decisions.sort(
        key=lambda x: safe_float(
            x["time"]
        )
    )

    # Keep strongest events first when too dense.
    if len(local_decisions) > MAX_MICRO_EDITS_PER_SEGMENT:

        priority_order = {
            "IMPACT": 5,
            "CUT": 4,
            "ACCENT": 3,
            "BUILD": 2,
            "HOLD": 1,
        }

        local_decisions.sort(
            key=lambda item: (
                priority_order.get(
                    item["action"],
                    0,
                ),
                item["event"]
                == "strong_onset",
            ),
            reverse=True,
        )

        local_decisions = local_decisions[
            :MAX_MICRO_EDITS_PER_SEGMENT
        ]

        local_decisions.sort(
            key=lambda x: safe_float(
                x["time"]
            )
        )

    # --------------------------------------------------------
    # Attach source match
    # --------------------------------------------------------

    source = None

    if matched:

        source = matched.get(
            "source"
        )

    # --------------------------------------------------------
    # Segment-level record
    # --------------------------------------------------------

    return {
        "segment": segment.get(
            "segment"
        ),
        "start": round(
            start,
            4,
        ),
        "end": round(
            end,
            4,
        ),
        "duration": round(
            duration,
            4,
        ),

        "music": {
            "editorial_role": role,
            "sync_type": sync_type(
                segment
            ),
            "sync_priority": sync_priority(
                segment
            ),
            "music_energy": safe_float(
                segment.get(
                    "music_energy"
                )
            ),
            "editorial_intensity": safe_float(
                segment.get(
                    "editorial_intensity"
                )
            ),
            "intensity_label": segment.get(
                "intensity_label"
            ),
            "transition_context": segment.get(
                "transition_context"
            ),
            "effect_budget": segment.get(
                "effect_budget"
            ),
        },

        "reference": {
            "density": reference_density,
        },

        "source": source,

        "edit_decisions": local_decisions,
    }


# ============================================================
# SECONDARY CUT TIMELINE
# ============================================================

def build_global_timeline(
    segments: List[Dict],
) -> List[Dict]:

    all_decisions = []

    for segment in segments:

        for decision in segment[
            "edit_decisions"
        ]:

            item = dict(
                decision
            )

            item[
                "segment"
            ] = segment[
                "segment"
            ]

            all_decisions.append(
                item
            )

    return enforce_cut_spacing(
        all_decisions
    )


# ============================================================
# BUILD SHOT WINDOWS
# ============================================================

# def build_shot_windows(
#     timeline: List[Dict],
#     total_duration: float,
#     first_source_match_by_segment: Dict,
# ) -> List[Dict]:
def build_shot_windows(
    timeline: List[Dict],
    segments: List[Dict],
    total_duration: float,
    first_source_match_by_segment: Dict,
) -> List[Dict]:
    # --------------------------------------------------------
    # Only CUT/IMPACT create a new source-shot boundary.
    # ACCENT/HOLD/BUILD modify the current shot.
    # --------------------------------------------------------

    boundaries = [
        safe_float(
            item["time"]
        )
        for item in timeline
        if item["action"]
        in (
            "CUT",
            "IMPACT",
        )
    ]

    boundaries = sorted(
        set(
            round(
                x,
                4,
            )
            for x in boundaries
            if (
                x > 0
                and x < total_duration
            )
        )
    )

    if not boundaries:

        return [
            {
                "start": 0.0,
                "end": total_duration,
                "duration": total_duration,
                "cut_trigger": "none",
                "source_match": (
                    first_source_match_by_segment.get(
                        1
                    )
                ),
            }
        ]

    windows = []

    cursor = 0.0

    for boundary in boundaries:

        if boundary <= cursor + EPSILON:
            continue

        windows.append(
            {
                "start": round(
                    cursor,
                    4,
                ),
                "end": round(
                    boundary,
                    4,
                ),
                "duration": round(
                    boundary - cursor,
                    4,
                ),
            }
        )

        cursor = boundary

    if cursor < total_duration:

        windows.append(
            {
                "start": round(
                    cursor,
                    4,
                ),
                "end": round(
                    total_duration - cursor,
                    4,
                ),
                "end": round(
                    total_duration,
                    4,
                ),
            }
        )

    # --------------------------------------------------------
    # Correct duplicated end key from fallback construction.
    # --------------------------------------------------------

    for window in windows:

        window["duration"] = round(
            safe_float(
                window["end"]
            )
            - safe_float(
                window["start"]
            ),
            4,
        )

    # --------------------------------------------------------
    # Pick source match by timeline midpoint.
    # --------------------------------------------------------

    for window in windows:

        midpoint = (
            safe_float(
                window["start"]
            )
            + safe_float(
                window["duration"]
            ) * 0.5
        )

        best_segment = None

        for segment in segments:

            start = safe_float(
                segment["start"]
            )

            end = safe_float(
                segment["end"]
            )

            if (
                start
                <= midpoint
                <= end
            ):

                best_segment = segment
                break

        if best_segment:

            window[
                "music_segment"
            ] = best_segment.get(
                "segment"
            )

            window[
                "source_match"
            ] = best_segment.get(
                "source"
            )

            window[
                "music_role"
            ] = best_segment[
                "music"
            ].get(
                "editorial_role"
            )

        else:

            window[
                "music_segment"
            ] = None

    return windows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("🎵 MUSIC-LOCK DIRECTOR")
    print("=" * 72)
    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    music_data = load_json(
        MUSIC_PLAN
    )

    shot_match_data = load_json(
        SHOT_MATCH_PLAN
    )

    reference_data = load_json(
        REFERENCE_SHOT_MAP
    )

    rhythm_data = (
        load_json(
            REFERENCE_RHYTHM
        )
        if REFERENCE_RHYTHM.exists()
        else {}
    )

    segments = music_data.get(
        "segments",
        [],
    )

    matches = shot_match_data.get(
        "matches",
        [],
    )

    # Reference shots can have different key names in
    # previous project versions.

    reference_shots = (
        reference_data.get(
            "shots",
            reference_data.get(
                "editorial_shots",
                [],
            ),
        )
    )

    if not segments:

        raise RuntimeError(
            "No music segments found."
        )

    if not matches:

        raise RuntimeError(
            "No shot matches found."
        )

    print(
        f"Music segments : {len(segments)}"
    )

    print(
        f"Shot matches   : {len(matches)}"
    )

    print(
        f"Reference shots: {len(reference_shots)}"
    )

    # --------------------------------------------------------
    # Extract actual musical events
    # --------------------------------------------------------

    music_events = (
        extract_music_events(
            music_data,
            rhythm_data,
        )
    )

    print()

    print(
        "Music events:"
    )

    print(
        f"  Strong events : "
        f"{len(music_events['strong_events'])}"
    )

    print(
        f"  Beats         : "
        f"{len(music_events['beats'])}"
    )

    print(
        f"  Onsets        : "
        f"{len(music_events['onsets'])}"
    )

    # --------------------------------------------------------
    # Match segment -> shot-plan entry
    # --------------------------------------------------------

    matches_by_segment = {}

    for match in matches:

        index = match.get(
            "target_index"
        )

        if index is not None:

            matches_by_segment[
                int(index)
            ] = match

    # --------------------------------------------------------
    # Determine total duration
    # --------------------------------------------------------

    total_duration = safe_float(
        music_data.get(
            "music",
            {},
        ).get(
            "duration",
            0.0,
        )
    )

    if total_duration <= 0:

        total_duration = max(
            safe_float(
                segment.get("end")
            )
            for segment in segments
        )

    # --------------------------------------------------------
    # Build locked segments
    # --------------------------------------------------------

    locked_segments = []

    for segment in segments:

        segment_number = int(
            segment.get(
                "segment",
                len(locked_segments) + 1,
            )
        )

        start = safe_float(
            segment.get("start")
        )

        end = safe_float(
            segment.get("end")
        )

        density = (
            reference_density_for_interval(
                reference_shots,
                start,
                end,
            )
        )

        matched = matches_by_segment.get(
            segment_number
        )

        locked = build_segment_decisions(
            segment=segment,
            matched=matched,
            music_events=music_events,
            reference_density=density,
        )

        locked_segments.append(
            locked
        )

    # --------------------------------------------------------
    # Global timeline
    # --------------------------------------------------------

    global_timeline = (
        build_global_timeline(
            locked_segments
        )
    )

    # --------------------------------------------------------
    # Add source shot windows
    # --------------------------------------------------------

    source_match_map = {
        int(
            item["segment"]
        ): item.get(
            "source"
        )
        for item in locked_segments
    }

    # shot_windows = build_shot_windows(
    #     global_timeline,
    #     total_duration,
    #     source_match_map,
    # )
    shot_windows = build_shot_windows(
    global_timeline,
    locked_segments,
    total_duration,
    source_match_map,
    )
    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    action_counts = {}

    for item in global_timeline:

        action = item[
            "action"
        ]

        action_counts[action] = (
            action_counts.get(
                action,
                0,
            )
            + 1
        )

    # --------------------------------------------------------
    # Strong sync stats
    # --------------------------------------------------------

    exact_music_locked = 0

    event_times = set(
        round(x, 3)
        for x in (
            music_events[
                "strong_events"
            ]
            + music_events[
                "beats"
            ]
        )
    )

    for decision in global_timeline:

        if decision[
            "action"
        ] not in (
            "CUT",
            "IMPACT",
            "ACCENT",
        ):
            continue

        decision_time = safe_float(
            decision["time"]
        )

        nearest = nearest_event(
            list(event_times),
            decision_time,
            STRONG_EVENT_WINDOW,
        )

        if nearest is not None:

            exact_music_locked += 1

    # --------------------------------------------------------
    # Build output
    # --------------------------------------------------------

    output = {

        "metadata": {

            "engine":
                "music_lock_director_v1",

            "total_duration":
                round(
                    total_duration,
                    4,
                ),

            "music_segments":
                len(segments),

            "locked_segments":
                len(
                    locked_segments
                ),

            "global_edit_decisions":
                len(
                    global_timeline
                ),

            "shot_windows":
                len(
                    shot_windows
                ),

            "music_events": {
                "strong":
                    len(
                        music_events[
                            "strong_events"
                        ]
                    ),
                "beats":
                    len(
                        music_events[
                            "beats"
                        ]
                    ),
                "onsets":
                    len(
                        music_events[
                            "onsets"
                        ]
                    ),
            },

            "action_counts":
                action_counts,

            "music_locked_visual_actions":
                exact_music_locked,
        },

        "music_events":
            music_events,

        "segments":
            locked_segments,

        "timeline":
            global_timeline,

        "shot_windows":
            shot_windows,
    }

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("✅ MUSIC-LOCKED EDIT PLAN COMPLETE")
    print("=" * 72)

    print()

    print(
        f"Duration              : "
        f"{total_duration:.3f}s"
    )

    print(
        f"Segments              : "
        f"{len(locked_segments)}"
    )

    print(
        f"Edit decisions        : "
        f"{len(global_timeline)}"
    )

    print(
        f"Shot windows          : "
        f"{len(shot_windows)}"
    )

    print()

    print(
        "Actions:"
    )

    for action, count in sorted(
        action_counts.items()
    ):

        print(
            f"  {action:10s}: {count}"
        )

    print()

    print(
        "Music-locked actions  : "
        f"{exact_music_locked}"
    )

    print()

    print(
        f"Output:\n{OUTPUT_JSON}"
    )

    # --------------------------------------------------------
    # Preview timeline
    # --------------------------------------------------------

    print()
    print(
        "FIRST EDIT DECISIONS:"
    )

    for item in global_timeline[:25]:

        print(
            f"  {safe_float(item['time']):7.3f}s | "
            f"{item['action']:7s} | "
            f"{item['event']:14s} | "
            f"{item['role']}"
        )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()