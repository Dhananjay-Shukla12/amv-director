import json
from pathlib import Path
from statistics import mean, median

import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "reference_shot_map.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "reference_edit_grammar.json"
)


# ============================================================
# SETTINGS
# ============================================================

MICRO = 0.25
VERY_SHORT = 0.50
SHORT = 0.90
MEDIUM = 1.80
LONG = 4.00

WINDOW = 5


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return default
            value = value[0]

        return float(value)

    except Exception:
        return default


def first_value(d, keys, default=0.0):
    for key in keys:
        if key in d and d[key] is not None:
            return safe_float(d[key], default)

    return default


def clamp(x, low=0.0, high=1.0):
    return max(low, min(high, x))


def normalize(x, low, high):
    if high <= low:
        return 0.0

    return clamp(
        (x - low) / (high - low)
    )


# ============================================================
# SHOT DURATION
# ============================================================

def duration_class(duration):

    if duration < MICRO:
        return "micro"

    if duration < VERY_SHORT:
        return "very_short"

    if duration < SHORT:
        return "short"

    if duration < MEDIUM:
        return "medium"

    if duration < LONG:
        return "long"

    return "epic"


def pacing_class(duration):

    if duration < VERY_SHORT:
        return "extreme"

    if duration < SHORT:
        return "fast"

    if duration < MEDIUM:
        return "medium"

    return "slow"


# ============================================================
# EVENT COUNT
# ============================================================

def event_count(shot):

    events = shot.get("events", [])

    if isinstance(events, list):
        return len(events)

    if isinstance(events, (int, float)):
        return int(events)

    return first_value(
        shot,
        [
            "event_count",
            "num_events"
        ],
        0
    )


# ============================================================
# SHOT METRICS
# ============================================================

def get_motion(shot):

    return first_value(
        shot,
        [
            "motion",
            "motion_score",
            "motion_strength",
            "motion_change",
            "motion_delta",
        ],
        0.0
    )


def get_hist_change(shot):

    return first_value(
        shot,
        [
            "hist_change",
            "histogram_change",
            "hist_distance",
            "visual_change",
        ],
        0.0
    )


def get_brightness_change(shot):

    return first_value(
        shot,
        [
            "brightness_change",
            "brightness_delta",
        ],
        0.0
    )


def get_cut_strength(shot):

    return first_value(
        shot,
        [
            "cut_strength",
            "cut",
            "cut_score",
        ],
        0.0
    )


# ============================================================
# VISUAL ENERGY
# ============================================================

def calculate_energy(shot):

    duration = safe_float(
        shot.get("duration", 0.0)
    )

    cut = clamp(
        get_cut_strength(shot)
    )

    motion = clamp(
        get_motion(shot)
    )

    hist = clamp(
        get_hist_change(shot)
    )

    brightness = clamp(
        abs(get_brightness_change(shot))
    )

    events = event_count(shot)

    # Event density
    event_signal = clamp(
        events / 4.0
    )

    # Fast-shot signal.
    # Short duration is editorial intensity in itself.
    fast_signal = 1.0 - normalize(
        duration,
        0.35,
        2.5
    )

    # Final visual/editorial energy.
    #
    # Important:
    # duration is deliberately included because
    # rapid cutting is itself part of the reference style.

    energy = (
        0.23 * cut
        + 0.18 * motion
        + 0.16 * hist
        + 0.10 * brightness
        + 0.18 * event_signal
        + 0.15 * fast_signal
    )

    return round(
        clamp(energy),
        4
    )


def energy_label(energy):

    if energy < 0.25:
        return "low"

    if energy < 0.45:
        return "moderate"

    if energy < 0.65:
        return "high"

    return "extreme"


# ============================================================
# MUSIC ALIGNMENT
# ============================================================

def get_beat_distance(shot):

    return abs(
        first_value(
            shot,
            [
                "beat_distance",
                "nearest_beat_distance",
                "distance_to_beat",
            ],
            999.0
        )
    )


def get_onset_distance(shot):

    return abs(
        first_value(
            shot,
            [
                "onset_distance",
                "nearest_onset_distance",
                "distance_to_onset",
            ],
            999.0
        )
    )


def music_alignment(shot):

    beat = get_beat_distance(shot)
    onset = get_onset_distance(shot)

    if beat <= 0.06:
        return "beat_locked"

    if onset <= 0.06:
        return "onset_locked"

    if beat <= 0.12:
        return "near_beat"

    if onset <= 0.12:
        return "near_onset"

    return "free_timing"


# ============================================================
# EDITORIAL ROLE
# ============================================================

def editorial_role(
    duration,
    energy,
    events,
    alignment
):

    # Long visual holds
    if duration >= LONG:
        if energy < 0.50:
            return "cinematic_hold"
        return "epic_hold"

    # Very dense visual moments
    if duration <= MICRO:

        if energy >= 0.45:
            return "micro_impact"

        return "micro_transition"

    if duration <= VERY_SHORT:

        if energy >= 0.45:
            return "impact_cut"

        return "fast_cut"

    if duration <= SHORT:

        if energy >= 0.40:
            if alignment in {
                "beat_locked",
                "onset_locked",
                "near_beat",
                "near_onset"
            }:
                return "music_accent"

            return "action_cut"

        return "transition"

    # Event-rich medium shots
    if events >= 2 and energy >= 0.40:
        return "effect_window"

    # Longer medium/slow shots
    if energy <= 0.25:
        return "breathing_space"

    if energy >= 0.55:
        return "high_intensity"

    return "transition"


# ============================================================
# LOCAL CUT DENSITY
# ============================================================

def calculate_density(shots, index):

    start = max(
        0,
        index - WINDOW
    )

    end = min(
        len(shots),
        index + WINDOW + 1
    )

    region = shots[start:end]

    durations = [
        safe_float(
            x.get("duration", 0.0)
        )
        for x in region
    ]

    durations = [
        x for x in durations
        if x > 0
    ]

    if len(durations) <= 1:
        return 0.0

    avg = mean(durations)

    # Smaller shots -> greater editorial density
    density = 1.0 - normalize(
        avg,
        0.35,
        2.5
    )

    return round(
        clamp(density),
        4
    )


# ============================================================
# PHASE CLASSIFICATION
# ============================================================

def phase_type(
    average_energy,
    density,
    slope
):

    # Strong acceleration
    if slope > 0.12 and density > 0.40:
        return "buildup"

    # Strong release
    if slope < -0.12:
        return "release"

    # Dense and energetic
    if average_energy >= 0.50 and density >= 0.40:
        return "peak"

    # Sparse / relaxed
    if average_energy < 0.35 and density < 0.30:
        return "breathing"

    # Moderate stable section
    return "steady"


# ============================================================
# TRANSITIONS
# ============================================================

def build_transitions(shots):

    transitions = []

    for i in range(
        len(shots) - 1
    ):

        a = shots[i]
        b = shots[i + 1]

        ea = safe_float(
            a.get("visual_energy", 0)
        )

        eb = safe_float(
            b.get("visual_energy", 0)
        )

        da = safe_float(
            a.get("duration", 0)
        )

        db = safe_float(
            b.get("duration", 0)
        )

        energy_delta = eb - ea

        if energy_delta > 0.15:
            transition = "escalation"

        elif energy_delta < -0.15:
            transition = "release"

        else:
            transition = "continuation"

        transitions.append(
            {
                "from_shot": i + 1,
                "to_shot": i + 2,
                "from_duration": round(da, 4),
                "to_duration": round(db, 4),
                "energy_delta": round(
                    energy_delta,
                    4
                ),
                "transition_type": transition,
            }
        )

    return transitions


# ============================================================
# PHASES
# ============================================================

def build_phases(shots):

    if not shots:
        return []

    # Use groups of roughly 6–10 shots rather than
    # assuming a fixed number of phases.

    phase_size = max(
        6,
        int(round(len(shots) / 5))
    )

    phases = []

    phase_id = 1

    for start in range(
        0,
        len(shots),
        phase_size
    ):

        end = min(
            len(shots),
            start + phase_size
        )

        region = shots[start:end]

        energies = np.array(
            [
                safe_float(
                    x.get(
                        "visual_energy",
                        0
                    )
                )
                for x in region
            ],
            dtype=float
        )

        densities = np.array(
            [
                safe_float(
                    x.get(
                        "local_cut_density",
                        0
                    )
                )
                for x in region
            ],
            dtype=float
        )

        avg_energy = (
            float(np.mean(energies))
            if len(energies)
            else 0.0
        )

        avg_density = (
            float(np.mean(densities))
            if len(densities)
            else 0.0
        )

        if len(energies) >= 2:
            slope = float(
                energies[-1]
                - energies[0]
            )
        else:
            slope = 0.0

        start_time = safe_float(
            region[0].get(
                "start",
                0
            )
        )

        end_time = safe_float(
            region[-1].get(
                "end",
                start_time
                + safe_float(
                    region[-1].get(
                        "duration",
                        0
                    )
                )
            )
        )

        phases.append(
            {
                "phase": phase_id,
                "start": round(
                    start_time,
                    4
                ),
                "end": round(
                    end_time,
                    4
                ),
                "duration": round(
                    end_time - start_time,
                    4
                ),
                "type": phase_type(
                    avg_energy,
                    avg_density,
                    slope,
                ),
                "average_energy": round(
                    avg_energy,
                    4
                ),
                "average_cut_density": round(
                    avg_density,
                    4
                ),
                "energy_slope": round(
                    slope,
                    4
                ),
                "shot_count": len(
                    region
                ),
            }
        )

        phase_id += 1

    return phases


# ============================================================
# STYLE FINGERPRINT
# ============================================================

def build_fingerprint(
    shots,
    transitions,
    phases,
):

    durations = [
        safe_float(
            x.get("duration", 0)
        )
        for x in shots
    ]

    energies = [
        safe_float(
            x.get("visual_energy", 0)
        )
        for x in shots
    ]

    roles = [
        x.get(
            "editorial_role",
            "unknown"
        )
        for x in shots
    ]

    alignments = [
        x.get(
            "music_alignment",
            "free_timing"
        )
        for x in shots
    ]

    pacing = [
        pacing_class(x)
        for x in durations
    ]

    role_counts = {}

    for role in roles:
        role_counts[role] = (
            role_counts.get(role, 0)
            + 1
        )

    def role_ratio(name):

        return round(
            role_counts.get(
                name,
                0
            )
            / len(roles),
            4
        ) if roles else 0.0

    return {
        "shot_count": len(shots),

        "mean_shot_duration": round(
            mean(durations),
            4
        ) if durations else 0.0,

        "median_shot_duration": round(
            median(durations),
            4
        ) if durations else 0.0,

        "shortest_shot": round(
            min(durations),
            4
        ) if durations else 0.0,

        "longest_shot": round(
            max(durations),
            4
        ) if durations else 0.0,

        "mean_visual_energy": round(
            mean(energies),
            4
        ) if energies else 0.0,

        "pacing_distribution": {
            "extreme": round(
                pacing.count("extreme")
                / len(pacing),
                4
            ) if pacing else 0,

            "fast": round(
                pacing.count("fast")
                / len(pacing),
                4
            ) if pacing else 0,

            "medium": round(
                pacing.count("medium")
                / len(pacing),
                4
            ) if pacing else 0,

            "slow": round(
                pacing.count("slow")
                / len(pacing),
                4
            ) if pacing else 0,
        },

        "rhythm_pattern": (
            "rapid_cutting"
            if (
                sum(
                    x in {
                        "extreme",
                        "fast"
                    }
                    for x in pacing
                )
                / len(pacing)
                >= 0.55
            )
            else "mixed_rhythm"
        ) if pacing else "unknown",

        "music_alignment": {
            "beat_locked_ratio": round(
                alignments.count(
                    "beat_locked"
                )
                / len(alignments),
                4
            ) if alignments else 0,

            "onset_locked_ratio": round(
                alignments.count(
                    "onset_locked"
                )
                / len(alignments),
                4
            ) if alignments else 0,

            "near_beat_ratio": round(
                alignments.count(
                    "near_beat"
                )
                / len(alignments),
                4
            ) if alignments else 0,
        },

        "editorial_roles": {
            key: role_ratio(key)
            for key in [
                "cinematic_hold",
                "epic_hold",
                "breathing_space",
                "micro_impact",
                "impact_cut",
                "fast_cut",
                "action_cut",
                "music_accent",
                "effect_window",
                "high_intensity",
                "transition",
            ]
        },

        "transition_behavior": {
            "escalation_ratio": round(
                sum(
                    x["transition_type"]
                    == "escalation"
                    for x in transitions
                )
                / len(transitions),
                4
            ) if transitions else 0,

            "release_ratio": round(
                sum(
                    x["transition_type"]
                    == "release"
                    for x in transitions
                )
                / len(transitions),
                4
            ) if transitions else 0,
        },

        "phase_count": len(
            phases
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 82)
    print("🎬 REFERENCE EDITING GRAMMAR BUILDER")
    print("=" * 82)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing:\n{INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    shots = data.get(
        "shots",
        []
    )

    if not shots:
        shots = data.get(
            "editorial_shots",
            []
        )

    if not shots:
        raise ValueError(
            "No shots found in reference shot map."
        )

    processed = []

    for index, original in enumerate(shots):

        shot = dict(original)

        start = safe_float(
            shot.get(
                "start",
                shot.get(
                    "start_time",
                    0
                )
            )
        )

        end = safe_float(
            shot.get(
                "end",
                shot.get(
                    "end_time",
                    start
                    + safe_float(
                        shot.get(
                            "duration",
                            0
                        )
                    )
                )
            )
        )

        duration = safe_float(
            shot.get(
                "duration",
                end - start
            )
        )

        energy = calculate_energy(
            shot
        )

        events = event_count(
            shot
        )

        alignment = music_alignment(
            shot
        )

        density = calculate_density(
            shots,
            index
        )

        role = editorial_role(
            duration,
            energy,
            events,
            alignment,
        )

        shot.update(
            {
                "shot_number": index + 1,
                "start": round(
                    start,
                    4
                ),
                "end": round(
                    end,
                    4
                ),
                "duration": round(
                    duration,
                    4
                ),
                "duration_class":
                    duration_class(
                        duration
                    ),
                "pacing":
                    pacing_class(
                        duration
                    ),
                "event_count":
                    events,
                "visual_energy":
                    energy,
                "energy_label":
                    energy_label(
                        energy
                    ),
                "music_alignment":
                    alignment,
                "local_cut_density":
                    density,
                "editorial_role":
                    role,
            }
        )

        processed.append(
            shot
        )

    transitions = build_transitions(
        processed
    )

    phases = build_phases(
        processed
    )

    fingerprint = build_fingerprint(
        processed,
        transitions,
        phases,
    )

    # --------------------------------------------------------
    # AUDIO SUMMARY
    # --------------------------------------------------------

    audio = data.get(
        "audio",
        {}
    )

    bpm = first_value(
        audio,
        ["bpm", "tempo"],
        first_value(
            data,
            ["bpm", "tempo"],
            0
        )
    )

    beat_times = audio.get(
        "beat_times",
        audio.get(
            "beats",
            data.get(
                "beat_times",
                []
            )
        )
    )

    strong_onsets = audio.get(
        "strong_onsets",
        audio.get(
            "strong_onset_times",
            data.get(
                "strong_onsets",
                []
            )
        )
    )

    output = {

        "meta": {
            "name":
                "reference_edit_grammar",

            "version":
                "1.1",

            "purpose":
                "Represent the editing behavior of the reference "
                "without copying its footage."
        },

        "reference": {
            "duration":
                safe_float(
                    data.get(
                        "duration",
                        0
                    )
                ),

            "fps":
                safe_float(
                    data.get(
                        "fps",
                        0
                    )
                ),
        },

        "music": {
            "bpm":
                round(
                    bpm,
                    4
                ),

            "beat_count":
                len(
                    beat_times
                )
                if isinstance(
                    beat_times,
                    (list, tuple)
                )
                else 0,

            "strong_onset_count":
                len(
                    strong_onsets
                )
                if isinstance(
                    strong_onsets,
                    (list, tuple)
                )
                else 0,
        },

        "style_fingerprint":
            fingerprint,

        "editorial_phases":
            phases,

        "shots":
            processed,

        "transitions":
            transitions,

        "director_rules": {

            "reference_controls_rhythm":
                True,

            "reference_does_not_control_subject":
                True,

            "do_not_cut_on_every_onset":
                True,

            "use_short_shots_for_peaks":
                True,

            "use_long_shots_for_contrast":
                True,

            "music_sync_is_weighted_not_absolute":
                True,

            "visual_moment_can_override_beat":
                True,

            "effect_density_should_follow_energy":
                True,

            "avoid_constant_effects":
                True,

            "preserve_breathing_spaces":
                True,

            "allow_original_creative_decisions":
                True,
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 82)
    print("✅ REFERENCE EDITING GRAMMAR COMPLETE")
    print("=" * 82)

    print(
        f"Editorial shots       : "
        f"{len(processed)}"
    )

    print(
        f"Transitions           : "
        f"{len(transitions)}"
    )

    print(
        f"Editorial phases      : "
        f"{len(phases)}"
    )

    print(
        f"BPM                   : "
        f"{bpm:.2f}"
    )

    print(
        f"Mean shot duration    : "
        f"{fingerprint['mean_shot_duration']:.3f}s"
    )

    print(
        f"Median shot duration  : "
        f"{fingerprint['median_shot_duration']:.3f}s"
    )

    print(
        f"Rhythm pattern        : "
        f"{fingerprint['rhythm_pattern']}"
    )

    print()
    print("PACING DISTRIBUTION")

    for key, value in fingerprint[
        "pacing_distribution"
    ].items():

        print(
            f"  {key:<10}: "
            f"{value * 100:5.1f}%"
        )

    print()
    print("EDITORIAL ROLES")

    for key, value in fingerprint[
        "editorial_roles"
    ].items():

        if value > 0:
            print(
                f"  {key:<20}: "
                f"{value * 100:5.1f}%"
            )

    print()
    print("PHASES")

    for phase in phases:

        print(
            f"  Phase {phase['phase']:02d} "
            f"{phase['start']:.2f}→"
            f"{phase['end']:.2f}s  "
            f"{phase['type']:<12} "
            f"energy={phase['average_energy']:.2f}  "
            f"density={phase['average_cut_density']:.2f}"
        )

    print()
    print("Saved:")
    print(OUTPUT_PATH)
    print("=" * 82)


if __name__ == "__main__":
    main()