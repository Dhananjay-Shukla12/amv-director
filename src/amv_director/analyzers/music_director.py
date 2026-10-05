import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

GRAMMAR_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "reference_edit_grammar.json"
)

DEFAULT_AUDIO = (
    BASE_DIR
    / "data"
    / "outputs"
    / "reference_audio.wav"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "music_director_plan.json"
)


# ============================================================
# SETTINGS
# ============================================================

HOP_LENGTH = 512

STRONG_ONSET_PERCENTILE = 82

MIN_EVENT_DISTANCE = 0.12

MIN_SHOT_DURATION = 0.20
MAX_SHOT_DURATION = 5.50

SNAP_WINDOW = 0.10

# We do not want every onset to become a cut.
MAX_MUSIC_SYNCS = 24

# Minimum time between music-driven snaps.
MUSIC_SYNC_COOLDOWN = 0.65

LOW_ENERGY = 0.33
HIGH_ENERGY = 0.66

# Reference pacing proportions
REFERENCE_DISTRIBUTION = {
    "extreme": 0.25,
    "fast": 0.409,
    "medium": 0.136,
    "slow": 0.204,
}


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        if isinstance(
            value,
            (list, tuple, np.ndarray)
        ):

            if len(value) == 0:
                return default

            value = value[0]

        return float(value)

    except Exception:

        return default


def clamp(
    value,
    low=0.0,
    high=1.0
):

    return max(
        low,
        min(high, value)
    )


def nearest_value(
    value,
    candidates
):

    if len(candidates) == 0:
        return None

    candidates = np.asarray(
        candidates,
        dtype=float
    )

    index = int(
        np.argmin(
            np.abs(
                candidates - value
            )
        )
    )

    return float(
        candidates[index]
    )


# ============================================================
# AUDIO ANALYSIS
# ============================================================

def analyze_audio(audio_path):

    print()
    print("=" * 82)
    print("🎵 MUSIC DIRECTOR")
    print("=" * 82)

    print()
    print("Loading:")
    print(audio_path)

    y, sr = librosa.load(
        str(audio_path),
        sr=None,
        mono=True
    )

    duration = float(
        librosa.get_duration(
            y=y,
            sr=sr
        )
    )

    print(
        f"Duration              : "
        f"{duration:.3f}s"
    )

    # --------------------------------------------------------
    # ONSET ENVELOPE
    # --------------------------------------------------------

    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH
    )

    onset_times = librosa.frames_to_time(
        np.arange(len(onset_env)),
        sr=sr,
        hop_length=HOP_LENGTH
    )

    # --------------------------------------------------------
    # BEATS
    # --------------------------------------------------------

    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        onset_envelope=onset_env,
        hop_length=HOP_LENGTH
    )

    tempo_arr = np.asarray(
        tempo
    ).reshape(-1)

    bpm = (
        float(tempo_arr[0])
        if len(tempo_arr)
        else 0.0
    )

    beat_frames = np.asarray(
        beat_frames
    ).reshape(-1)

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=HOP_LENGTH
    )

    # --------------------------------------------------------
    # STRONG ONSETS
    # --------------------------------------------------------

    threshold = np.percentile(
        onset_env,
        STRONG_ONSET_PERCENTILE
    )

    min_distance_frames = max(
        1,
        int(
            MIN_EVENT_DISTANCE
            * sr
            / HOP_LENGTH
        )
    )

    peak_frames, properties = find_peaks(
        onset_env,
        height=threshold,
        distance=min_distance_frames
    )

    strong_onset_times = onset_times[
        peak_frames
    ]

    strong_onset_strengths = (
        properties.get(
            "peak_heights",
            np.array([])
        )
    )

    # --------------------------------------------------------
    # RMS ENERGY
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=HOP_LENGTH
    )[0]

    rms_times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=HOP_LENGTH
    )

    # Robust normalization
    low_rms = np.percentile(
        rms,
        10
    )

    high_rms = np.percentile(
        rms,
        95
    )

    if high_rms <= low_rms:
        normalized_rms = np.zeros_like(
            rms
        )

    else:
        normalized_rms = np.clip(
            (
                rms - low_rms
            )
            / (
                high_rms - low_rms
            ),
            0.0,
            1.0
        )

    return {
        "duration": duration,
        "sr": sr,
        "bpm": bpm,
        "y": y,
        "onset_env": onset_env,
        "onset_times": onset_times,
        "beat_times": beat_times,
        "strong_onset_times":
            strong_onset_times,
        "strong_onset_strengths":
            strong_onset_strengths,
        "rms_times": rms_times,
        "normalized_rms":
            normalized_rms,
    }


# ============================================================
# ENERGY AT TIME
# ============================================================

def energy_at_time(
    time,
    rms_times,
    normalized_rms
):

    if len(rms_times) == 0:
        return 0.0

    index = int(
        np.searchsorted(
            rms_times,
            time
        )
    )

    index = max(
        0,
        min(
            index,
            len(normalized_rms) - 1
        )
    )

    return float(
        normalized_rms[index]
    )


# ============================================================
# LOCAL ENERGY TREND
# ============================================================

def energy_trend(
    time,
    rms_times,
    normalized_rms
):

    current = energy_at_time(
        time,
        rms_times,
        normalized_rms
    )

    before = energy_at_time(
        max(0.0, time - 1.0),
        rms_times,
        normalized_rms
    )

    after = energy_at_time(
        time + 1.0,
        rms_times,
        normalized_rms
    )

    return clamp(
        (after - before + 1.0) / 2.0
    )


# ============================================================
# EDITORIAL INTENSITY
# ============================================================

def editorial_intensity(
    energy,
    trend,
    density
):

    # Energy is the primary driver.
    #
    # Density tells us how aggressively the editor
    # should respond to the music.
    #
    # Trend tells us whether intensity is rising or falling.

    value = (
        0.55 * energy
        + 0.25 * density
        + 0.20 * trend
    )

    return clamp(value)


def intensity_label(value):

    if value < 0.25:
        return "low"

    if value < 0.45:
        return "moderate"

    if value < 0.70:
        return "high"

    return "extreme"


# ============================================================
# REFERENCE SHOT TARGET
# ============================================================

def duration_target(
    intensity,
    reference_durations
):

    if not reference_durations:

        if intensity >= 0.75:
            return 0.45

        if intensity >= 0.50:
            return 0.75

        if intensity >= 0.30:
            return 1.25

        return 2.50

    durations = np.asarray(
        reference_durations,
        dtype=float
    )

    durations = durations[
        durations > 0
    ]

    if len(durations) == 0:
        return 1.0

    # Reference quantiles
    q20 = float(
        np.percentile(
            durations,
            20
        )
    )

    q40 = float(
        np.percentile(
            durations,
            40
        )
    )

    q65 = float(
        np.percentile(
            durations,
            65
        )
    )

    q85 = float(
        np.percentile(
            durations,
            85
        )
    )

    if intensity >= 0.75:

        return clamp(
            q20,
            MIN_SHOT_DURATION,
            MAX_SHOT_DURATION
        )

    if intensity >= 0.55:

        return clamp(
            q40,
            MIN_SHOT_DURATION,
            MAX_SHOT_DURATION
        )

    if intensity >= 0.35:

        return clamp(
            q65,
            MIN_SHOT_DURATION,
            MAX_SHOT_DURATION
        )

    return clamp(
        q85,
        MIN_SHOT_DURATION,
        MAX_SHOT_DURATION
    )


# ============================================================
# SYNC POINTS
# ============================================================

def build_sync_points(
    beat_times,
    strong_onset_times,
    onset_env,
    onset_times
):

    points = []

    # Beat points
    for t in beat_times:

        points.append(
            {
                "time": float(t),
                "type": "beat",
                "priority": 0.55,
            }
        )

    # Strong onset points
    for index, t in enumerate(
        strong_onset_times
    ):

        strength = (
            safe_float(
                onset_env[
                    int(
                        np.argmin(
                            np.abs(
                                onset_times
                                - t
                            )
                        )
                    )
                ]
            )
        )

        points.append(
            {
                "time": float(t),
                "type": "strong_onset",
                "priority":
                    0.75
                    + min(
                        strength,
                        1.0
                    ) * 0.25,
            }
        )

    # Sort
    points.sort(
        key=lambda x: x["time"]
    )

    # Merge nearby candidates
    merged = []

    for point in points:

        if not merged:

            merged.append(
                point
            )
            continue

        previous = merged[-1]

        if (
            abs(
                point["time"]
                - previous["time"]
            )
            < 0.06
        ):

            # Keep stronger event
            if (
                point["priority"]
                > previous["priority"]
            ):

                merged[-1] = point

        else:

            merged.append(
                point
            )

    return merged


# ============================================================
# FIND BEST SYNC
# ============================================================

# def best_sync(
#     target,
#     sync_points
# ):

#     if not sync_points:

#         return None

#     candidate_times = [
#         x["time"]
#         for x in sync_points
#     ]

#     closest = nearest_value(
#         target,
#         candidate_times
#     )

#     if closest is None:
#         return None

#     distance = abs(
#         closest - target
#     )

#     if distance > SNAP_WINDOW:
#         return None

#     for point in sync_points:

#         if abs(
#             point["time"]
#             - closest
#         ) < 1e-6:

#             return {
#                 "time": closest,
#                 "type": point["type"],
#                 "distance": distance,
#                 "priority": point["priority"],
#             }

#     return None

def best_sync(
    target,
    sync_points,
    last_sync_time=-999.0,
    sync_count=0,
):
    """
    Find a useful musical synchronization point.

    Important:
    Music events are opportunities, not mandatory cuts.
    """

    if not sync_points:
        return None

    if sync_count >= MAX_MUSIC_SYNCS:
        return None

    candidates = []

    for point in sync_points:

        t = float(point["time"])

        distance = abs(
            t - target
        )

        if distance > SNAP_WINDOW:
            continue

        # Prevent repeated music-driven cuts too close together.
        if (
            t - last_sync_time
            < MUSIC_SYNC_COOLDOWN
        ):
            continue

        # Prefer stronger events and closer timing.
        score = (
            0.65 * float(
                point.get(
                    "priority",
                    0.0
                )
            )
            +
            0.35 * (
                1.0
                - min(
                    distance
                    / SNAP_WINDOW,
                    1.0
                )
            )
        )

        candidates.append(
            (
                score,
                point,
                distance,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    score, point, distance = (
        candidates[0]
    )

    return {
        "time": float(
            point["time"]
        ),
        "type": point["type"],
        "distance": float(
            distance
        ),
        "priority": float(
            point.get(
                "priority",
                0.0
            )
        ),
        "score": float(score),
    }
# ============================================================
# BUILD EDITORIAL PLAN
# ============================================================

def build_plan(
    duration,
    beat_times,
    strong_onset_times,
    onset_env,
    onset_times,
    rms_times,
    normalized_rms,
    grammar
):

    reference_durations = []

    for shot in grammar.get(
        "shots",
        []
    ):

        value = safe_float(
            shot.get(
                "duration",
                0
            )
        )

        if value > 0:
            reference_durations.append(
                value
            )

    reference_density = (
        grammar
        .get(
            "style_fingerprint",
            {}
        )
        .get(
            "pacing_distribution",
            {}
        )
    )

    sync_points = build_sync_points(
        beat_times,
        strong_onset_times,
        onset_env,
        onset_times
    )

    segments = []

    cursor = 0.0

    safety_counter = 0

    last_sync_time = -999.0
    sync_count = 0

    while (
        cursor < duration - 0.05
        and safety_counter < 1000
    ):

        safety_counter += 1

        energy = energy_at_time(
            cursor + 0.01,
            rms_times,
            normalized_rms
        )

        trend = energy_trend(
            cursor,
            rms_times,
            normalized_rms
        )

        # Reference density curve.
        #
        # This is deliberately blended rather than
        # copied exactly.

        ref_extreme = safe_float(
            reference_density.get(
                "extreme",
                0.25
            )
        )

        ref_fast = safe_float(
            reference_density.get(
                "fast",
                0.40
            )
        )

        density = clamp(
            (
                0.70 * energy
                + 0.20 * trend
                + 0.10 *
                (
                    ref_extreme
                    + ref_fast
                )
            )
        )

        intensity = editorial_intensity(
            energy,
            trend,
            density
        )

        label = intensity_label(
            intensity
        )

        target = duration_target(
            intensity,
            reference_durations
        )
        # Prevent the music director from producing
        # dramatically more cuts than the reference.

        if intensity < 0.35:
            target *= 1.20

        elif intensity < 0.55:
            target *= 1.10

        target = clamp(
            target,
            MIN_SHOT_DURATION,
            MAX_SHOT_DURATION
        )

        proposed_end = min(
            duration,
            cursor + target
        )

        # ----------------------------------------------------
        # Musical snapping
        # ----------------------------------------------------

        sync = best_sync(
            proposed_end,
            sync_points,
            last_sync_time,
            sync_count,
        )

        sync_type = "free"

        snap_distance = 0.0

        if sync is not None:

            snapped_end = sync["time"]

            proposed_duration = (
                snapped_end - cursor
            )

            # Don't allow pathological tiny segments.
            if (
                proposed_duration
                >= MIN_SHOT_DURATION
            ):

                proposed_end = snapped_end

                sync_type = sync["type"]

                snap_distance = (
                    sync["distance"]
                )
                last_sync_time = snapped_end
                sync_count += 1

        final_duration = (
            proposed_end - cursor
        )

        if final_duration < MIN_SHOT_DURATION:

            proposed_end = min(
                duration,
                cursor
                + MIN_SHOT_DURATION
            )

            final_duration = (
                proposed_end - cursor
            )

            sync_type = "free"

        # ----------------------------------------------------
        # Editorial role
        # ----------------------------------------------------

        if intensity >= 0.75:

            role = "impact_zone"

        elif intensity >= 0.55:

            if sync_type in {
                "strong_onset",
                "beat"
            }:

                role = "accent_zone"

            else:

                role = "action_zone"

        elif intensity >= 0.35:

            role = "build_zone"

        else:

            role = "cinematic_zone"

        # ----------------------------------------------------
        # Sync priority
        # ----------------------------------------------------

        if sync_type == "strong_onset":

            sync_priority = "high"

        elif sync_type == "beat":

            sync_priority = "medium"

        else:

            sync_priority = "low"

        segments.append(
            {
                "segment": len(
                    segments
                ) + 1,

                "start": round(
                    cursor,
                    4
                ),

                "end": round(
                    proposed_end,
                    4
                ),

                "duration": round(
                    final_duration,
                    4
                ),

                "music_energy": round(
                    energy,
                    4
                ),

                "energy_trend": round(
                    trend,
                    4
                ),

                "editorial_intensity":
                    round(
                        intensity,
                        4
                    ),

                "intensity_label":
                    label,

                "editorial_role":
                    role,

                "sync_type":
                    sync_type,

                "sync_priority":
                    sync_priority,

                "snap_distance":
                    round(
                        snap_distance,
                        4
                    ),

                "target_duration": round(
                    target,
                    4
                ),
            }
        )

        cursor = proposed_end

    return segments


# ============================================================
# POST PROCESS
# ============================================================

def add_context(
    segments,
    duration
):

    total = len(
        segments
    )

    for index, segment in enumerate(
        segments
    ):

        previous = (
            segments[index - 1]
            if index > 0
            else None
        )

        next_segment = (
            segments[index + 1]
            if index < total - 1
            else None
        )

        # ----------------------------------------------------
        # Change intensity
        # ----------------------------------------------------

        current = safe_float(
            segment[
                "editorial_intensity"
            ]
        )

        previous_value = (
            safe_float(
                previous[
                    "editorial_intensity"
                ]
            )
            if previous
            else current
        )

        next_value = (
            safe_float(
                next_segment[
                    "editorial_intensity"
                ]
            )
            if next_segment
            else current
        )

        delta_before = (
            current
            - previous_value
        )

        delta_after = (
            next_value
            - current
        )

        if delta_before > 0.20:

            transition_context = (
                "entering_peak"
            )

        elif delta_before < -0.20:

            transition_context = (
                "releasing"
            )

        elif delta_after > 0.20:

            transition_context = (
                "building_toward_peak"
            )

        elif delta_after < -0.20:

            transition_context = (
                "preparing_release"
            )

        else:

            transition_context = (
                "stable"
            )

        segment[
            "transition_context"
        ] = transition_context

        # ----------------------------------------------------
        # Effect budget
        # ----------------------------------------------------

        if current >= 0.75:

            effect_budget = "maximum"

        elif current >= 0.55:

            effect_budget = "high"

        elif current >= 0.35:

            effect_budget = "moderate"

        else:

            effect_budget = "minimal"

        segment[
            "effect_budget"
        ] = effect_budget

        # ----------------------------------------------------
        # Creative freedom
        # ----------------------------------------------------

        if current >= 0.75:

            creativity = "focused"

        elif current >= 0.55:

            creativity = "moderate"

        else:

            creativity = "high"

        segment[
            "creative_freedom"
        ] = creativity

    return segments


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    segments,
    audio
):

    if not segments:

        return {}

    durations = np.array(
        [
            safe_float(
                s["duration"]
            )
            for s in segments
        ]
    )

    intensities = np.array(
        [
            safe_float(
                s["editorial_intensity"]
            )
            for s in segments
        ]
    )

    sync_counts = {}

    for segment in segments:

        value = segment[
            "sync_type"
        ]

        sync_counts[value] = (
            sync_counts.get(
                value,
                0
            )
            + 1
        )

    role_counts = {}

    for segment in segments:

        value = segment[
            "editorial_role"
        ]

        role_counts[value] = (
            role_counts.get(
                value,
                0
            )
            + 1
        )

    return {

        "segment_count":
            len(segments),

        "mean_segment_duration":
            round(
                float(
                    np.mean(
                        durations
                    )
                ),
                4
            ),

        "median_segment_duration":
            round(
                float(
                    np.median(
                        durations
                    )
                ),
                4
            ),

        "shortest_segment":
            round(
                float(
                    np.min(
                        durations
                    )
                ),
                4
            ),

        "longest_segment":
            round(
                float(
                    np.max(
                        durations
                    )
                ),
                4
            ),

        "mean_intensity":
            round(
                float(
                    np.mean(
                        intensities
                    )
                ),
                4
            ),

        "sync_counts":
            sync_counts,

        "role_counts":
            role_counts,

        "music": {
            "bpm":
                round(
                    audio["bpm"],
                    4
                ),

            "beat_count":
                len(
                    audio["beat_times"]
                ),

            "strong_onset_count":
                len(
                    audio[
                        "strong_onset_times"
                    ]
                ),
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build a music-aware editorial timing "
            "plan using the reference editing grammar."
        )
    )

    parser.add_argument(
        "audio",
        nargs="?",
        default=str(
            DEFAULT_AUDIO
        ),
        help=(
            "Path to music/audio file. "
            "Defaults to reference_audio.wav"
        ),
    )

    args = parser.parse_args()

    audio_path = Path(
        args.audio
    ).expanduser().resolve()

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    if not GRAMMAR_PATH.exists():

        raise FileNotFoundError(
            "Reference grammar not found:\n"
            f"{GRAMMAR_PATH}"
        )

    # --------------------------------------------------------
    # Load grammar
    # --------------------------------------------------------

    with open(
        GRAMMAR_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        grammar = json.load(f)

    # --------------------------------------------------------
    # Analyze music
    # --------------------------------------------------------

    audio = analyze_audio(
        audio_path
    )

    # --------------------------------------------------------
    # Build plan
    # --------------------------------------------------------

    segments = build_plan(
        duration=audio["duration"],
        beat_times=audio["beat_times"],
        strong_onset_times=
            audio["strong_onset_times"],
        onset_env=
            audio["onset_env"],
        onset_times=
            audio["onset_times"],
        rms_times=
            audio["rms_times"],
        normalized_rms=
            audio["normalized_rms"],
        grammar=grammar,
    )

    segments = add_context(
        segments,
        audio["duration"]
    )

    summary = build_summary(
        segments,
        audio
    )

    output = {

        "meta": {
            "name":
                "music_director_plan",

            "version":
                "1.0",

            "purpose":
                (
                    "Create a music-synchronized editorial "
                    "timeline before source footage selection."
                ),

            "audio_source":
                str(audio_path),
        },

        "music": {
            "duration":
                round(
                    audio["duration"],
                    4
                ),

            "sample_rate":
                audio["sr"],

            "bpm":
                round(
                    audio["bpm"],
                    4
                ),

            "beats":
                [
                    round(
                        float(x),
                        4
                    )
                    for x in audio[
                        "beat_times"
                    ]
                ],

            "strong_onsets":
                [
                    round(
                        float(x),
                        4
                    )
                    for x in audio[
                        "strong_onset_times"
                    ]
                ],
        },

        "reference_style": {
            "mean_shot_duration":
                grammar[
                    "style_fingerprint"
                ].get(
                    "mean_shot_duration",
                    0
                ),

            "median_shot_duration":
                grammar[
                    "style_fingerprint"
                ].get(
                    "median_shot_duration",
                    0
                ),

            "pacing_distribution":
                grammar[
                    "style_fingerprint"
                ].get(
                    "pacing_distribution",
                    {}
                ),

            "rhythm_pattern":
                grammar[
                    "style_fingerprint"
                ].get(
                    "rhythm_pattern",
                    "unknown"
                ),
        },

        "summary":
            summary,

        "segments":
            segments,

        "director_principles": [

            (
                "Music determines timing opportunities; "
                "it does not dictate every cut."
            ),

            (
                "Strong onsets receive higher sync priority "
                "than ordinary beats."
            ),

            (
                "High-energy music should generally produce "
                "shorter editorial windows."
            ),

            (
                "Low-energy sections should create breathing "
                "space rather than constant cutting."
            ),

            (
                "Reference pacing is used as a style constraint, "
                "not copied frame-for-frame."
            ),

            (
                "The Creative Director still decides which "
                "source footage best serves each segment."
            ),

            (
                "Effect intensity is bounded by the music's "
                "editorial intensity."
            ),
        ],
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 82)
    print("✅ MUSIC DIRECTOR PLAN COMPLETE")
    print("=" * 82)

    print(
        f"Audio duration        : "
        f"{audio['duration']:.3f}s"
    )

    print(
        f"BPM                   : "
        f"{audio['bpm']:.2f}"
    )

    print(
        f"Beats                 : "
        f"{len(audio['beat_times'])}"
    )

    print(
        f"Strong onsets         : "
        f"{len(audio['strong_onset_times'])}"
    )

    print(
        f"Editorial segments    : "
        f"{len(segments)}"
    )

    print(
        f"Mean segment duration : "
        f"{summary['mean_segment_duration']:.3f}s"
    )

    print(
        f"Median segment        : "
        f"{summary['median_segment_duration']:.3f}s"
    )

    print()
    print("SYNC TYPES")

    for key, value in summary[
        "sync_counts"
    ].items():

        print(
            f"  {key:<15}: "
            f"{value}"
        )

    print()
    print("EDITORIAL ROLES")

    for key, value in summary[
        "role_counts"
    ].items():

        print(
            f"  {key:<20}: "
            f"{value}"
        )

    print()
    print("FIRST 20 SEGMENTS")

    for segment in segments[:20]:

        print(
            f"  #{segment['segment']:02d} "
            f"{segment['start']:.3f}→"
            f"{segment['end']:.3f}s "
            f"dur={segment['duration']:.3f}s "
            f"energy={segment['music_energy']:.2f} "
            f"role={segment['editorial_role']} "
            f"sync={segment['sync_type']}"
        )

    print()
    print("Saved:")
    print(OUTPUT_PATH)
    print("=" * 82)


if __name__ == "__main__":
    main()