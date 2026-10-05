import json
from pathlib import Path
from statistics import mean


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

CREATIVE_PLAN_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "creative_director_plan.json"
)

MUSIC_PLAN_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "music_director_plan.json"
)

REFERENCE_MAP_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "reference_shot_map.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "effect_director_plan.json"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_EFFECTS_PER_SEGMENT = 4

MIN_DURATION_FOR_HEAVY_EFFECTS = 0.22

FLASH_DURATION = 0.045

HEAVY_INTENSITY = 0.72

MEDIUM_INTENSITY = 0.48


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        if isinstance(
            value,
            (list, tuple)
        ):

            if not value:
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


def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# REFERENCE EFFECT SIGNAL
# ============================================================

def get_reference_effect_signal(
    reference_map,
    target_time
):
    """
    Find the strongest reference visual event near
    the same relative timestamp.

    We don't copy the reference footage.
    We only borrow its effect/editing behavior.
    """

    events = reference_map.get(
        "events",
        []
    )

    if not events:

        # Some versions may store events under shots.
        for shot in reference_map.get(
            "shots",
            []
        ):

            for event in shot.get(
                "events",
                []
            ):

                if isinstance(
                    event,
                    dict
                ):

                    events.append(
                        event
                    )

    if not events:

        return {
            "type": "none",
            "strength": 0.0,
            "distance": 999.0,
        }

    best = None

    for event in events:

        event_time = safe_float(
            event.get(
                "time",
                event.get(
                    "timestamp",
                    event.get(
                        "start",
                        0
                    )
                )
            )
        )

        distance = abs(
            event_time
            - target_time
        )

        if best is None:

            best = (
                distance,
                event
            )

        elif distance < best[0]:

            best = (
                distance,
                event
            )

    if best is None:

        return {
            "type": "none",
            "strength": 0.0,
            "distance": 999.0,
        }

    distance, event = best

    event_type = str(
        event.get(
            "type",
            "visual_change"
        )
    )

    strength = safe_float(
        event.get(
            "score",
            event.get(
                "strength",
                event.get(
                    "event_strength",
                    0.5
                )
            )
        ),
        0.5
    )

    return {
        "type":
            event_type,

        "strength":
            clamp(
                strength
            ),

        "distance":
            distance,
    }


# ============================================================
# EFFECT INTENSITY
# ============================================================

def determine_effect_intensity(
    segment,
    reference_signal
):

    music_intensity = safe_float(
        segment.get(
            "editorial_intensity",
            0.0
        )
    )

    music_energy = safe_float(
        segment.get(
            "music_energy",
            0.0
        )
    )

    ref_strength = safe_float(
        reference_signal.get(
            "strength",
            0.0
        )
    )

    intensity = (
        0.50 * music_intensity
        + 0.30 * music_energy
        + 0.20 * ref_strength
    )

    return clamp(
        intensity
    )


# ============================================================
# REFERENCE EVENT → EFFECT FAMILY
# ============================================================

def event_to_effect_family(
    event_type
):

    event_type = str(
        event_type
    ).lower()

    if (
        "flash" in event_type
        or "impact" in event_type
    ):

        return "impact"

    if (
        "motion" in event_type
        or "structural" in event_type
    ):

        return "movement"

    if (
        "brightness" in event_type
        or "visual" in event_type
    ):

        return "transition"

    return "none"


# ============================================================
# EFFECT PROFILES
# ============================================================

def base_profile(
    role,
    intensity
):

    intensity = clamp(
        intensity
    )

    if role == "cinematic_zone":

        return {
            "zoom": 0.28,
            "pan": 0.08,
            "shake": 0.0,
            "blur": 0.0,
            "flash": 0.0,
            "contrast": 0.18,
            "vignette": 0.12,
            "speed_ramp": 0.0,
            "grain": 0.04,
        }

    if role == "build_zone":

        return {
            "zoom": 0.42,
            "pan": 0.18,
            "shake": 0.08,
            "blur": 0.05,
            "flash": 0.0,
            "contrast": 0.25,
            "vignette": 0.10,
            "speed_ramp": 0.08,
            "grain": 0.05,
        }

    if role == "accent_zone":

        return {
            "zoom": 0.60,
            "pan": 0.25,
            "shake": 0.30,
            "blur": 0.10,
            "flash": 0.12,
            "contrast": 0.35,
            "vignette": 0.12,
            "speed_ramp": 0.15,
            "grain": 0.06,
        }

    if role == "action_zone":

        return {
            "zoom": 0.72,
            "pan": 0.32,
            "shake": 0.45,
            "blur": 0.14,
            "flash": 0.18,
            "contrast": 0.42,
            "vignette": 0.14,
            "speed_ramp": 0.22,
            "grain": 0.07,
        }

    if role == "impact_zone":

        return {
            "zoom": 0.90,
            "pan": 0.38,
            "shake": 0.72,
            "blur": 0.24,
            "flash": 0.65,
            "contrast": 0.55,
            "vignette": 0.18,
            "speed_ramp": 0.30,
            "grain": 0.08,
        }

    return {
        "zoom": 0.30,
        "pan": 0.10,
        "shake": 0.0,
        "blur": 0.0,
        "flash": 0.0,
        "contrast": 0.20,
        "vignette": 0.10,
        "speed_ramp": 0.0,
        "grain": 0.04,
    }


# ============================================================
# SCALE PROFILE
# ============================================================

def scale_profile(
    profile,
    intensity
):

    # Never blindly scale all effects linearly.
    # Some effects are intentionally nonlinear.

    output = {}

    for key, value in profile.items():

        value = safe_float(
            value
        )

        if key == "shake":

            scaled = (
                value
                * (
                    0.65
                    + 0.35
                    * intensity
                )
            )

        elif key == "flash":

            scaled = (
                value
                * (
                    intensity
                    ** 1.7
                )
            )

        elif key == "blur":

            scaled = (
                value
                * (
                    intensity
                    ** 1.3
                )
            )

        else:

            scaled = (
                value
                * (
                    0.70
                    + 0.30
                    * intensity
                )
            )

        output[key] = round(
            clamp(
                scaled
            ),
            4
        )

    return output


# ============================================================
# EFFECT CHOREOGRAPHY
# ============================================================

def build_choreography(
    segment,
    profile,
    reference_signal
):

    role = segment.get(
        "editorial_role",
        "build_zone"
    )

    intensity = safe_float(
        segment.get(
            "editorial_intensity",
            0.5
        )
    )

    duration = safe_float(
        segment.get(
            "duration",
            0.5
        )
    )

    sync_type = segment.get(
        "sync_type",
        "free"
    )

    transition_context = segment.get(
        "transition_context",
        "stable"
    )

    reference_family = (
        event_to_effect_family(
            reference_signal.get(
                "type",
                "none"
            )
        )
    )

    effects = []

    # ========================================================
    # CINEMATIC
    # ========================================================

    if role == "cinematic_zone":

        effects.append(
            {
                "type": "slow_zoom",
                "amount":
                    profile["zoom"],
                "direction":
                    "in",
            }
        )

        if profile["pan"] > 0.06:

            effects.append(
                {
                    "type": "subtle_pan",
                    "amount":
                        profile["pan"],
                    "direction":
                        "auto",
                }
            )

        return effects

    # ========================================================
    # BUILD
    # ========================================================

    if role == "build_zone":

        effects.append(
            {
                "type": "progressive_zoom",
                "amount":
                    profile["zoom"],
                "start":
                    0.0,
                "end":
                    1.0,
            }
        )

        if intensity >= MEDIUM_INTENSITY:

            effects.append(
                {
                    "type": "directional_motion",
                    "amount":
                        profile["pan"],
                    "direction":
                        "auto",
                }
            )

        if (
            transition_context
            == "building_toward_peak"
        ):

            effects.append(
                {
                    "type": "speed_ramp",
                    "amount":
                        profile[
                            "speed_ramp"
                        ],
                    "direction":
                        "accelerate",
                }
            )

        return effects[
            :MAX_EFFECTS_PER_SEGMENT
        ]

    # ========================================================
    # ACCENT
    # ========================================================

    if role == "accent_zone":

        effects.append(
            {
                "type": "push_zoom",
                "amount":
                    profile["zoom"],
                "anchor":
                    "center",
            }
        )

        if sync_type != "free":

            effects.append(
                {
                    "type": "micro_shake",
                    "amount":
                        profile["shake"],
                    "duration":
                        min(
                            0.12,
                            duration
                        ),
                }
            )

        if (
            reference_family
            == "movement"
        ):

            effects.append(
                {
                    "type": "motion_blur",
                    "amount":
                        profile["blur"],
                }
            )

        return effects[
            :MAX_EFFECTS_PER_SEGMENT
        ]

    # ========================================================
    # ACTION
    # ========================================================

    if role == "action_zone":

        effects.append(
            {
                "type": "dynamic_zoom",
                "amount":
                    profile["zoom"],
                "direction":
                    "impact_direction",
            }
        )

        effects.append(
            {
                "type": "directional_shake",
                "amount":
                    profile["shake"],
                "duration":
                    min(
                        0.16,
                        duration
                    ),
            }
        )

        effects.append(
            {
                "type": "motion_blur",
                "amount":
                    profile["blur"],
            }
        )

        return effects[
            :MAX_EFFECTS_PER_SEGMENT
        ]

    # ========================================================
    # IMPACT
    # ========================================================

    if role == "impact_zone":

        # ----------------------------------------------------
        # BEFORE IMPACT
        # ----------------------------------------------------

        if (
            duration
            >= MIN_DURATION_FOR_HEAVY_EFFECTS
        ):

            effects.append(
                {
                    "type":
                        "impact_pre_zoom",

                    "amount":
                        min(
                            1.0,
                            profile[
                                "zoom"
                            ]
                        ),

                    "timing":
                        "pre_impact",
                }
            )

        # ----------------------------------------------------
        # IMPACT
        # ----------------------------------------------------

        effects.append(
            {
                "type":
                    "impact_shake",

                "amount":
                    profile[
                        "shake"
                    ],

                "timing":
                    "impact",

                "duration":
                    min(
                        0.14,
                        duration
                    ),
            }
        )

        # ----------------------------------------------------
        # FLASH
        # ----------------------------------------------------

        if (
            reference_family
            == "impact"
            or sync_type
            == "strong_onset"
        ):

            effects.append(
                {
                    "type":
                        "impact_flash",

                    "amount":
                        profile[
                            "flash"
                        ],

                    "duration":
                        min(
                            FLASH_DURATION,
                            duration
                        ),
                }
            )

        # ----------------------------------------------------
        # POST IMPACT
        # ----------------------------------------------------

        effects.append(
            {
                "type":
                    "post_impact_blur",

                "amount":
                    profile[
                        "blur"
                    ],

                "timing":
                    "post_impact",
            }
        )

        return effects[
            :MAX_EFFECTS_PER_SEGMENT
        ]

    return effects


# ============================================================
# EFFECT BUDGET
# ============================================================

def effect_budget(
    intensity,
    role
):

    if role == "cinematic_zone":

        return "minimal"

    if intensity >= 0.80:

        return "maximum"

    if intensity >= 0.60:

        return "high"

    if intensity >= 0.35:

        return "moderate"

    return "minimal"


# ============================================================
# BUILD EFFECT PLAN
# ============================================================

def build_plan(
    creative_plan,
    music_plan,
    reference_map
):

    segments = creative_plan.get(
        "segments",
        []
    )

    if not segments:

        raise ValueError(
            "No creative director segments found."
        )

    output = []

    for segment in segments:

        start = safe_float(
            segment.get(
                "start",
                0
            )
        )

        end = safe_float(
            segment.get(
                "end",
                start
            )
        )

        duration = safe_float(
            segment.get(
                "duration",
                end - start
            )
        )

        midpoint = (
            start
            + duration / 2.0
        )

        reference_signal = (
            get_reference_effect_signal(
                reference_map,
                midpoint
            )
        )

        intensity = (
            determine_effect_intensity(
                segment,
                reference_signal
            )
        )

        role = segment.get(
            "editorial_role",
            "build_zone"
        )

        profile = base_profile(
            role,
            intensity
        )

        profile = scale_profile(
            profile,
            intensity
        )

        choreography = build_choreography(
            segment,
            profile,
            reference_signal
        )

        budget = effect_budget(
            intensity,
            role
        )

        # ----------------------------------------------------
        # Global finishing parameters
        # ----------------------------------------------------

        finishing = {
            "contrast":
                profile[
                    "contrast"
                ],

            "vignette":
                profile[
                    "vignette"
                ],

            "grain":
                profile[
                    "grain"
                ],
        }

        # Don't add visible grain aggressively
        # during cinematic sections.
        if role == "cinematic_zone":

            finishing[
                "grain"
            ] *= 0.5

        output.append(
            {
                "segment":
                    segment[
                        "segment"
                    ],

                "start":
                    start,

                "end":
                    end,

                "duration":
                    duration,

                "editorial_role":
                    role,

                "intensity":
                    round(
                        intensity,
                        4
                    ),

                "effect_budget":
                    budget,

                "sync_type":
                    segment.get(
                        "sync_type",
                        "free"
                    ),

                "reference_effect_signal":
                    {
                        "type":
                            reference_signal[
                                "type"
                            ],

                        "strength":
                            round(
                                reference_signal[
                                    "strength"
                                ],
                                4
                            ),

                        "distance":
                            round(
                                reference_signal[
                                    "distance"
                                ],
                                4
                            ),
                    },

                "effect_profile":
                    profile,

                "choreography":
                    choreography,

                "finishing":
                    {
                        key:
                            round(
                                safe_float(value),
                                4
                            )
                        for key, value
                        in finishing.items()
                    },

                "creative_principle":
                    (
                        "Effects reinforce the editorial "
                        "moment; they do not compensate for "
                        "weak source footage."
                    ),
            }
        )

    return output


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    segments
):

    if not segments:

        return {}

    intensities = [
        safe_float(
            x.get(
                "intensity",
                0
            )
        )
        for x in segments
    ]

    choreography_counts = {}

    for segment in segments:

        for effect in segment.get(
            "choreography",
            []
        ):

            effect_type = effect.get(
                "type",
                "unknown"
            )

            choreography_counts[
                effect_type
            ] = (
                choreography_counts.get(
                    effect_type,
                    0
                )
                + 1
            )

    budgets = {}

    for segment in segments:

        budget = segment.get(
            "effect_budget",
            "unknown"
        )

        budgets[budget] = (
            budgets.get(
                budget,
                0
            )
            + 1
        )

    return {

        "segment_count":
            len(segments),

        "mean_effect_intensity":
            round(
                mean(
                    intensities
                ),
                4
            )
            if intensities
            else 0.0,

        "effect_budget_distribution":
            budgets,

        "effect_type_counts":
            choreography_counts,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 82)
    print("🎬 EFFECT DIRECTOR")
    print("=" * 82)

    creative_plan = load_json(
        CREATIVE_PLAN_PATH
    )

    music_plan = load_json(
        MUSIC_PLAN_PATH
    )

    reference_map = load_json(
        REFERENCE_MAP_PATH
    )

    print()
    print(
        f"Creative segments      : "
        f"{len(creative_plan.get('segments', []))}"
    )

    print(
        f"Reference events       : "
        f"{len(reference_map.get('events', []))}"
    )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    segments = build_plan(
        creative_plan,
        music_plan,
        reference_map
    )

    summary = build_summary(
        segments
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {

        "meta": {

            "name":
                "effect_director_plan",

            "version":
                "1.0",

            "purpose":
                (
                    "Translate musical/editorial intent "
                    "and reference effect behavior into "
                    "deterministic effect choreography."
                ),
        },

        "summary":
            summary,

        "principles": [

            (
                "Effects are motivated by editorial intent."
            ),

            (
                "The reference contributes effect grammar, "
                "not copied footage."
            ),

            (
                "High intensity permits stronger effects."
            ),

            (
                "Cinematic sections stay restrained."
            ),

            (
                "Impact moments may combine several effects "
                "in a coordinated sequence."
            ),

            (
                "Effects must never compensate for poor "
                "source footage."
            ),

            (
                "Creative variation is controlled rather "
                "than random."
            ),
        ],

        "segments":
            segments,
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

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 82)
    print("✅ EFFECT DIRECTOR PLAN COMPLETE")
    print("=" * 82)

    print(
        f"Segments              : "
        f"{summary['segment_count']}"
    )

    print(
        f"Mean effect intensity : "
        f"{summary['mean_effect_intensity']:.3f}"
    )

    print()
    print("EFFECT BUDGET")

    for key, value in summary[
        "effect_budget_distribution"
    ].items():

        print(
            f"  {key:<10}: "
            f"{value}"
        )

    print()
    print("EFFECT TYPES")

    for key, value in sorted(
        summary[
            "effect_type_counts"
        ].items(),
        key=lambda x:
            x[1],
        reverse=True
    ):

        print(
            f"  {key:<25}: "
            f"{value}"
        )

    print()
    print("FIRST 15 EFFECT DECISIONS")

    for segment in segments[:15]:

        effects = [
            x.get(
                "type",
                "unknown"
            )
            for x in segment.get(
                "choreography",
                []
            )
        ]

        print(
            f"  #{segment['segment']:02d} "
            f"{segment['start']:.3f}→"
            f"{segment['end']:.3f}s "
            f"{segment['editorial_role']:<15} "
            f"intensity="
            f"{segment['intensity']:.2f} "
            f"→ "
            f"{', '.join(effects) if effects else 'none'}"
        )

    print()
    print("Saved:")
    print(OUTPUT_PATH)
    print("=" * 82)


if __name__ == "__main__":
    main()