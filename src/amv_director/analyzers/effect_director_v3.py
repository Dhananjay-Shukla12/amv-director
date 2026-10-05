import json
from pathlib import Path

import numpy as np


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

REFERENCE_MOTION_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "reference_motion_grammar.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "effect_director_plan_v3.json"
)


# ============================================================
# SETTINGS
# ============================================================

EPSILON = 1e-8

MAX_ACTIONS = 6

MIN_ACTION_DURATION = 0.035

STRONG_MUSIC = 0.70

MEDIUM_MUSIC = 0.45

SOURCE_MOTION_THRESHOLD = 0.40

SOURCE_HIGH_MOTION = 0.65

REFERENCE_STRONG_MOTION = 0.65

REFERENCE_STRONG_FLASH = 0.72

REFERENCE_STRONG_SHAKE = 0.60


# ============================================================
# HELPERS
# ============================================================

def safe_float(
    value,
    default=0.0
):

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
# REFERENCE SHOT MATCHING
# ============================================================

def find_reference_shot(
    reference_shots,
    target_time
):

    if not reference_shots:

        return None

    # First try a containing shot.
    containing = []

    for shot in reference_shots:

        start = safe_float(
            shot.get(
                "start",
                0
            )
        )

        end = safe_float(
            shot.get(
                "end",
                start
            )
        )

        if (
            start <= target_time
            <= end
        ):

            containing.append(
                shot
            )

    if containing:

        # Prefer the shortest containing shot,
        # because it usually represents the more
        # precise local editing behavior.
        return min(
            containing,
            key=lambda x:
                safe_float(
                    x.get(
                        "duration",
                        999
                    )
                )
        )

    # Otherwise use nearest midpoint.
    best = None
    best_distance = 999999.0

    for shot in reference_shots:

        start = safe_float(
            shot.get(
                "start",
                0
            )
        )

        end = safe_float(
            shot.get(
                "end",
                start
            )
        )

        midpoint = (
            start + end
        ) / 2.0

        distance = abs(
            midpoint
            - target_time
        )

        if distance < best_distance:

            best_distance = distance
            best = shot

    return best


# ============================================================
# SOURCE PROFILE
# ============================================================

def get_source_profile(
    segment
):

    clips = segment.get(
        "clips",
        []
    )

    if not clips:

        return {
            "motion": 0.0,
            "brightness": 0.5,
            "contrast": 0.5,
            "sharpness": 0.5,
            "edge_density": 0.5,
        }

    motion = []
    brightness = []
    contrast = []
    sharpness = []
    edges = []

    for clip in clips:

        features = clip.get(
            "normalized_features",
            {}
        )

        motion.append(
            safe_float(
                features.get(
                    "motion",
                    0
                )
            )
        )

        brightness.append(
            safe_float(
                features.get(
                    "brightness",
                    0.5
                )
            )
        )

        contrast.append(
            safe_float(
                features.get(
                    "contrast",
                    0.5
                )
            )
        )

        sharpness.append(
            safe_float(
                features.get(
                    "sharpness",
                    0.5
                )
            )
        )

        edges.append(
            safe_float(
                features.get(
                    "edge_density",
                    0.5
                )
            )
        )

    return {

        "motion":
            float(
                np.mean(
                    motion
                )
            ),

        "brightness":
            float(
                np.mean(
                    brightness
                )
            ),

        "contrast":
            float(
                np.mean(
                    contrast
                )
            ),

        "sharpness":
            float(
                np.mean(
                    sharpness
                )
            ),

        "edge_density":
            float(
                np.mean(
                    edges
                )
            ),
    }


# ============================================================
# REFERENCE PROFILE
# ============================================================

def get_reference_profile(
    shot
):

    if shot is None:

        return {
            "pattern":
                "static_hold",

            "motion":
                0.0,

            "peak_motion":
                0.0,

            "shake":
                0.0,

            "flash":
                0.0,

            "direction":
                "none",
        }

    profile = shot.get(
        "motion_profile",
        {}
    )

    return {

        "pattern":
            profile.get(
                "motion_pattern",
                "static_hold"
            ),

        "motion":
            safe_float(
                profile.get(
                    "motion_score",
                    0
                )
            ),

        "peak_motion":
            safe_float(
                profile.get(
                    "peak_motion_score",
                    0
                )
            ),

        "shake":
            safe_float(
                profile.get(
                    "shake_score",
                    0
                )
            ),

        "flash":
            safe_float(
                profile.get(
                    "flash_score",
                    0
                )
            ),

        "direction":
            profile.get(
                "dominant_direction",
                "none"
            ),
    }


# ============================================================
# MUSIC PROFILE
# ============================================================

def get_music_profile(
    segment
):

    return {

        "energy":
            clamp(
                safe_float(
                    segment.get(
                        "music_energy",
                        0
                    )
                )
            ),

        "intensity":
            clamp(
                safe_float(
                    segment.get(
                        "editorial_intensity",
                        0
                    )
                )
            ),

        "sync":
            segment.get(
                "sync_type",
                "free"
            ),

        "role":
            segment.get(
                "editorial_role",
                "build_zone"
            ),

        "context":
            segment.get(
                "transition_context",
                "stable"
            ),
    }


# ============================================================
# DIRECTION
# ============================================================

def choose_direction(
    reference,
    source
):

    ref_direction = reference[
        "direction"
    ]

    if ref_direction != "none":

        return ref_direction

    # If reference has no explicit direction,
    # use the source only when the source itself
    # has meaningful motion.

    source_motion = source[
        "motion"
    ]

    if source_motion < SOURCE_MOTION_THRESHOLD:

        return "none"

    # There is no source direction in the current
    # Creative Director schema, so don't invent one.
    return "auto"


# ============================================================
# EFFECT STRENGTH
# ============================================================

def calculate_strength(
    reference,
    music,
    source
):

    reference_motion = reference[
        "motion"
    ]

    music_intensity = music[
        "intensity"
    ]

    source_motion = source[
        "motion"
    ]

    # Reference remains the primary visual influence.
    base = (
        0.50 * reference_motion
        + 0.30 * music_intensity
        + 0.20 * source_motion
    )

    return clamp(
        base
    )


# ============================================================
# ACTION FACTORY
# ============================================================

def make_action(
    effect,
    start,
    end,
    **kwargs
):

    return {
        "effect":
            effect,

        "start":
            round(
                max(
                    0.0,
                    start
                ),
                4
            ),

        "end":
            round(
                max(
                    start,
                    end
                ),
                4
            ),

        **kwargs,
    }


def append_action(
    actions,
    item,
    duration
):

    if len(actions) >= MAX_ACTIONS:

        return

    start = safe_float(
        item.get(
            "start",
            0
        )
    )

    end = safe_float(
        item.get(
            "end",
            start
        )
    )

    item["start"] = max(
        0.0,
        start
    )

    item["end"] = min(
        duration,
        end
    )

    if (
        item["effect"]
        in {
            "flash",
            "impact_snap"
        }
    ):

        if (
            item["end"]
            > item["start"]
        ):

            actions.append(
                item
            )

        return

    if (
        item["end"]
        - item["start"]
        >= MIN_ACTION_DURATION
    ):

        actions.append(
            item
        )


# ============================================================
# STATIC HOLD
# ============================================================

def build_static_hold(
    duration,
    strength,
    source
):

    actions = []

    # IMPORTANT:
    # No automatic zoom.
    #
    # A still reference shot should stay still.
    # We only introduce a very small drift if the
    # source itself has enough movement.

    if (
        source["motion"]
        >= SOURCE_MOTION_THRESHOLD
        and strength >= 0.25
    ):

        append_action(
            actions,
            make_action(
                "camera_drift",
                0.0,
                duration,
                amount=
                    0.01
                    + 0.02
                    * strength,
                direction=
                    "auto",
                easing=
                    "ease_in_out",
            ),
            duration
        )

    return actions


# ============================================================
# SUBTLE DRIFT
# ============================================================

def build_subtle_drift(
    duration,
    reference,
    strength
):

    actions = []

    direction = reference[
        "direction"
    ]

    if direction == "none":

        direction = "auto"

    append_action(
        actions,
        make_action(
            "camera_drift",
            0.0,
            duration,
            amount=
                0.015
                + 0.035
                * strength,
            direction=
                direction,
            easing=
                "ease_in_out",
        ),
        duration
    )

    return actions


# ============================================================
# DIRECTIONAL MOTION
# ============================================================

def build_directional_motion(
    duration,
    reference,
    strength
):

    actions = []

    direction = reference[
        "direction"
    ]

    if direction == "none":

        direction = "auto"

    amount = (
        0.025
        + 0.065
        * strength
    )

    append_action(
        actions,
        make_action(
            "directional_motion",
            0.0,
            duration,
            direction=
                direction,
            amount=
                amount,
            easing=
                "ease_in_out",
        ),
        duration
    )

    # Only add a very small secondary zoom when
    # music is sufficiently strong.
    if strength >= 0.65:

        append_action(
            actions,
            make_action(
                "micro_push",
                duration * 0.25,
                duration * 0.78,
                amount=
                    0.025
                    * strength,
                easing=
                    "ease_in",
            ),
            duration
        )

    return actions


# ============================================================
# ZOOM IN
# ============================================================

def build_zoom_in(
    duration,
    strength
):

    actions = []

    zoom_amount = (
        0.035
        + 0.075
        * strength
    )

    append_action(
        actions,
        make_action(
            "zoom_curve",
            0.0,
            duration,
            start_scale=1.0,
            end_scale=
                1.0
                + zoom_amount,
            easing=
                "ease_in_out",
        ),
        duration
    )

    return actions


# ============================================================
# ZOOM OUT
# ============================================================

def build_zoom_out(
    duration,
    strength
):

    actions = []

    start_scale = (
        1.0
        + 0.035
        * strength
    )

    append_action(
        actions,
        make_action(
            "zoom_curve",
            0.0,
            duration,
            start_scale=
                start_scale,
            end_scale=1.0,
            easing=
                "ease_out",
        ),
        duration
    )

    return actions


# ============================================================
# IMPACT BURST
# ============================================================

def build_impact_burst(
    duration,
    strength,
    music,
    source
):

    actions = []

    # Very short clips need compressed choreography.

    pre_end = duration * 0.45

    snap_start = duration * 0.42
    snap_end = duration * 0.60

    recovery_start = (
        duration * 0.58
    )

    # --------------------------------------------------------
    # Pre-impact push
    # --------------------------------------------------------

    append_action(
        actions,
        make_action(
            "pre_push",
            0.0,
            pre_end,
            amount=
                0.035
                + 0.055
                * strength,
            easing=
                "ease_in",
        ),
        duration
    )

    # --------------------------------------------------------
    # Snap
    # --------------------------------------------------------

    append_action(
        actions,
        make_action(
            "impact_snap",
            snap_start,
            snap_end,
            strength=
                0.45
                + 0.45
                * strength,
        ),
        duration
    )

    # --------------------------------------------------------
    # Blur
    #
    # Only if there is actual motion.
    # --------------------------------------------------------

    if (
        source["motion"]
        >= SOURCE_MOTION_THRESHOLD
    ):

        append_action(
            actions,
            make_action(
                "motion_blur",
                snap_start,
                min(
                    duration,
                    snap_end
                    + 0.12
                ),
                amount=
                    0.05
                    + 0.10
                    * strength,
            ),
            duration
        )

    # --------------------------------------------------------
    # Recovery
    # --------------------------------------------------------

    append_action(
        actions,
        make_action(
            "settle",
            recovery_start,
            duration,
            strength=
                0.20
                + 0.35
                * strength,
        ),
        duration
    )

    # --------------------------------------------------------
    # Shake is NOT automatic.
    # It must be supported by the reference.
    # --------------------------------------------------------

    if (
        strength >= 0.75
        and music["energy"] >= STRONG_MUSIC
    ):

        append_action(
            actions,
            make_action(
                "shake_burst",
                snap_start,
                min(
                    duration,
                    snap_end
                    + 0.10
                ),
                amplitude=
                    0.05
                    + 0.20
                    * strength,
                frequency=
                    18,
            ),
            duration
        )

    return actions


# ============================================================
# FLASH IMPACT
# ============================================================

def build_flash_impact(
    duration,
    strength,
    reference,
    music,
    source
):

    actions = []

    snap_start = (
        duration * 0.44
    )

    snap_end = (
        duration * 0.58
    )

    # --------------------------------------------------------
    # Small pre-approach
    # --------------------------------------------------------

    append_action(
        actions,
        make_action(
            "pre_push",
            0.0,
            duration * 0.40,
            amount=
                0.025
                + 0.045
                * strength,
            easing=
                "ease_in",
        ),
        duration
    )

    # --------------------------------------------------------
    # Snap
    # --------------------------------------------------------

    append_action(
        actions,
        make_action(
            "impact_snap",
            snap_start,
            snap_end,
            strength=
                0.50
                + 0.40
                * strength,
        ),
        duration
    )

    # --------------------------------------------------------
    # Flash
    #
    # Reference flash strength must justify it.
    # --------------------------------------------------------

    if (
        reference["flash"]
        >= REFERENCE_STRONG_FLASH
        and music["energy"]
        >= MEDIUM_MUSIC
    ):

        flash_center = (
            duration * 0.50
        )

        flash_width = max(
            0.025,
            min(
                0.075,
                duration * 0.16
            )
        )

        append_action(
            actions,
            make_action(
                "flash",
                flash_center
                - flash_width / 2,
                flash_center
                + flash_width / 2,
                strength=
                    0.30
                    + 0.50
                    * strength,
            ),
            duration
        )

    # --------------------------------------------------------
    # Blur only when source supports movement
    # --------------------------------------------------------

    if (
        source["motion"]
        >= SOURCE_MOTION_THRESHOLD
    ):

        append_action(
            actions,
            make_action(
                "motion_blur",
                snap_start,
                min(
                    duration,
                    snap_end
                    + 0.10
                ),
                amount=
                    0.04
                    + 0.08
                    * strength,
            ),
            duration
        )

    # --------------------------------------------------------
    # Shake only if reference actually indicates it.
    # --------------------------------------------------------

    if (
        reference["shake"]
        >= REFERENCE_STRONG_SHAKE
        and music["energy"]
        >= STRONG_MUSIC
    ):

        append_action(
            actions,
            make_action(
                "shake_burst",
                snap_start,
                min(
                    duration,
                    snap_end
                    + 0.10
                ),
                amplitude=
                    0.08
                    + 0.18
                    * reference["shake"],
                frequency=
                    16,
            ),
            duration
        )

    # --------------------------------------------------------
    # Settle
    # --------------------------------------------------------

    append_action(
        actions,
        make_action(
            "settle",
            duration * 0.62,
            duration,
            strength=
                0.20
                + 0.30
                * strength,
        ),
        duration
    )

    return actions


# ============================================================
# MIXED MOTION
# ============================================================

def build_mixed_motion(
    duration,
    reference,
    music,
    source,
    strength
):

    actions = []

    direction = choose_direction(
        reference,
        source
    )

    # --------------------------------------------------------
    # Directional camera movement
    # --------------------------------------------------------

    if direction != "none":

        append_action(
            actions,
            make_action(
                "directional_motion",
                0.0,
                duration * 0.72,
                direction=
                    direction,
                amount=
                    0.02
                    + 0.05
                    * strength,
                easing=
                    "ease_in_out",
            ),
            duration
        )

    # --------------------------------------------------------
    # Small local push if music is building.
    # --------------------------------------------------------

    if (
        music["intensity"]
        >= MEDIUM_MUSIC
    ):

        append_action(
            actions,
            make_action(
                "micro_push",
                duration * 0.30,
                duration * 0.82,
                amount=
                    0.02
                    + 0.035
                    * strength,
                easing=
                    "ease_in",
            ),
            duration
        )

    # --------------------------------------------------------
    # Motion blur only when source is already moving.
    # --------------------------------------------------------

    if (
        source["motion"]
        >= SOURCE_HIGH_MOTION
        and strength
        >= 0.55
    ):

        append_action(
            actions,
            make_action(
                "motion_blur",
                duration * 0.42,
                duration * 0.68,
                amount=
                    0.025
                    + 0.06
                    * strength,
            ),
            duration
        )

    return actions


# ============================================================
# BUILD SEGMENT
# ============================================================

def build_segment(
    segment,
    reference_shot
):

    duration = max(
        0.08,
        safe_float(
            segment.get(
                "duration",
                0.5
            )
        )
    )

    reference = get_reference_profile(
        reference_shot
    )

    music = get_music_profile(
        segment
    )

    source = get_source_profile(
        segment
    )

    strength = calculate_strength(
        reference,
        music,
        source
    )

    pattern = reference[
        "pattern"
    ]

    # ========================================================
    # PATTERN DISPATCH
    # ========================================================

    if pattern == "static_hold":

        actions = build_static_hold(
            duration,
            strength,
            source
        )

    elif pattern == "subtle_drift":

        actions = build_subtle_drift(
            duration,
            reference,
            strength
        )

    elif pattern == "directional_motion":

        actions = build_directional_motion(
            duration,
            reference,
            strength
        )

    elif pattern == "zoom_in":

        actions = build_zoom_in(
            duration,
            strength
        )

    elif pattern == "zoom_out":

        actions = build_zoom_out(
            duration,
            strength
        )

    elif pattern == "impact_burst":

        actions = build_impact_burst(
            duration,
            strength,
            music,
            source
        )

    elif pattern == "flash_impact":

        actions = build_flash_impact(
            duration,
            strength,
            reference,
            music,
            source
        )

    else:

        actions = build_mixed_motion(
            duration,
            reference,
            music,
            source,
            strength
        )

    # ========================================================
    # FALLBACK
    # ========================================================

    if not actions:

        # A completely clean shot is preferable
        # to forcing an effect.
        actions = []

    # ========================================================
    # EFFECT BUDGET
    # ========================================================

    if not actions:

        budget = "none"

    elif len(actions) == 1:

        budget = "minimal"

    elif len(actions) <= 3:

        budget = "moderate"

    else:

        budget = "high"

    # ========================================================
    # FINISHING
    # ========================================================

    # Keep finishing subtle.
    finishing = {

        "contrast":
            round(
                1.02
                + 0.10
                * music["intensity"],
                4
            ),

        "saturation":
            round(
                1.00
                + 0.05
                * music["intensity"],
                4
            ),

        "vignette":
            round(
                0.01
                + 0.04
                * music["intensity"],
                4
            ),

        "grain":
            round(
                0.005
                + 0.015
                * music["intensity"],
                4
            ),
    }

    return {

        "segment":
            segment[
                "segment"
            ],

        "start":
            safe_float(
                segment.get(
                    "start",
                    0
                )
            ),

        "end":
            safe_float(
                segment.get(
                    "end",
                    duration
                )
            ),

        "duration":
            duration,

        "editorial_role":
            music["role"],

        "music_energy":
            music["energy"],

        "editorial_intensity":
            music["intensity"],

        "sync_type":
            music["sync"],

        "reference_source_shot":
            (
                reference_shot.get(
                    "shot_number"
                )
                if reference_shot
                else None
            ),

        "reference_motion":

            {
                "pattern":
                    reference[
                        "pattern"
                    ],

                "motion":
                    round(
                        reference[
                            "motion"
                        ],
                        4
                    ),

                "peak_motion":
                    round(
                        reference[
                            "peak_motion"
                        ],
                        4
                    ),

                "shake":
                    round(
                        reference[
                            "shake"
                        ],
                        4
                    ),

                "flash":
                    round(
                        reference[
                            "flash"
                        ],
                        4
                    ),

                "direction":
                    reference[
                        "direction"
                    ],
            },

        "source_motion":
            {
                key:
                    round(
                        value,
                        4
                    )
                for key, value
                in source.items()
            },

        "strength":
            round(
                strength,
                4
            ),

        "effect_budget":
            budget,

        "actions":
            actions,

        "action_count":
            len(actions),

        "finishing":
            finishing,

        "principle":
            (
                "Reference motion determines movement type; "
                "music determines timing/intensity; source "
                "footage determines whether movement is safe."
            ),
    }


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    segments
):

    pattern_counts = {}

    action_counts = {}

    total_actions = 0

    shake_segments = 0

    flash_segments = 0

    clean_segments = 0

    for segment in segments:

        pattern = (
            segment[
                "reference_motion"
            ][
                "pattern"
            ]
        )

        pattern_counts[
            pattern
        ] = (
            pattern_counts.get(
                pattern,
                0
            )
            + 1
        )

        total_actions += (
            segment[
                "action_count"
            ]
        )

        if any(
            action["effect"]
            == "shake_burst"
            for action
            in segment[
                "actions"
            ]
        ):

            shake_segments += 1

        if any(
            action["effect"]
            == "flash"
            for action
            in segment[
                "actions"
            ]
        ):

            flash_segments += 1

        if (
            segment[
                "action_count"
            ]
            == 0
        ):

            clean_segments += 1

        for action in segment[
            "actions"
        ]:

            effect = action[
                "effect"
            ]

            action_counts[
                effect
            ] = (
                action_counts.get(
                    effect,
                    0
                )
                + 1
            )

    return {

        "segment_count":
            len(segments),

        "total_actions":
            total_actions,

        "average_actions_per_segment":
            round(
                total_actions
                /
                max(
                    len(segments),
                    1
                ),
                4
            ),

        "reference_pattern_distribution":
            pattern_counts,

        "effect_distribution":
            action_counts,

        "segments_with_shake":
            shake_segments,

        "segments_with_flash":
            flash_segments,

        "clean_segments":
            clean_segments,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 82)
    print("🎬 EFFECT DIRECTOR V3")
    print("=" * 82)

    creative_plan = load_json(
        CREATIVE_PLAN_PATH
    )

    music_plan = load_json(
        MUSIC_PLAN_PATH
    )

    reference_motion = load_json(
        REFERENCE_MOTION_PATH
    )

    creative_segments = (
        creative_plan.get(
            "segments",
            []
        )
    )

    music_segments = (
        music_plan.get(
            "segments",
            []
        )
    )

    reference_shots = (
        reference_motion.get(
            "shots",
            []
        )
    )

    if not creative_segments:

        raise RuntimeError(
            "No creative segments found."
        )

    if not reference_shots:

        raise RuntimeError(
            "No reference motion shots found."
        )

    print()
    print(
        f"Creative segments : "
        f"{len(creative_segments)}"
    )

    print(
        f"Music segments    : "
        f"{len(music_segments)}"
    )

    print(
        f"Reference shots    : "
        f"{len(reference_shots)}"
    )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    output_segments = []

    for segment in creative_segments:

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

        midpoint = (
            start
            + end
        ) / 2.0

        reference_shot = (
            find_reference_shot(
                reference_shots,
                midpoint
            )
        )

        output_segments.append(
            build_segment(
                segment,
                reference_shot
            )
        )

    summary = build_summary(
        output_segments
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {

        "meta": {

            "name":
                "effect_director_plan_v3",

            "version":
                "3.0",

            "purpose":
                (
                    "Translate learned reference motion "
                    "behavior into source-adaptive, "
                    "time-localized effect choreography."
                ),
        },

        "design_rules": [

            (
                "Do not use a universal impact preset."
            ),

            (
                "Reference motion pattern determines "
                "the movement family."
            ),

            (
                "Reference direction influences camera direction."
            ),

            (
                "Music intensity controls effect strength."
            ),

            (
                "Source motion limits aggressive effects."
            ),

            (
                "Shake is only allowed when reference "
                "and music justify it."
            ),

            (
                "Flash is only allowed when the reference "
                "contains meaningful flash behavior."
            ),

            (
                "Static reference shots should remain mostly clean."
            ),

            (
                "Zoom is not the default effect."
            ),

            (
                "Clean frames are preferable to unnecessary effects."
            ),
        ],

        "summary":
            summary,

        "segments":
            output_segments,
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
    print("✅ EFFECT DIRECTOR V3 COMPLETE")
    print("=" * 82)

    print(
        f"Segments              : "
        f"{summary['segment_count']}"
    )

    print(
        f"Total timed actions   : "
        f"{summary['total_actions']}"
    )

    print(
        f"Average actions       : "
        f"{summary['average_actions_per_segment']:.2f}"
    )

    print(
        f"Segments with shake   : "
        f"{summary['segments_with_shake']}"
    )

    print(
        f"Segments with flash   : "
        f"{summary['segments_with_flash']}"
    )

    print(
        f"Clean segments        : "
        f"{summary['clean_segments']}"
    )

    print()
    print("REFERENCE MOTION USED")

    for pattern, count in sorted(
        summary[
            "reference_pattern_distribution"
        ].items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"  {pattern:<20}: "
            f"{count}"
        )

    print()
    print("EFFECT ACTIONS")

    for effect, count in sorted(
        summary[
            "effect_distribution"
        ].items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"  {effect:<20}: "
            f"{count}"
        )

    print()
    print("FIRST 15 DECISIONS")

    for segment in output_segments[:15]:

        reference = segment[
            "reference_motion"
        ]

        actions = ", ".join(
            x["effect"]
            for x in segment[
                "actions"
            ]
        )

        print(
            f"  #{segment['segment']:02d} "
            f"{segment['start']:.3f}→"
            f"{segment['end']:.3f}s "
            f"ref="
            f"{reference['pattern']:<18} "
            f"dir="
            f"{reference['direction']:<18} "
            f"→ "
            f"{actions if actions else 'CLEAN'}"
        )

    print()
    print("Saved:")
    print(
        OUTPUT_PATH
    )

    print(
        "=" * 82
    )


if __name__ == "__main__":
    main()