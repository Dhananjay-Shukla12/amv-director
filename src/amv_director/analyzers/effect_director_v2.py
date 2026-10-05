import json
from pathlib import Path


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
    / "effect_director_plan_v2.json"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_EVENT_DURATION = 0.12

MAX_ACTIONS_PER_SEGMENT = 7


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, (list, tuple)):
            if not value:
                return default
            value = value[0]

        return float(value)

    except Exception:
        return default


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


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
# REFERENCE EVENT SIGNAL
# ============================================================

def nearest_reference_event(
    reference_map,
    target_time
):
    events = reference_map.get(
        "events",
        []
    )

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
            event_time - target_time
        )

        if (
            best is None
            or distance < best["distance"]
        ):
            best = {
                "time": event_time,
                "distance": distance,
                "type": event.get(
                    "type",
                    "visual_change"
                ),
                "strength": clamp(
                    safe_float(
                        event.get(
                            "score",
                            event.get(
                                "strength",
                                0.5
                            )
                        ),
                        0.5
                    )
                ),
            }

    if best is None:

        return {
            "time": target_time,
            "distance": 999.0,
            "type": "none",
            "strength": 0.0,
        }

    return best


# ============================================================
# INTENSITY
# ============================================================

def segment_intensity(segment):

    editorial = clamp(
        safe_float(
            segment.get(
                "editorial_intensity",
                0.0
            )
        )
    )

    music = clamp(
        safe_float(
            segment.get(
                "music_energy",
                0.0
            )
        )
    )

    return clamp(
        0.60 * editorial
        + 0.40 * music
    )


# ============================================================
# BASE CAMERA STYLE
# ============================================================

def base_camera(segment, intensity):

    role = segment.get(
        "editorial_role",
        "build_zone"
    )

    if role == "cinematic_zone":

        return {
            "zoom_start": 1.00,
            "zoom_end":
                1.035 + 0.02 * intensity,

            "pan_x": 0.0,
            "pan_y": 0.0,

            "shake": 0.0,
            "blur": 0.0,

            "contrast":
                1.04 + 0.10 * intensity,

            "saturation":
                1.00 + 0.04 * intensity,
        }

    if role == "build_zone":

        return {
            "zoom_start": 1.00,
            "zoom_end":
                1.06 + 0.04 * intensity,

            "pan_x":
                0.015 * intensity,

            "pan_y":
                -0.008 * intensity,

            "shake":
                0.02 + 0.05 * intensity,

            "blur":
                0.0,

            "contrast":
                1.06 + 0.12 * intensity,

            "saturation":
                1.01 + 0.05 * intensity,
        }

    if role == "accent_zone":

        return {
            "zoom_start":
                1.01,

            "zoom_end":
                1.09 + 0.04 * intensity,

            "pan_x":
                0.025 * intensity,

            "pan_y":
                -0.015 * intensity,

            "shake":
                0.08 + 0.16 * intensity,

            "blur":
                0.03 + 0.04 * intensity,

            "contrast":
                1.08 + 0.16 * intensity,

            "saturation":
                1.02 + 0.07 * intensity,
        }

    if role == "action_zone":

        return {
            "zoom_start":
                1.015,

            "zoom_end":
                1.11 + 0.05 * intensity,

            "pan_x":
                0.035 * intensity,

            "pan_y":
                -0.02 * intensity,

            "shake":
                0.14 + 0.20 * intensity,

            "blur":
                0.04 + 0.07 * intensity,

            "contrast":
                1.10 + 0.20 * intensity,

            "saturation":
                1.03 + 0.08 * intensity,
        }

    # impact
    return {
        "zoom_start":
            1.02,

        "zoom_end":
            1.13 + 0.07 * intensity,

        "pan_x":
            0.04 * intensity,

        "pan_y":
            -0.025 * intensity,

        "shake":
            0.30 + 0.35 * intensity,

        "blur":
            0.06 + 0.12 * intensity,

        "contrast":
            1.12 + 0.24 * intensity,

        "saturation":
            1.04 + 0.10 * intensity,
    }


# ============================================================
# ACTION HELPERS
# ============================================================

def action(
    effect,
    start,
    end,
    **params
):

    return {
        "effect": effect,
        "start": round(
            max(0.0, start),
            4
        ),
        "end": round(
            max(start, end),
            4
        ),
        **params,
    }


def add_if_fits(
    actions,
    item,
    duration
):

    if len(actions) >= MAX_ACTIONS_PER_SEGMENT:
        return

    if (
        item["start"] >= duration
        or item["end"] <= 0
    ):
        return

    item["start"] = max(
        0.0,
        item["start"]
    )

    item["end"] = min(
        duration,
        item["end"]
    )

    if (
        item["end"]
        - item["start"]
        >= MIN_EVENT_DURATION
        or item["effect"]
        in {
            "flash",
            "impact_snap",
        }
    ):
        actions.append(item)


# ============================================================
# CINEMATIC CHOREOGRAPHY
# ============================================================

def cinematic_choreography(
    duration,
    camera,
    intensity
):

    actions = []

    add_if_fits(
        actions,
        action(
            "camera_zoom",
            0.0,
            duration,
            start_scale=
                camera["zoom_start"],
            end_scale=
                camera["zoom_end"],
            easing="ease_in_out",
        ),
        duration
    )

    if intensity > 0.18:

        add_if_fits(
            actions,
            action(
                "camera_pan",
                0.0,
                duration,
                x=camera["pan_x"],
                y=camera["pan_y"],
                easing="ease_in_out",
            ),
            duration
        )

    return actions


# ============================================================
# BUILD CHOREOGRAPHY
# ============================================================

def build_choreography(
    duration,
    camera,
    intensity,
    context,
    sync_type
):

    actions = []

    # ========================================================
    # CINEMATIC
    # ========================================================

    if context == "cinematic":

        return cinematic_choreography(
            duration,
            camera,
            intensity
        )

    # ========================================================
    # BUILD
    # ========================================================

    if context == "build":

        add_if_fits(
            actions,
            action(
                "camera_zoom",
                0.0,
                duration,
                start_scale=
                    camera["zoom_start"],
                end_scale=
                    camera["zoom_end"],
                easing="ease_in",
            ),
            duration
        )

        if intensity > 0.42:

            add_if_fits(
                actions,
                action(
                    "directional_motion",
                    0.0,
                    duration,
                    x=camera["pan_x"],
                    y=camera["pan_y"],
                    easing="ease_in",
                ),
                duration
            )

        # Small speed push toward the end.
        if (
            intensity > 0.55
            and duration >= 0.35
        ):

            add_if_fits(
                actions,
                action(
                    "speed_curve",
                    max(
                        0.0,
                        duration - 0.25
                    ),
                    duration,
                    start_speed=1.0,
                    end_speed=
                        1.05
                        + 0.10
                        * intensity,
                    easing="ease_in",
                ),
                duration
            )

        return actions

    # ========================================================
    # ACCENT
    # ========================================================

    if context == "accent":

        pre_end = duration * 0.58
        snap_start = duration * 0.58
        snap_end = duration * 0.68
        recovery_end = duration * 0.88

        # Pre-approach
        add_if_fits(
            actions,
            action(
                "camera_zoom",
                0.0,
                pre_end,
                start_scale=
                    camera["zoom_start"],
                end_scale=1.045,
                easing="ease_in",
            ),
            duration
        )

        # Snap
        add_if_fits(
            actions,
            action(
                "impact_snap",
                snap_start,
                snap_end,
                strength=
                    0.45
                    + 0.40
                    * intensity,
            ),
            duration
        )

        # Controlled shake
        if sync_type != "free":

            add_if_fits(
                actions,
                action(
                    "shake_burst",
                    snap_start,
                    min(
                        duration,
                        snap_end
                        + 0.10
                    ),
                    amplitude=
                        camera["shake"],
                    frequency=
                        14,
                ),
                duration
            )

        # Recovery
        add_if_fits(
            actions,
            action(
                "settle",
                snap_end,
                recovery_end,
                strength=
                    0.35
                    * intensity,
            ),
            duration
        )

        return actions

    # ========================================================
    # ACTION
    # ========================================================

    if context == "action":

        approach_end = duration * 0.50
        snap_start = duration * 0.50
        snap_end = duration * 0.61
        recovery_end = duration * 0.84

        add_if_fits(
            actions,
            action(
                "camera_zoom",
                0.0,
                approach_end,
                start_scale=1.0,
                end_scale=
                    1.06
                    + 0.04
                    * intensity,
                easing="ease_in",
            ),
            duration
        )

        add_if_fits(
            actions,
            action(
                "directional_motion",
                0.0,
                approach_end,
                x=camera["pan_x"],
                y=camera["pan_y"],
                easing="ease_in",
            ),
            duration
        )

        add_if_fits(
            actions,
            action(
                "impact_snap",
                snap_start,
                snap_end,
                strength=
                    0.55
                    + 0.35
                    * intensity,
            ),
            duration
        )

        add_if_fits(
            actions,
            action(
                "shake_burst",
                snap_start,
                min(
                    duration,
                    snap_end + 0.12
                ),
                amplitude=
                    camera["shake"],
                frequency=17,
            ),
            duration
        )

        add_if_fits(
            actions,
            action(
                "motion_blur",
                snap_start,
                recovery_end,
                amount=
                    camera["blur"],
                curve="burst",
            ),
            duration
        )

        add_if_fits(
            actions,
            action(
                "settle",
                recovery_end,
                duration,
                strength=
                    0.40
                    * intensity,
            ),
            duration
        )

        return actions

    # ========================================================
    # IMPACT
    # ========================================================

    if context == "impact":

        # For extremely short clips, compress the choreography.
        if duration < 0.30:

            add_if_fits(
                actions,
                action(
                    "pre_zoom",
                    0.0,
                    duration * 0.48,
                    start_scale=1.0,
                    end_scale=
                        1.07
                        + 0.04
                        * intensity,
                    easing="ease_in",
                ),
                duration
            )

            add_if_fits(
                actions,
                action(
                    "impact_snap",
                    duration * 0.43,
                    duration * 0.58,
                    strength=
                        0.70
                        + 0.30
                        * intensity,
                ),
                duration
            )

            add_if_fits(
                actions,
                action(
                    "flash",
                    duration * 0.46,
                    duration * 0.57,
                    strength=
                        0.45
                        + 0.45
                        * intensity,
                ),
                duration
            )

            add_if_fits(
                actions,
                action(
                    "shake_burst",
                    duration * 0.47,
                    duration * 0.76,
                    amplitude=
                        camera["shake"],
                    frequency=23,
                ),
                duration
            )

            add_if_fits(
                actions,
                action(
                    "settle",
                    duration * 0.68,
                    duration,
                    strength=
                        0.45
                        * intensity,
                ),
                duration
            )

            return actions

        # ----------------------------------------------------
        # Standard impact
        # ----------------------------------------------------

        pre_end = duration * 0.45
        snap_start = duration * 0.42
        snap_end = duration * 0.56
        flash_start = duration * 0.46
        flash_end = duration * 0.53
        recovery_start = duration * 0.55

        add_if_fits(
            actions,
            action(
                "pre_zoom",
                0.0,
                pre_end,
                start_scale=1.0,
                end_scale=
                    1.08
                    + 0.05
                    * intensity,
                easing="ease_in",
            ),
            duration
        )

        add_if_fits(
            actions,
            action(
                "impact_snap",
                snap_start,
                snap_end,
                strength=
                    0.75
                    + 0.25
                    * intensity,
            ),
            duration
        )

        # Flash only when the segment is strong enough.
        if intensity >= 0.60:

            add_if_fits(
                actions,
                action(
                    "flash",
                    flash_start,
                    flash_end,
                    strength=
                        0.45
                        + 0.45
                        * intensity,
                ),
                duration
            )

        add_if_fits(
            actions,
            action(
                "shake_burst",
                snap_start,
                min(
                    duration,
                    snap_end + 0.13
                ),
                amplitude=
                    camera["shake"],
                frequency=24,
            ),
            duration
        )

        add_if_fits(
            actions,
            action(
                "motion_blur",
                snap_start,
                min(
                    duration,
                    recovery_start + 0.15
                ),
                amount=
                    camera["blur"],
                curve="impact",
            ),
            duration
        )

        add_if_fits(
            actions,
            action(
                "settle",
                recovery_start,
                duration,
                strength=
                    0.45
                    * intensity,
            ),
            duration
        )

        return actions

    return cinematic_choreography(
        duration,
        camera,
        intensity
    )


# ============================================================
# CLASSIFY CONTEXT
# ============================================================

def classify_context(segment):

    role = segment.get(
        "editorial_role",
        "build_zone"
    )

    if role == "cinematic_zone":
        return "cinematic"

    if role == "build_zone":
        return "build"

    if role == "accent_zone":
        return "accent"

    if role == "action_zone":
        return "action"

    if role == "impact_zone":
        return "impact"

    return "build"


# ============================================================
# BUILD SEGMENT
# ============================================================

def process_segment(
    segment,
    reference_map
):

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

    intensity = segment_intensity(
        segment
    )

    context = classify_context(
        segment
    )

    midpoint = (
        start
        + duration / 2.0
    )

    reference_event = (
        nearest_reference_event(
            reference_map,
            midpoint
        )
    )

    camera = base_camera(
        segment,
        intensity
    )

    actions = build_choreography(
        duration,
        camera,
        intensity,
        context,
        segment.get(
            "sync_type",
            "free"
        )
    )

    finishing = {
        "contrast":
            round(
                camera["contrast"],
                4
            ),

        "saturation":
            round(
                camera["saturation"],
                4
            ),

        "vignette":
            round(
                0.03
                + 0.08
                * intensity,
                4
            ),

        "grain":
            round(
                0.01
                + 0.03
                * intensity,
                4
            ),
    }

    return {
        "segment":
            segment["segment"],

        "start":
            start,

        "end":
            end,

        "duration":
            duration,

        "editorial_role":
            segment.get(
                "editorial_role",
                "build_zone"
            ),

        "music_energy":
            safe_float(
                segment.get(
                    "music_energy",
                    0
                )
            ),

        "editorial_intensity":
            safe_float(
                segment.get(
                    "editorial_intensity",
                    0
                )
            ),

        "sync_type":
            segment.get(
                "sync_type",
                "free"
            ),

        "transition_context":
            segment.get(
                "transition_context",
                "stable"
            ),

        "choreography_context":
            context,

        "reference_influence":
            {
                "nearest_event_type":
                    reference_event[
                        "type"
                    ],

                "distance":
                    round(
                        reference_event[
                            "distance"
                        ],
                        4
                    ),

                "strength":
                    round(
                        reference_event[
                            "strength"
                        ],
                        4
                    ),
            },

        "camera_profile":
            {
                key:
                    round(
                        value,
                        4
                    )
                for key, value
                in camera.items()
            },

        "finishing":
            finishing,

        "actions":
            actions,

        "action_count":
            len(actions),

        "effect_principle":
            (
                "Effects are time-localized around "
                "editorial moments instead of being "
                "applied uniformly across the clip."
            ),
    }


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    segments
):

    action_counts = {}

    context_counts = {}

    max_actions = 0

    for segment in segments:

        context = segment[
            "choreography_context"
        ]

        context_counts[
            context
        ] = (
            context_counts.get(
                context,
                0
            )
            + 1
        )

        max_actions = max(
            max_actions,
            segment[
                "action_count"
            ]
        )

        for item in segment[
            "actions"
        ]:

            effect = item[
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
            sum(
                x["action_count"]
                for x in segments
            ),

        "maximum_actions_in_segment":
            max_actions,

        "context_distribution":
            context_counts,

        "effect_distribution":
            action_counts,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 82)
    print("🎬 EFFECT DIRECTOR V2")
    print("=" * 82)

    creative_plan = load_json(
        CREATIVE_PLAN_PATH
    )

    # Loaded explicitly so the pipeline remains traceable.
    music_plan = load_json(
        MUSIC_PLAN_PATH
    )

    reference_map = load_json(
        REFERENCE_MAP_PATH
    )

    segments = creative_plan.get(
        "segments",
        []
    )

    if not segments:

        raise RuntimeError(
            "No creative segments found."
        )

    print()
    print(
        f"Creative segments : "
        f"{len(segments)}"
    )

    print(
        f"Reference events  : "
        f"{len(reference_map.get('events', []))}"
    )

    print(
        f"Music segments    : "
        f"{len(music_plan.get('segments', []))}"
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    output_segments = []

    for segment in segments:

        output_segments.append(
            process_segment(
                segment,
                reference_map
            )
        )

    summary = build_summary(
        output_segments
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {

        "meta": {

            "name":
                "effect_director_plan_v2",

            "version":
                "2.0",

            "purpose":
                (
                    "Generate time-localized AMV effect "
                    "choreography rather than static filters."
                ),
        },

        "design_rules": [

            (
                "Effects must have temporal boundaries."
            ),

            (
                "Impact effects are concentrated around "
                "the actual editorial impact point."
            ),

            (
                "Shake is burst-based, not continuous."
            ),

            (
                "Flash is short and event-driven."
            ),

            (
                "Zoom uses a curve rather than a fixed scale."
            ),

            (
                "High-energy moments can stack effects."
            ),

            (
                "Cinematic sections remain restrained."
            ),

            (
                "Recovery/settle is part of the choreography."
            ),

            (
                "Reference controls effect grammar, not footage."
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
    print("✅ EFFECT DIRECTOR V2 COMPLETE")
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
        f"Max actions / segment : "
        f"{summary['maximum_actions_in_segment']}"
    )

    print()
    print("CHOREOGRAPHY")

    for key, value in (
        summary[
            "context_distribution"
        ].items()
    ):

        print(
            f"  {key:<12}: "
            f"{value}"
        )

    print()
    print("EFFECT ACTIONS")

    for key, value in sorted(
        summary[
            "effect_distribution"
        ].items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"  {key:<20}: "
            f"{value}"
        )

    print()
    print("FIRST 12 CHOREOGRAPHIES")

    for segment in output_segments[:12]:

        print()
        print(
            f"  #{segment['segment']:02d} "
            f"{segment['start']:.3f}→"
            f"{segment['end']:.3f}s "
            f"{segment['choreography_context']}"
        )

        for item in segment[
            "actions"
        ]:

            print(
                f"      "
                f"{item['start']:.3f}→"
                f"{item['end']:.3f} "
                f"{item['effect']}"
            )

    print()
    print("Saved:")
    print(OUTPUT_PATH)
    print("=" * 82)


if __name__ == "__main__":
    main()