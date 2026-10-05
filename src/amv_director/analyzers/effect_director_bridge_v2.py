from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

CREATIVE_PLAN = (
    BASE_DIR
    / "data/outputs/creative_director_plan_v3.json"
)

REFERENCE_EFFECT_PLAN = (
    BASE_DIR
    / "data/outputs/effect_director_plan_v3.json"
)

OUTPUT = (
    BASE_DIR
    / "data/outputs/effect_director_bridge_v2.json"
)


# ============================================================
# LIMITS
# ============================================================

MAX_ACTIONS_PER_SHOT = 6

MAX_FLASH_SHOTS = 4

MIN_DURATION = 0.025


# ============================================================
# HELPERS
# ============================================================

def f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
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
    with open(path, "r") as fh:
        return json.load(fh)


def add_action(
    actions: List[Dict[str, Any]],
    effect: str,
    at: float,
    duration: float,
    strength: float,
    **kwargs,
):
    actions.append({
        "effect": effect,
        "at": round(
            max(0.0, at),
            4,
        ),
        "duration": round(
            max(
                MIN_DURATION,
                duration,
            ),
            4,
        ),
        "strength": round(
            clamp(strength),
            4,
        ),
        **kwargs,
    })


# ============================================================
# REFERENCE STYLE PROFILE
# ============================================================

def build_reference_profile(
    reference_plan: Dict[str, Any],
) -> Dict[str, Any]:

    summary = reference_plan.get(
        "summary",
        {},
    )

    action_counts = summary.get(
        "action_counts",
        {},
    )

    # Existing Effect Director V3 already tells us the
    # reference's preferred vocabulary.
    return {

        "camera_drift":
            f(
                action_counts.get(
                    "camera_drift",
                    26,
                )
            ),

        "directional_motion":
            f(
                action_counts.get(
                    "directional_motion",
                    21,
                )
            ),

        "micro_push":
            f(
                action_counts.get(
                    "micro_push",
                    20,
                )
            ),

        "motion_blur":
            f(
                action_counts.get(
                    "motion_blur",
                    6,
                )
            ),

        "impact_snap":
            f(
                action_counts.get(
                    "impact_snap",
                    3,
                )
            ),

        "flash":
            f(
                action_counts.get(
                    "flash",
                    3,
                )
            ),

        "pre_push":
            f(
                action_counts.get(
                    "pre_push",
                    3,
                )
            ),

        "settle":
            f(
                action_counts.get(
                    "settle",
                    3,
                )
            ),
    }


# ============================================================
# CONTEXT
# ============================================================

def get_music_role(
    shot: Dict[str, Any],
) -> str:

    return (
        shot
        .get("music", {})
        .get(
            "role",
            "cinematic_zone",
        )
    )


def get_intensity(
    shot: Dict[str, Any],
) -> float:

    return clamp(
        f(
            shot
            .get("music", {})
            .get(
                "intensity",
                0.0,
            )
        )
    )


def get_visual_role(
    shot: Dict[str, Any],
) -> str:

    return (
        shot
        .get("director", {})
        .get(
            "visual_role",
            "character",
        )
    )


def get_camera_behavior(
    shot: Dict[str, Any],
) -> List[str]:

    return [
        str(x)
        for x in (
            shot
            .get("director", {})
            .get(
                "camera_behavior",
                [],
            )
        )
    ]


def get_sequence(
    shot: Dict[str, Any],
) -> str | None:

    events = (
        shot
        .get("reference", {})
        .get(
            "active_av_events",
            [],
        )
    )

    if not events:
        return None

    return events[0].get(
        "sequence_type"
    )


def get_event_role(
    shot: Dict[str, Any],
) -> str | None:

    events = (
        shot
        .get("reference", {})
        .get(
            "active_av_events",
            [],
        )
    )

    if not events:
        return None

    return events[0].get(
        "event_role"
    )


# ============================================================
# CAMERA LANGUAGE
# ============================================================

def build_camera_actions(
    shot: Dict[str, Any],
) -> List[Dict[str, Any]]:

    actions = []

    duration = max(
        MIN_DURATION,
        f(
            shot["timeline"].get(
                "duration",
                0.1,
            )
        ),
    )

    intensity = get_intensity(
        shot
    )

    visual_role = get_visual_role(
        shot
    )

    music_role = get_music_role(
        shot
    )

    behaviors = get_camera_behavior(
        shot
    )

    # --------------------------------------------------------
    # Cinematic
    # --------------------------------------------------------

    if (
        music_role
        == "cinematic_zone"
    ):

        direction = (
            "left"
            if shot["shot_id"] % 2 == 0
            else "right"
        )

        add_action(
            actions,
            "camera_drift",
            0.0,
            duration,
            0.40
            + 0.20 * intensity,
            direction=direction,
        )

        # Character closeups receive a little push.
        if (
            visual_role
            in {
                "closeup",
                "character",
            }
            and intensity > 0.35
        ):

            add_action(
                actions,
                "micro_push",
                max(
                    0.0,
                    duration * 0.58,
                ),
                min(
                    0.20,
                    duration * 0.20,
                ),
                0.25
                + 0.15 * intensity,
            )

        return actions

    # --------------------------------------------------------
    # Explicit Director behaviors
    # --------------------------------------------------------

    behavior_text = " ".join(
        behaviors
    ).lower()

    if any(
        x in behavior_text
        for x in (
            "directional_motion",
            "directional_move",
            "directional",
        )
    ):

        direction = (
            "left"
            if shot["shot_id"] % 2 == 0
            else "right"
        )

        add_action(
            actions,
            "directional_motion",
            0.0,
            max(
                0.08,
                duration * 0.65,
            ),
            0.42
            + 0.20 * intensity,
            direction=direction,
        )

    elif any(
        x in behavior_text
        for x in (
            "slow_drift",
            "subtle_push",
            "restrained_motion",
        )
    ):

        add_action(
            actions,
            "camera_drift",
            0.0,
            duration,
            0.38
            + 0.16 * intensity,
            direction=(
                "left"
                if shot["shot_id"] % 2 == 0
                else "right"
            ),
        )

    elif any(
        x in behavior_text
        for x in (
            "controlled_push",
            "gradual_push",
            "rising_motion",
        )
    ):

        add_action(
            actions,
            "micro_push",
            0.0,
            max(
                0.10,
                duration * 0.72,
            ),
            0.38
            + 0.22 * intensity,
        )

    elif any(
        x in behavior_text
        for x in (
            "compressed_push",
            "impact_emphasis",
            "fast_reframe",
        )
    ):

        add_action(
            actions,
            "micro_push",
            0.0,
            max(
                0.10,
                duration * 0.48,
            ),
            0.48
            + 0.20 * intensity,
        )

    elif any(
        x in behavior_text
        for x in (
            "controlled_hold",
            "hold",
        )
    ):

        # Intentionally clean.
        pass

    else:

        # Fallback based on role.
        if visual_role == "environment":

            add_action(
                actions,
                "camera_drift",
                0.0,
                duration,
                0.35,
                direction="left",
            )

        elif visual_role == "motion":

            add_action(
                actions,
                "directional_motion",
                0.0,
                max(
                    0.08,
                    duration * 0.65,
                ),
                0.42,
                direction=(
                    "left"
                    if shot["shot_id"] % 2 == 0
                    else "right"
                ),
            )

        else:

            add_action(
                actions,
                "micro_push",
                0.0,
                max(
                    0.10,
                    duration * 0.70,
                ),
                0.30,
            )

    return actions


# ============================================================
# REFERENCE AV CHOREOGRAPHY
# ============================================================

def build_av_actions(
    shot: Dict[str, Any],
    flash_budget_used: int,
) -> tuple[List[Dict[str, Any]], int]:

    actions = []

    duration = max(
        MIN_DURATION,
        f(
            shot["timeline"].get(
                "duration",
                0.1,
            )
        ),
    )

    intensity = get_intensity(
        shot
    )

    sequence = get_sequence(
        shot
    )

    event_role = get_event_role(
        shot
    )

    # --------------------------------------------------------
    # Build → Impact
    # --------------------------------------------------------

    if sequence == "BUILD_TO_IMPACT":

        impact_at = (
            duration * 0.74
        )

        add_action(
            actions,
            "pre_push",
            max(
                0.0,
                impact_at - 0.18,
            ),
            min(
                0.14,
                duration * 0.15,
            ),
            0.45
            + 0.12 * intensity,
        )

        add_action(
            actions,
            "impact_snap",
            impact_at,
            min(
                0.07,
                duration * 0.10,
            ),
            0.65
            + 0.25 * intensity,
        )

        # Reference has very few flashes, so keep them rare.
        if (
            flash_budget_used
            < MAX_FLASH_SHOTS
            and shot["shot_id"] % 11 == 0
        ):

            add_action(
                actions,
                "flash",
                max(
                    0.0,
                    impact_at - 0.015,
                ),
                0.04,
                0.22
                + 0.10 * intensity,
                mode="white",
            )

            flash_budget_used += 1

        # One small blur around stronger builds.
        if intensity >= 0.65:

            add_action(
                actions,
                "motion_blur",
                max(
                    0.0,
                    impact_at - 0.10,
                ),
                min(
                    0.18,
                    duration * 0.16,
                ),
                0.16
                + 0.10 * intensity,
            )

    # --------------------------------------------------------
    # Direct impact
    # --------------------------------------------------------

    elif sequence == "DIRECT_IMPACT":

        impact_at = (
            duration * 0.72
        )

        add_action(
            actions,
            "impact_snap",
            impact_at,
            min(
                0.07,
                duration * 0.10,
            ),
            0.58
            + 0.25 * intensity,
        )

    # --------------------------------------------------------
    # Motion accent
    # --------------------------------------------------------

    elif sequence == "MOTION_ACCENT":

        add_action(
            actions,
            "directional_motion",
            duration * 0.05,
            max(
                0.08,
                duration * 0.45,
            ),
            0.48,
            direction=(
                "left"
                if shot["shot_id"] % 2 == 0
                else "right"
            ),
        )

        if intensity >= 0.55:

            add_action(
                actions,
                "motion_blur",
                duration * 0.08,
                max(
                    0.06,
                    duration * 0.28,
                ),
                0.18,
            )

    # --------------------------------------------------------
    # Event-specific accent
    # --------------------------------------------------------

    if event_role:

        role_text = event_role.lower()

        if (
            "scream" in role_text
            and duration > 0.20
        ):

            add_action(
                actions,
                "micro_push",
                duration * 0.58,
                min(
                    0.14,
                    duration * 0.12,
                ),
                0.34,
            )

        elif (
            "glass" in role_text
            and duration > 0.20
        ):

            add_action(
                actions,
                "flash",
                duration * 0.70,
                0.035,
                0.18,
                mode="white",
            )

    return actions, flash_budget_used


# ============================================================
# MERGE
# ============================================================

def merge_actions(
    camera_actions: List[Dict[str, Any]],
    av_actions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    actions = (
        camera_actions
        + av_actions
    )

    # --------------------------------------------------------
    # Remove near duplicates of the same effect.
    # --------------------------------------------------------

    result = []

    for item in sorted(
        actions,
        key=lambda x: (
            x["at"],
            x["effect"],
        ),
    ):

        duplicate = False

        for existing in result:

            if (
                existing["effect"]
                == item["effect"]
                and abs(
                    existing["at"]
                    - item["at"]
                )
                < 0.08
            ):

                duplicate = True

                break

        if not duplicate:
            result.append(
                item
            )

    # --------------------------------------------------------
    # Priority:
    # impact > flash > pre-push > directional > blur >
    # push > drift
    # --------------------------------------------------------

    priority = {

        "impact_snap": 10,
        "flash": 9,
        "pre_push": 8,
        "directional_motion": 7,
        "motion_blur": 6,
        "micro_push": 5,
        "camera_drift": 3,

    }

    result.sort(
        key=lambda x: (
            priority.get(
                x["effect"],
                1,
            ),
            x["strength"],
        ),
        reverse=True,
    )

    result = result[
        :MAX_ACTIONS_PER_SHOT
    ]

    result.sort(
        key=lambda x: (
            x["at"],
            x["effect"],
        )
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("🎬 EFFECT DIRECTOR BRIDGE V2")
    print("=" * 72)

    creative = load_json(
        CREATIVE_PLAN
    )

    reference_effects = load_json(
        REFERENCE_EFFECT_PLAN
    )

    shots = creative.get(
        "shot_plan",
        [],
    )

    if not shots:

        raise RuntimeError(
            "No shot_plan found."
        )

    reference_profile = (
        build_reference_profile(
            reference_effects
        )
    )

    print(
        f"\nCreative shots : "
        f"{len(shots)}"
    )

    print("\nReference vocabulary:")

    for key, value in sorted(
        reference_profile.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"  {key:<22}: "
            f"{int(value)}"
        )

    plans = []

    flash_budget_used = 0

    for index, shot in enumerate(
        shots,
        start=1,
    ):

        camera_actions = (
            build_camera_actions(
                shot
            )
        )

        av_actions, flash_budget_used = (
            build_av_actions(
                shot,
                flash_budget_used,
            )
        )

        final_actions = merge_actions(
            camera_actions,
            av_actions,
        )

        plans.append({

            "shot_id":
                shot["shot_id"],

            "timeline":
                shot["timeline"],

            "music":
                shot["music"],

            "source":
                shot["source"],

            "director":
                shot["director"],

            "reference":
                shot["reference"],

            "effect_context": {

                "music_role":
                    get_music_role(shot),

                "visual_role":
                    get_visual_role(shot),

                "reference_sequence":
                    get_sequence(shot),

                "reference_event_role":
                    get_event_role(shot),

                "camera_behavior":
                    get_camera_behavior(shot),

            },

            "actions":
                final_actions,

        })

        print(
            f"[{index:02d}/{len(shots):02d}] "
            f"{shot['timeline']['start']:6.3f}"
            f"→"
            f"{shot['timeline']['end']:6.3f} | "
            f"{get_music_role(shot):<16} | "
            f"{get_sequence(shot) or 'NONE':<18} | "
            f"{' → '.join(x['effect'] for x in final_actions) if final_actions else 'CLEAN'}"
        )

    # ========================================================
    # AUDIT
    # ========================================================

    action_counts: Dict[str, int] = {}

    choreography_counts: Dict[str, int] = {}

    total_actions = 0

    max_actions = 0

    for plan in plans:

        total_actions += len(
            plan["actions"]
        )

        max_actions = max(
            max_actions,
            len(
                plan["actions"]
            ),
        )

        for item in plan["actions"]:

            effect = item["effect"]

            action_counts[effect] = (
                action_counts.get(
                    effect,
                    0,
                )
                + 1
            )

        sequence = (
            plan[
                "effect_context"
            ]["reference_sequence"]
            or "NONE"
        )

        choreography_counts[
            sequence
        ] = (
            choreography_counts.get(
                sequence,
                0,
            )
            + 1
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    result = {

        "meta": {

            "version":
                "v2",

            "purpose":
                (
                    "Integrates Creative Director camera "
                    "language with reference AV choreography "
                    "without relying on old segment timestamps."
                ),

            "principles":
                {

                    "creative_director":
                        "primary camera instruction",

                    "reference_effect_director":
                        "effect vocabulary/style reference",

                    "av_grammar":
                        "impact sequence timing",

                    "music":
                        "timing backbone",

                    "renderer":
                        "deterministic execution",

                },

        },

        "summary": {

            "shots":
                len(plans),

            "total_actions":
                total_actions,

            "average_actions_per_shot":
                round(
                    total_actions
                    / len(plans),
                    3,
                ),

            "max_actions_per_shot":
                max_actions,

            "flash_budget_used":
                flash_budget_used,

            "action_distribution":
                action_counts,

            "sequence_distribution":
                choreography_counts,

        },

        "shot_effects":
            plans,

    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT,
        "w",
    ) as fh:

        json.dump(
            result,
            fh,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("✅ EFFECT DIRECTOR BRIDGE V2 COMPLETE")
    print("=" * 72)

    print(
        f"\nShots              : "
        f"{len(plans)}"
    )

    print(
        f"Total actions      : "
        f"{total_actions}"
    )

    print(
        f"Average / shot    : "
        f"{total_actions / len(plans):.2f}"
    )

    print(
        f"Max actions / shot: "
        f"{max_actions}"
    )

    print(
        f"Flash budget used : "
        f"{flash_budget_used}"
    )

    print("\nAction distribution:")

    for key, value in sorted(
        action_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"  {key:<22}: "
            f"{value}"
        )

    print("\nSequence distribution:")

    for key, value in sorted(
        choreography_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"  {key:<22}: "
            f"{value}"
        )

    print("\nFirst 15 shot choreographies:")

    for plan in plans[:15]:

        names = [
            x["effect"]
            for x in plan["actions"]
        ]

        print(
            f"  Shot {plan['shot_id']:02d} | "
            f"{' → '.join(names) if names else 'CLEAN'}"
        )

    print(
        f"\nOutput: {OUTPUT}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()