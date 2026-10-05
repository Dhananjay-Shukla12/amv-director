from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parents[3]

CREATIVE_PLAN = (
    BASE_DIR
    / "data/outputs/creative_director_plan_v3.json"
)

MUSIC_LOCK = (
    BASE_DIR
    / "data/outputs/music_locked_edit_plan_v3.json"
)

OLD_EFFECT_PLAN = (
    BASE_DIR
    / "data/outputs/effect_director_plan_v3.json"
)

OUTPUT = (
    BASE_DIR
    / "data/outputs/effect_director_bridge_v1.json"
)


def f(value: Any, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def load_json(path: Path):
    with open(path, "r") as fh:
        return json.load(fh)


def overlap(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:

    left = max(a_start, b_start)
    right = min(a_end, b_end)

    return max(
        0.0,
        right - left,
    )


def copy_action(
    action: Dict[str, Any],
    old_start: float,
    old_end: float,
    new_start: float,
    new_end: float,
) -> Dict[str, Any]:

    old_duration = max(
        0.001,
        old_end - old_start,
    )

    new_duration = max(
        0.001,
        new_end - new_start,
    )

    old_at = f(
        action.get("at", 0.0)
    )

    old_action_duration = f(
        action.get(
            "duration",
            0.05,
        )
    )

    # Convert absolute/global action time into relative
    # time within its old segment.
    relative_at = old_at - old_start

    # Preserve proportional position when mapping to shot.
    normalized = max(
        0.0,
        min(
            1.0,
            relative_at / old_duration,
        ),
    )

    new_at = (
        new_start
        + normalized * new_duration
    )

    duration_ratio = (
        old_action_duration
        / old_duration
    )

    mapped_duration = max(
        0.025,
        min(
            0.40,
            duration_ratio * new_duration,
        ),
    )

    result = dict(action)

    result["at"] = round(
        new_at,
        4,
    )

    result["duration"] = round(
        min(
            mapped_duration,
            max(
                0.025,
                new_end - new_at,
            ),
        ),
        4,
    )

    result["bridge_source"] = (
        "reference_effect_director_v3"
    )

    return result


def effect_segments(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:

    for key in (
        "segments",
        "effect_segments",
        "timeline",
    ):

        value = data.get(key)

        if isinstance(value, list):
            return value

    return []


def get_actions(
    segment: Dict[str, Any],
) -> List[Dict[str, Any]]:

    for key in (
        "actions",
        "timed_actions",
        "effect_actions",
    ):

        value = segment.get(key)

        if isinstance(value, list):
            return value

    return []


def event_sequence(
    shot: Dict[str, Any],
) -> str | None:

    events = (
        shot
        .get("reference", {})
        .get("active_av_events", [])
    )

    if not events:
        return None

    return events[0].get(
        "sequence_type"
    )


def music_role(
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


def make_extra_actions(
    shot: Dict[str, Any],
) -> List[Dict[str, Any]]:

    start = f(
        shot["timeline"]["start"]
    )

    duration = max(
        0.05,
        f(
            shot["timeline"]["duration"]
        ),
    )

    sequence = event_sequence(
        shot
    )

    role = music_role(
        shot
    )

    extra = []

    # --------------------------------------------------------
    # We only add actions that complement the existing
    # reference-derived motion layer.
    # --------------------------------------------------------

    if sequence == "BUILD_TO_IMPACT":

        extra.append({
            "effect": "pre_push",
            "at": round(
                start
                + duration * 0.48,
                4,
            ),
            "duration": round(
                min(
                    0.12,
                    duration * 0.12,
                ),
                4,
            ),
            "strength": 0.45,
            "bridge_source": "av_grammar",
        })

        extra.append({
            "effect": "impact_snap",
            "at": round(
                start
                + duration * 0.76,
                4,
            ),
            "duration": 0.055,
            "strength": 0.75,
            "bridge_source": "av_grammar",
        })

    elif sequence == "DIRECT_IMPACT":

        extra.append({
            "effect": "impact_snap",
            "at": round(
                start
                + duration * 0.72,
                4,
            ),
            "duration": 0.055,
            "strength": 0.70,
            "bridge_source": "av_grammar",
        })

    elif sequence == "MOTION_ACCENT":

        extra.append({
            "effect": "directional_motion",
            "at": round(
                start
                + duration * 0.15,
                4,
            ),
            "duration": round(
                max(
                    0.08,
                    duration * 0.45,
                ),
                4,
            ),
            "strength": 0.45,
            "bridge_source": "av_grammar",
        })

    elif role == "accent_zone":

        extra.append({
            "effect": "micro_push",
            "at": round(
                start
                + duration * 0.55,
                4,
            ),
            "duration": round(
                max(
                    0.08,
                    duration * 0.18,
                ),
                4,
            ),
            "strength": 0.30,
            "bridge_source": "music_role",
        })

    return extra


def main():

    print("=" * 72)
    print("🎬 EFFECT DIRECTOR BRIDGE V1")
    print("=" * 72)

    creative = load_json(
        CREATIVE_PLAN
    )

    music_lock = load_json(
        MUSIC_LOCK
    )

    old_effects = load_json(
        OLD_EFFECT_PLAN
    )

    shots = creative.get(
        "shot_plan",
        [],
    )

    effect_segments_data = (
        effect_segments(
            old_effects
        )
    )

    if not shots:
        raise RuntimeError(
            "No shot_plan found."
        )

    if not effect_segments_data:
        raise RuntimeError(
            "Could not find effect segments "
            "in effect_director_plan_v3.json."
        )

    print(
        f"\nCreative shots     : {len(shots)}"
    )

    print(
        f"Reference segments : "
        f"{len(effect_segments_data)}"
    )

    bridged = []

    for index, shot in enumerate(
        shots,
        start=1,
    ):

        shot_start = f(
            shot["timeline"]["start"]
        )

        shot_end = f(
            shot["timeline"]["end"]
        )

        shot_duration = max(
            0.05,
            shot_end - shot_start,
        )

        mapped_actions = []

        matched_segments = []

        # ----------------------------------------------------
        # Map old effect segments that overlap this new shot.
        # ----------------------------------------------------

        for segment in effect_segments_data:

            seg_start = f(
                segment.get("start")
            )

            seg_end = f(
                segment.get("end")
            )

            ov = overlap(
                shot_start,
                shot_end,
                seg_start,
                seg_end,
            )

            if ov <= 0:
                continue

            matched_segments.append(
                segment
            )

            actions = get_actions(
                segment
            )

            for action in actions:

                old_at = f(
                    action.get("at")
                )

                old_action_end = (
                    old_at
                    + f(
                        action.get(
                            "duration",
                            0.05,
                        )
                    )
                )

                if overlap(
                    shot_start,
                    shot_end,
                    old_at,
                    old_action_end,
                ) <= 0:

                    continue

                mapped = copy_action(
                    action,
                    seg_start,
                    seg_end,
                    shot_start,
                    shot_end,
                )

                mapped_actions.append(
                    mapped
                )

        # ----------------------------------------------------
        # Add AV grammar actions.
        # ----------------------------------------------------

        mapped_actions.extend(
            make_extra_actions(
                shot
            )
        )

        # ----------------------------------------------------
        # Deduplicate approximate duplicates.
        # ----------------------------------------------------

        final_actions = []

        for action in sorted(
            mapped_actions,
            key=lambda x: (
                f(x.get("at")),
                str(
                    x.get("effect")
                ),
            ),
        ):

            duplicate = False

            for existing in final_actions:

                if (
                    existing["effect"]
                    == action["effect"]
                    and abs(
                        existing["at"]
                        - action["at"]
                    )
                    < 0.07
                ):

                    duplicate = True

                    break

            if not duplicate:
                final_actions.append(
                    action
                )

        # ----------------------------------------------------
        # Safety cap.
        # ----------------------------------------------------

        priority = {

            "impact_snap": 10,
            "flash": 9,
            "pre_push": 8,
            "directional_motion": 7,
            "motion_blur": 6,
            "micro_push": 5,
            "camera_drift": 4,
            "settle": 3,

        }

        final_actions.sort(
            key=lambda x: (
                priority.get(
                    x.get(
                        "effect"
                    ),
                    1,
                ),
                -f(
                    x.get(
                        "strength",
                        0.0,
                    )
                ),
            ),
            reverse=True,
        )

        final_actions = final_actions[:6]

        final_actions.sort(
            key=lambda x: f(
                x.get("at")
            )
        )

        bridged.append({

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

                "matched_reference_segments":
                    len(matched_segments),

                "reference_motion_types": sorted(
                    set(
                        str(
                            seg.get(
                                "reference_motion",
                                seg.get(
                                    "motion_type",
                                    "unknown",
                                ),
                            )
                        )
                        for seg
                        in matched_segments
                    )
                ),

                "av_sequence":
                    event_sequence(
                        shot
                    ),

            },

            "actions":
                final_actions,

        })

        print(
            f"[{index:02d}/{len(shots):02d}] "
            f"{shot_start:6.3f}"
            f"→"
            f"{shot_end:6.3f} | "
            f"{music_role(shot):<16} | "
            f"{event_sequence(shot) or 'NONE':<18} | "
            f"actions={len(final_actions)}"
        )

    # ========================================================
    # AUDIT
    # ========================================================

    total_actions = sum(
        len(
            x["actions"]
        )
        for x in bridged
    )

    counts = {}

    for shot in bridged:

        for action in shot["actions"]:

            effect = action.get(
                "effect",
                "unknown",
            )

            counts[effect] = (
                counts.get(
                    effect,
                    0,
                )
                + 1
            )

    result = {

        "meta": {

            "version": "v1",

            "purpose": (
                "Maps the existing reference motion "
                "Effect Director onto the new 47-shot "
                "Creative Director timeline."
            ),

            "inputs": {

                "creative":
                    str(
                        CREATIVE_PLAN
                    ),

                "music_lock":
                    str(
                        MUSIC_LOCK
                    ),

                "reference_effects":
                    str(
                        OLD_EFFECT_PLAN
                    ),

            },

        },

        "summary": {

            "shots":
                len(bridged),

            "total_actions":
                total_actions,

            "average_actions_per_shot":
                round(
                    total_actions
                    / len(bridged),
                    3,
                ),

            "action_distribution":
                counts,

        },

        "shot_effects":
            bridged,

    }

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
    print("✅ EFFECT DIRECTOR BRIDGE COMPLETE")
    print("=" * 72)

    print(
        f"\nShots          : "
        f"{len(bridged)}"
    )

    print(
        f"Total actions  : "
        f"{total_actions}"
    )

    print(
        f"Average / shot : "
        f"{total_actions / len(bridged):.2f}"
    )

    print("\nAction distribution:")

    for key, value in sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"  {key:<22}: "
            f"{value}"
        )

    print("\nFirst 15 shot effect plans:")

    for shot in bridged[:15]:

        names = [
            x.get(
                "effect",
                "unknown",
            )
            for x in shot["actions"]
        ]

        print(
            f"  Shot {shot['shot_id']:02d} | "
            f"{' → '.join(names) if names else 'CLEAN'}"
        )

    print(
        f"\nOutput: {OUTPUT}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()