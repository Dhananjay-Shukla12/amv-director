from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


EPSILON = 0.01
MAX_PREFERRED_REUSE = 2

SOFT_CAPS = {
    "impact_zone": 0.90,
    "accent_zone": 0.90,
    "action_zone": 1.10,
    "build_zone": 1.40,
    "cinematic_zone": 2.20,
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def num(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def source_duration(source: dict) -> float:
    start = num(source.get("original_start"))
    end = num(source.get("original_end"))

    if start is not None and end is not None:
        return max(0.0, end - start)

    window = source.get("selected_window")

    if isinstance(window, dict):
        start = num(window.get("start"))
        end = num(window.get("end"))

        if start is not None and end is not None:
            return max(0.0, end - start)

    if isinstance(window, list) and len(window) >= 2:
        start = num(window[0])
        end = num(window[1])

        if start is not None and end is not None:
            return max(0.0, end - start)

    return 0.0


def selected_window(source: dict) -> tuple[float, float]:
    """
    Return the currently selected source window.

    Falls back to the full original scene.
    """

    window = source.get("selected_window")

    if isinstance(window, dict):
        start = num(window.get("start"))
        end = num(window.get("end"))

        if start is not None and end is not None:
            return start, end

    if isinstance(window, list) and len(window) >= 2:
        start = num(window[0])
        end = num(window[1])

        if start is not None and end is not None:
            return start, end

    start = num(source.get("original_start"), 0.0)
    end = num(source.get("original_end"), start)

    return start, end


def scene_key(video: str | None, scene_id) -> str:
    return f"{video or 'unknown'}::{scene_id if scene_id is not None else 'unknown'}"


def build_catalog(plan: dict) -> dict:
    """
    Build a source-scene catalog from the existing Creative Director plan.

    We use scenes already known to the plan so Temporal Editor V2 does not
    depend on a new external scene-analysis schema.
    """

    catalog = {}

    for shot in plan.get("shot_plan", []):
        source = shot.get("source", {})

        if not isinstance(source, dict):
            continue

        video = source.get("video")
        sid = source.get("scene_id")

        key = scene_key(video, sid)

        duration = source_duration(source)

        if key not in catalog:
            catalog[key] = {
                "key": key,
                "video": video,
                "scene_id": sid,
                "source_role": source.get(
                    "source_role",
                    "unknown",
                ),
                "original_start": num(
                    source.get("original_start"),
                    0.0,
                ),
                "original_end": num(
                    source.get("original_end"),
                    0.0,
                ),
                "duration": duration,
                "selection_score": num(
                    source.get("selection_score"),
                    0.0,
                ),
                "usage_count": 0,
            }

        catalog[key]["usage_count"] += 1

        # Prefer the longest known source span for candidate reuse.
        if duration > catalog[key]["duration"]:
            catalog[key]["duration"] = duration
            catalog[key]["original_start"] = num(
                source.get("original_start"),
                catalog[key]["original_start"],
            )
            catalog[key]["original_end"] = num(
                source.get("original_end"),
                catalog[key]["original_end"],
            )

    return catalog


def extract_parent_shots(plan: dict) -> list[dict]:
    result = []

    for shot in plan.get("shot_plan", []):
        timeline = shot.get("timeline", {})
        music = shot.get("music", {})
        director = shot.get("director", {})
        reference = shot.get("reference", {})
        source = shot.get("source", {})
        alternatives = shot.get("alternatives", [])

        start = num(timeline.get("start"), 0.0)
        end = num(timeline.get("end"), start)
        duration = max(
            0.0,
            end - start,
        )

        source_start, source_end = selected_window(
            source
        )

        source_span = max(
            0.0,
            source_end - source_start,
        )

        result.append(
            {
                "shot_id": shot.get("shot_id"),
                "start": start,
                "end": end,
                "duration": duration,
                "music": music,
                "director": director,
                "reference": reference,
                "source": source,
                "alternatives": (
                    alternatives
                    if isinstance(alternatives, list)
                    else []
                ),
                "source_start": source_start,
                "source_end": source_end,
                "source_span": source_span,
                "music_role": music.get(
                    "role",
                    "unknown",
                ),
                "visual_role": director.get(
                    "visual_role",
                    "unknown",
                ),
            }
        )

    return result


def candidate_from_catalog(
    candidate: dict,
    catalog: dict,
) -> dict | None:

    video = candidate.get("video")
    sid = candidate.get("scene_id")

    key = scene_key(video, sid)

    item = catalog.get(key)

    if item is None:
        return None

    return item.copy()


def candidate_score(
    candidate: dict,
    parent: dict,
    current_usage: Counter,
    previous_scene: str | None,
) -> float:

    score = num(
        candidate.get("selection_score"),
        0.0,
    ) or 0.0

    if candidate.get("source_role") == parent["music_role"]:
        score += 0.20

    if candidate.get("source_role") == parent["visual_role"]:
        score += 0.15

    key = candidate["key"]

    usage = current_usage.get(
        key,
        candidate.get("usage_count", 0),
    )

    # Strong preference for under-used scenes.
    score -= usage * 0.25

    if usage >= MAX_PREFERRED_REUSE:
        score -= 0.50

    if key == previous_scene:
        score -= 10.0

    current_source = parent["source"]
    current_key = scene_key(
        current_source.get("video"),
        current_source.get("scene_id"),
    )

    if key == current_key:
        score -= 2.0

    return score


def choose_candidate(
    parent: dict,
    catalog: dict,
    current_usage: Counter,
    previous_scene: str | None,
    used_in_parent: set[str],
) -> dict | None:

    candidates = []

    # First use the Creative Director's explicit alternatives.
    for alternative in parent["alternatives"]:

        if not isinstance(alternative, dict):
            continue

        candidate = candidate_from_catalog(
            alternative,
            catalog,
        )

        if candidate is None:
            continue

        if candidate["duration"] <= EPSILON:
            continue

        if candidate["key"] in used_in_parent:
            continue

        candidates.append(candidate)

    # Then fall back to other scenes already present in the plan.
    for candidate in catalog.values():

        if candidate["duration"] <= EPSILON:
            continue

        if candidate["key"] in used_in_parent:
            continue

        candidates.append(candidate)

    if not candidates:
        return None

    # Remove duplicates.
    unique = {}

    for candidate in candidates:
        unique[candidate["key"]] = candidate

    candidates = list(unique.values())

    ranked = sorted(
        candidates,
        key=lambda c: candidate_score(
            c,
            parent,
            current_usage,
            previous_scene,
        ),
        reverse=True,
    )

    return ranked[0]


def make_source_window(
    candidate: dict,
    duration: float,
) -> tuple[float, float]:

    available = candidate["duration"]

    original_start = candidate["original_start"]
    original_end = candidate["original_end"]

    if available <= EPSILON:
        return original_start, original_start

    if duration >= available - EPSILON:
        return (
            original_start,
            original_end,
        )

    # Take a centered window from the scene so we don't repeatedly use
    # the same beginning frame when a candidate is reused.
    midpoint = (
        original_start
        + original_end
    ) / 2.0

    start = midpoint - duration / 2.0
    end = start + duration

    start = max(
        original_start,
        start,
    )

    end = min(
        original_end,
        end,
    )

    # Re-correct after clamping.
    if end - start < duration:
        start = max(
            original_start,
            end - duration,
        )

    return start, end


def preferred_piece_duration(
    remaining: float,
    available: float,
    music_role: str,
) -> float:

    soft_cap = SOFT_CAPS.get(
        music_role,
        1.10,
    )

    # For a source that already nearly satisfies the duration, prefer
    # preserving a continuous shot rather than introducing unnecessary cuts.
    if available >= remaining - EPSILON:
        return remaining

    # Otherwise split the missing duration into editorially reasonable pieces.
    return min(
        remaining,
        available,
        soft_cap,
    )


def build_temporal_plan(
    plan: dict,
) -> tuple[dict, dict]:

    parents = extract_parent_shots(plan)
    catalog = build_catalog(plan)

   # Start usage counts from the original Creative Director plan.
    usage = Counter(
    {
        key: item["usage_count"]
        for key, item in catalog.items()
    }
    )

    output_shots = []

    unresolved = []

    rewritten_parent_count = 0
    total_original_duration = 0.0
    total_new_duration = 0.0

    previous_scene = None

    for parent in parents:

        parent_duration = parent["duration"]
        total_original_duration += parent_duration

        remaining = parent_duration
        cursor = parent["start"]

        source = parent["source"]

        current_key = scene_key(
            source.get("video"),
            source.get("scene_id"),
        )

        parent_used = set()

        parent_parts = []

        # -------------------------------------------------------------
        # PART 1: Use the Creative Director's selected window first.
        # -------------------------------------------------------------

        if parent["source_span"] > EPSILON:

            first_duration = min(
                remaining,
                parent["source_span"],
            )

            source_start = parent["source_start"]
            source_end = (
                source_start
                + first_duration
            )

            part = {
                "parent_shot_id": parent["shot_id"],
                "part_index": len(parent_parts) + 1,
                "timeline": {
                    "start": cursor,
                    "end": cursor + first_duration,
                    "duration": first_duration,
                },
                "music": parent["music"],
                "reference": parent["reference"],
                "director": parent["director"],
                "source": {
                    "video": source.get("video"),
                    "scene_id": source.get("scene_id"),
                    "source_role": source.get(
                        "source_role",
                        "unknown",
                    ),
                    "source_start": source_start,
                    "source_end": source_end,
                    "duration": first_duration,
                },
                "temporal": {
                    "method": (
                        "original_selected_window"
                    ),
                    "loop_required": False,
                },
            }

            parent_parts.append(part)

            remaining -= first_duration
            cursor += first_duration
            parent_used.add(current_key)

        # -------------------------------------------------------------
        # PARTS 2+: Fill missing duration with additional scenes.
        # -------------------------------------------------------------

        while remaining > EPSILON:

            candidate = choose_candidate(
                parent=parent,
                catalog=catalog,
                current_usage=usage,
                previous_scene=previous_scene,
                used_in_parent=parent_used,
            )

            if candidate is None:

                unresolved.append(
                    {
                        "parent_shot_id": parent["shot_id"],
                        "remaining_duration": round(
                            remaining,
                            3,
                        ),
                        "reason": (
                            "No candidate source scene "
                            "available without violating "
                            "temporal constraints."
                        ),
                    }
                )

                break

            duration = preferred_piece_duration(
                remaining=remaining,
                available=candidate["duration"],
                music_role=parent["music_role"],
            )

            if duration <= EPSILON:
                break

            source_start, source_end = make_source_window(
                candidate,
                duration,
            )

            part = {
                "parent_shot_id": parent["shot_id"],
                "part_index": len(parent_parts) + 1,
                "timeline": {
                    "start": cursor,
                    "end": cursor + duration,
                    "duration": duration,
                },
                "music": parent["music"],
                "reference": parent["reference"],
                "director": parent["director"],
                "source": {
                    "video": candidate["video"],
                    "scene_id": candidate["scene_id"],
                    "source_role": candidate.get(
                        "source_role",
                        "unknown",
                    ),
                    "source_start": source_start,
                    "source_end": source_end,
                    "duration": duration,
                },
                "temporal": {
                    "method": (
                        "alternative_source_fill"
                    ),
                    "loop_required": False,
                    "candidate_score": candidate.get(
                        "selection_score"
                    ),
                },
            }

            parent_parts.append(part)

            candidate_key = candidate["key"]

            usage[candidate_key] += 1
            parent_used.add(candidate_key)

            remaining -= duration
            cursor += duration

        # -------------------------------------------------------------
        # Normalize final timing so the parent shot ends exactly where
        # the original timeline said it should end.
        # -------------------------------------------------------------

        if parent_parts:

            final_part = parent_parts[-1]

            correction = (
                parent["end"]
                - final_part["timeline"]["end"]
            )

            if abs(correction) > 1e-6:
                final_part["timeline"]["end"] += correction
                final_part["timeline"]["duration"] += correction

                final_source_start = final_part[
                    "source"
                ]["source_start"]

                final_part["source"]["source_end"] = (
                    final_source_start
                    + final_part["timeline"]["duration"]
                )

        if len(parent_parts) > 1:
            rewritten_parent_count += 1

        for part in parent_parts:

            part["timeline"]["start"] = round(
                part["timeline"]["start"],
                4,
            )

            part["timeline"]["end"] = round(
                part["timeline"]["end"],
                4,
            )

            part["timeline"]["duration"] = round(
                part["timeline"]["duration"],
                4,
            )

            part["source"]["source_start"] = round(
                part["source"]["source_start"],
                4,
            )

            part["source"]["source_end"] = round(
                part["source"]["source_end"],
                4,
            )

            part["source"]["duration"] = round(
                part["source"]["duration"],
                4,
            )

            output_shots.append(part)

            total_new_duration += part[
                "timeline"
            ]["duration"]

            previous_scene = scene_key(
                part["source"].get("video"),
                part["source"].get("scene_id"),
            )

    source_scene_counts = Counter(
        scene_key(
            shot["source"].get("video"),
            shot["source"].get("scene_id"),
        )
        for shot in output_shots
    )

    loop_required = [
        shot
        for shot in output_shots
        if shot["temporal"].get("loop_required")
    ]

    summary = {
        "original_shots": len(parents),
        "new_shots": len(output_shots),
        "rewritten_parent_shots": rewritten_parent_count,
        "original_duration": round(
            total_original_duration,
            3,
        ),
        "new_duration": round(
            total_new_duration,
            3,
        ),
        "duration_difference": round(
            total_new_duration
            - total_original_duration,
            3,
        ),
        "loop_required": len(loop_required),
        "unresolved_parent_shots": len(
            unresolved
        ),
        "unique_source_scenes": len(
            source_scene_counts
        ),
        "maximum_scene_reuse": (
            max(
                source_scene_counts.values()
            )
            if source_scene_counts
            else 0
        ),
    }

    temporal_plan = {
        "meta": {
            "version": "2.0",
            "base_plan": (
                "creative_director_plan_v3.json"
            ),
            "purpose": (
                "Rewrite temporal shot structure so "
                "source media is never looped while "
                "preserving the original music timeline."
            ),
        },

        "summary": summary,

        "rules": {
            "preserve_total_duration": True,
            "never_loop_source_frames": True,
            "prefer_original_source_window": True,
            "use_alternative_sources_for_missing_duration": True,
            "avoid_consecutive_scene_reuse": True,
            "prefer_underused_source_scenes": True,
            "keep_parent_music_alignment": True,
        },

        "shot_plan": output_shots,

        "unresolved": unresolved,
    }

    diagnostics = {
        "meta": {
            "version": "2.0",
        },

        "summary": summary,

        "scene_usage": [
            {
                "scene_key": key,
                "count": count,
            }
            for key, count
            in source_scene_counts.most_common()
        ],

        "rewritten_parents": sorted(
            {
                shot["parent_shot_id"]
                for shot in output_shots
                if shot["part_index"] > 1
            }
        ),

        "unresolved": unresolved,
    }

    return temporal_plan, diagnostics


def save_json(
    data: dict,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Temporal Editor V2"
    )

    parser.add_argument(
        "--creative-plan",
        default=(
            "data/outputs/"
            "creative_director_plan_v3.json"
        ),
    )

    parser.add_argument(
        "--plan-output",
        default=(
            "data/outputs/"
            "temporal_edit_plan_v2.json"
        ),
    )

    parser.add_argument(
        "--diagnostics-output",
        default=(
            "data/outputs/"
            "temporal_edit_diagnostics_v2.json"
        ),
    )

    args = parser.parse_args()

    creative_plan = load_json(
        Path(args.creative_plan)
    )

    temporal_plan, diagnostics = build_temporal_plan(
        creative_plan
    )

    save_json(
        temporal_plan,
        Path(args.plan_output)
    )

    save_json(
        diagnostics,
        Path(args.diagnostics_output)
    )

    summary = temporal_plan["summary"]

    print()
    print("====================================")
    print("TEMPORAL EDITOR V2")
    print("====================================")
    print(
        f"Original shots       : "
        f"{summary['original_shots']}"
    )
    print(
        f"New shots            : "
        f"{summary['new_shots']}"
    )
    print(
        f"Rewritten parents    : "
        f"{summary['rewritten_parent_shots']}"
    )
    print(
        f"Original duration    : "
        f"{summary['original_duration']:.3f}s"
    )
    print(
        f"New duration         : "
        f"{summary['new_duration']:.3f}s"
    )
    print(
        f"Duration difference  : "
        f"{summary['duration_difference']:.3f}s"
    )
    print(
        f"Loop-required shots  : "
        f"{summary['loop_required']}"
    )
    print(
        f"Unresolved parents   : "
        f"{summary['unresolved_parent_shots']}"
    )
    print(
        f"Unique source scenes : "
        f"{summary['unique_source_scenes']}"
    )
    print(
        f"Maximum scene reuse  : "
        f"{summary['maximum_scene_reuse']}"
    )

    print()
    print(
        f"✅ Plan: {args.plan_output}"
    )

    print(
        f"✅ Diagnostics: "
        f"{args.diagnostics_output}"
    )

    if diagnostics["unresolved"]:
        print()
        print("⚠️ UNRESOLVED")
        for item in diagnostics["unresolved"]:
            print(
                f"  Shot {item['parent_shot_id']}: "
                f"{item['remaining_duration']:.3f}s"
            )

    print("====================================")


if __name__ == "__main__":
    main()