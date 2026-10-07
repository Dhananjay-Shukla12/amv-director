from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROLE_CAPS = {
    "impact_zone": 0.90,
    "accent_zone": 0.90,
    "action_zone": 1.10,
    "build_zone": 1.40,
    "cinematic_zone": 2.20,
}

DEFAULT_CAP = 1.10


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_source_window(source: dict) -> tuple[float | None, float | None]:
    """
    Extract the selected source window from the Creative Director schema.

    V3 normally exposes selected_window, but we support several reasonable
    representations and fall back to original_start/original_end.
    """

    window = source.get("selected_window")

    # {"start": ..., "end": ...}
    if isinstance(window, dict):
        start = None
        end = None

        for key in (
            "start",
            "source_start",
            "window_start",
            "original_start",
        ):
            value = number(window.get(key))
            if value is not None:
                start = value
                break

        for key in (
            "end",
            "source_end",
            "window_end",
            "original_end",
        ):
            value = number(window.get(key))
            if value is not None:
                end = value
                break

        if start is not None and end is not None:
            return start, end

    # [start, end]
    if isinstance(window, list) and len(window) >= 2:
        start = number(window[0])
        end = number(window[1])

        if start is not None and end is not None:
            return start, end

    # Fallback to the original source scene window.
    start = number(source.get("original_start"))
    end = number(source.get("original_end"))

    return start, end


def extract_shots(plan: dict) -> list[dict]:
    shot_plan = plan.get("shot_plan")

    if not isinstance(shot_plan, list):
        raise ValueError(
            "creative_director_plan_v3.json does not contain "
            "a valid 'shot_plan' list."
        )

    shots = []

    for shot in shot_plan:
        if not isinstance(shot, dict):
            continue

        timeline = shot.get("timeline", {})
        music = shot.get("music", {})
        director = shot.get("director", {})
        source = shot.get("source", {})

        if not isinstance(timeline, dict):
            timeline = {}

        if not isinstance(music, dict):
            music = {}

        if not isinstance(director, dict):
            director = {}

        if not isinstance(source, dict):
            source = {}

        target_duration = number(
            timeline.get("duration")
        )

        if target_duration is None:
            start = number(timeline.get("start")) or 0.0
            end = number(timeline.get("end")) or start
            target_duration = max(0.0, end - start)

        source_start, source_end = extract_source_window(source)

        source_span = 0.0

        if source_start is not None and source_end is not None:
            source_span = max(
                0.0,
                source_end - source_start,
            )

        music_role = music.get("role") or "unknown"
        visual_role = director.get("visual_role") or "unknown"

        scene_key = (
            f"{source.get('video', 'unknown')}::"
            f"{source.get('scene_id', 'unknown')}"
        )

        shots.append(
            {
                "shot_id": shot.get("shot_id"),
                "timeline_start": number(
                    timeline.get("start")
                ) or 0.0,
                "timeline_end": number(
                    timeline.get("end")
                ) or 0.0,
                "target_duration": round(
                    target_duration,
                    3,
                ),
                "music_role": music_role,
                "visual_role": visual_role,
                "scene_key": scene_key,
                "video": source.get("video"),
                "scene_id": source.get("scene_id"),
                "source_start": source_start,
                "source_end": source_end,
                "source_span": round(
                    source_span,
                    3,
                ),
                "selection_score": source.get(
                    "selection_score"
                ),
            }
        )

    return shots


def evaluate_shot(
    shot: dict,
    previous_scene: str | None,
) -> dict:

    target = shot["target_duration"]
    source_span = shot["source_span"]

    music_role = shot["music_role"]
    visual_role = shot["visual_role"]

    role_cap = ROLE_CAPS.get(
        music_role,
        DEFAULT_CAP,
    )

    issues = []

    if target <= 0:
        issues.append("invalid_duration")

    if target > role_cap:
        issues.append("role_duration_exceeded")

    if source_span <= 0:
        issues.append("missing_source_window")

    else:
        if target > source_span + 0.03:
            issues.append("source_loop_required")

        elif target > source_span * 0.90:
            issues.append("source_window_near_limit")

    if previous_scene and shot["scene_key"] == previous_scene:
        issues.append("consecutive_scene_reuse")

    # We don't want a 5-second shot in a high-paced editorial zone
    # unless the source and musical structure genuinely justify it.
    if target >= 2.5 and music_role != "cinematic_zone":
        issues.append("very_long_editorial_hold")

    recommended_duration = target

    # First priority: don't loop source material.
    if source_span > 0:
        recommended_duration = min(
            recommended_duration,
            source_span,
        )

    # Second priority: respect editorial-role duration.
    recommended_duration = min(
        recommended_duration,
        role_cap,
    )

    recommended_duration = round(
        max(0.0, recommended_duration),
        3,
    )

    if not issues:
        status = "healthy"
    elif "source_loop_required" in issues:
        status = "critical"
    elif "consecutive_scene_reuse" in issues:
        status = "warning"
    else:
        status = "adjust"

    return {
        "status": status,
        "issues": issues,
        "role_cap": role_cap,
        "recommended_duration": recommended_duration,
        "duration_reduction": round(
            max(
                0.0,
                target - recommended_duration,
            ),
            3,
        ),
    }


def analyze(shots: list[dict]) -> dict:

    scene_counts = Counter(
        shot["scene_key"]
        for shot in shots
    )

    previous_scene = None
    analyzed = []

    for shot in shots:

        evaluation = evaluate_shot(
            shot,
            previous_scene,
        )

        analyzed_shot = {
            **shot,
            **evaluation,
        }

        analyzed.append(
            analyzed_shot
        )

        previous_scene = shot["scene_key"]

    healthy = [
        shot
        for shot in analyzed
        if shot["status"] == "healthy"
    ]

    adjust = [
        shot
        for shot in analyzed
        if shot["status"] == "adjust"
    ]

    warnings = [
        shot
        for shot in analyzed
        if shot["status"] == "warning"
    ]

    critical = [
        shot
        for shot in analyzed
        if shot["status"] == "critical"
    ]

    repeated_scenes = [
        {
            "scene_key": scene,
            "count": count,
        }
        for scene, count in scene_counts.most_common()
        if count >= 2
    ]

    return {
        "meta": {
            "version": "1.1",
            "source_plan": "creative_director_plan_v3.json",
            "purpose": (
                "Diagnose temporal problems in the 47-shot "
                "Creative Director plan before rendering."
            ),
        },

        "summary": {
            "total_shots": len(analyzed),
            "healthy": len(healthy),
            "adjust": len(adjust),
            "warnings": len(warnings),
            "critical": len(critical),
            "unique_scenes": len(scene_counts),
            "maximum_scene_reuse": (
                max(scene_counts.values())
                if scene_counts
                else 0
            ),
            "total_duration": round(
                sum(
                    shot["target_duration"]
                    for shot in analyzed
                ),
                3,
            ),
            "recommended_duration": round(
                sum(
                    shot["recommended_duration"]
                    for shot in analyzed
                ),
                3,
            ),
        },

        "shots": analyzed,

        "repeated_scenes": repeated_scenes,

        "critical_shots": [
            shot["shot_id"]
            for shot in critical
        ],

        "recommended_changes": [
            {
                "shot_id": shot["shot_id"],
                "target_duration": shot[
                    "target_duration"
                ],
                "recommended_duration": shot[
                    "recommended_duration"
                ],
                "duration_reduction": shot[
                    "duration_reduction"
                ],
                "issues": shot["issues"],
            }
            for shot in analyzed
            if shot["duration_reduction"] > 0
        ],
    }


def save_json(data: dict, path: Path) -> None:

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


def print_report(data: dict) -> None:

    summary = data["summary"]

    print()
    print("================================")
    print("TEMPORAL EDITOR V1.1")
    print("================================")

    print(
        f"Total shots          : "
        f"{summary['total_shots']}"
    )

    print(
        f"Healthy              : "
        f"{summary['healthy']}"
    )

    print(
        f"Need adjustment      : "
        f"{summary['adjust']}"
    )

    print(
        f"Warnings             : "
        f"{summary['warnings']}"
    )

    print(
        f"Critical             : "
        f"{summary['critical']}"
    )

    print(
        f"Unique scenes        : "
        f"{summary['unique_scenes']}"
    )

    print(
        f"Maximum scene reuse  : "
        f"{summary['maximum_scene_reuse']}"
    )

    print(
        f"Original duration    : "
        f"{summary['total_duration']:.3f}s"
    )

    print(
        f"Recommended duration : "
        f"{summary['recommended_duration']:.3f}s"
    )

    print()
    print("CRITICAL SHOTS")
    print("----------------")

    for shot in data["shots"]:
        if shot["status"] != "critical":
            continue

        print(
            f"Shot {shot['shot_id']:>2} | "
            f"target={shot['target_duration']:.3f}s | "
            f"source={shot['source_span']:.3f}s | "
            f"recommended={shot['recommended_duration']:.3f}s | "
            f"{', '.join(shot['issues'])}"
        )

    print()
    print("LONG / ADJUSTMENT SHOTS")
    print("-----------------------")

    for shot in data["recommended_changes"][:20]:
        print(
            f"Shot {shot['shot_id']:>2} | "
            f"{shot['target_duration']:.3f}s → "
            f"{shot['recommended_duration']:.3f}s | "
            f"{', '.join(shot['issues'])}"
        )

    print()
    print("REPEATED SCENES")
    print("----------------")

    for item in data["repeated_scenes"][:15]:
        print(
            f"{item['scene_key']} "
            f"→ {item['count']} uses"
        )

    print()
    print("================================")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Temporal Editor V1.1"
    )

    parser.add_argument(
        "--creative-plan",
        default=(
            "data/outputs/"
            "creative_director_plan_v3.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/outputs/"
            "temporal_edit_diagnostics_v1.json"
        ),
    )

    args = parser.parse_args()

    plan_path = Path(
        args.creative_plan
    )

    output_path = Path(
        args.output
    )

    plan = load_json(
        plan_path
    )

    shots = extract_shots(
        plan
    )

    if not shots:
        raise RuntimeError(
            "No shots found in creative director plan."
        )

    analysis = analyze(
        shots
    )

    save_json(
        analysis,
        output_path,
    )

    print_report(
        analysis
    )

    print()
    print(
        f"✅ Saved: {output_path}"
    )


if __name__ == "__main__":
    main()