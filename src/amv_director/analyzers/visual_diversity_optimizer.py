from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

PLAN_FILE = (
    ROOT
    / "data"
    / "outputs"
    / "director_edit_plan_v9.json"
)

TEMPORAL_FILE = (
    ROOT
    / "data"
    / "outputs"
    / "clip_analysis"
    / "temporal_window_analysis.json"
)

AUDIT_FILE = (
    ROOT
    / "data"
    / "outputs"
    / "visual_diversity_audit.json"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "outputs"
    / "director_edit_plan_v10.json"
)

MIN_QUALITY = 0.30

# Don't replace a good shot with something dramatically worse.
QUALITY_DROP_ALLOWANCE = 0.08

# Hard duplicates from the critic.
DUPLICATE_THRESHOLD = 0.97


# ============================================================
# LOAD
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# TEMPORAL SCENES
# ============================================================

def collect_scenes(data):

    scenes = []

    for video in data.get("videos", []):

        video_name = video["video"]

        for scene in video.get("scenes", []):

            windows = []

            for key, window in scene.get(
                "best_windows",
                {},
            ).items():

                try:
                    duration = float(
                        key.replace("s", "")
                    )
                except Exception:
                    duration = float(
                        window.get(
                            "duration",
                            0.0,
                        )
                    )

                quality = float(
                    window.get(
                        "quality_score",
                        0.0,
                    )
                )

                reason = str(
                    window.get(
                        "reason",
                        "unknown",
                    )
                ).lower()

                windows.append(
                    {
                        "duration": duration,
                        "start": float(
                            window["start"]
                        ),
                        "end": float(
                            window["end"]
                        ),
                        "quality": quality,
                        "usable": bool(
                            window.get(
                                "usable",
                                False,
                            )
                        ),
                        "reason": reason,
                    }
                )

            windows.sort(
                key=lambda x: x["duration"]
            )

            usable = [
                w for w in windows
                if (
                    w["quality"] >= MIN_QUALITY
                    and w["reason"]
                    not in {
                        "mostly_black",
                        "black",
                    }
                )
            ]

            best_quality = max(
                (
                    w["quality"]
                    for w in usable
                ),
                default=0.0,
            )

            scenes.append(
                {
                    "video": video_name,
                    "scene_id": int(
                        scene["scene_id"]
                    ),
                    "start": float(
                        scene["start"]
                    ),
                    "end": float(
                        scene["end"]
                    ),
                    "duration": float(
                        scene["duration"]
                    ),
                    "windows": windows,
                    "best_quality": best_quality,
                }
            )

    return scenes


# ============================================================
# WINDOW SELECTION
# ============================================================

def crop_window(
    start,
    end,
    duration,
):

    available = end - start

    if available <= duration:
        return start, end

    extra = available - duration

    new_start = (
        start
        + extra / 2.0
    )

    return (
        new_start,
        new_start + duration,
    )


def get_window(
    scene,
    requested_duration,
    minimum_quality,
):

    # --------------------------------------------------------
    # Prefer a measured window that is long enough.
    # --------------------------------------------------------

    fitting = [
        w
        for w in scene["windows"]
        if (
            w["duration"]
            >= requested_duration
            and w["quality"]
            >= minimum_quality
            and w["reason"]
            not in {
                "mostly_black",
                "black",
            }
        )
    ]

    if fitting:

        # Prefer the smallest sufficient window,
        # then highest quality.
        fitting.sort(
            key=lambda w: (
                w["duration"],
                -w["quality"],
            )
        )

        w = fitting[0]

        start, end = crop_window(
            w["start"],
            w["end"],
            requested_duration,
        )

        return {
            "start": start,
            "end": end,
            "quality": w["quality"],
            "mode": "measured_window",
        }

    # --------------------------------------------------------
    # Try expanding a strong window for longer takes.
    # --------------------------------------------------------

    strong = [
        w
        for w in scene["windows"]
        if (
            w["quality"]
            >= minimum_quality
            and w["reason"]
            not in {
                "mostly_black",
                "black",
            }
        )
    ]

    if (
        strong
        and requested_duration
        <= scene["duration"]
    ):

        core = max(
            strong,
            key=lambda w:
                w["quality"],
        )

        center = (
            core["start"]
            + core["end"]
        ) / 2.0

        start = (
            center
            - requested_duration / 2.0
        )

        end = (
            start
            + requested_duration
        )

        if start < scene["start"]:
            start = scene["start"]
            end = (
                start
                + requested_duration
            )

        if end > scene["end"]:
            end = scene["end"]
            start = (
                end
                - requested_duration
            )

        if (
            end - start
            >= requested_duration - 1e-6
        ):

            estimated_quality = (
                core["quality"] * 0.65
                + scene["best_quality"] * 0.35
            )

            if estimated_quality >= minimum_quality:

                return {
                    "start": start,
                    "end": end,
                    "quality":
                        estimated_quality,
                    "mode":
                        "expanded_window",
                }

    return None


# ============================================================
# FLATTEN V9
# ============================================================

def flatten_plan(plan):

    records = []

    for interval in plan["intervals"]:

        cursor = float(
            interval["reference_start"]
        )

        for clip_index, clip in enumerate(
            interval["clips"]
        ):

            duration = float(
                clip["duration"]
            )

            records.append(
                {
                    "interval_id":
                        interval["interval_id"],

                    "style":
                        interval.get(
                            "style",
                            "medium",
                        ),

                    "clip_index":
                        clip_index,

                    "target_start":
                        cursor,

                    "target_end":
                        cursor + duration,

                    "clip":
                        clip,
                }
            )

            cursor += duration

    return records


# ============================================================
# HARD-DUPLICATE GROUPS
# ============================================================

def build_duplicate_groups(
    audit,
):

    pairs = [
        p
        for p in audit.get(
            "top_pairs",
            [],
        )
        if float(
            p["similarity"]
        ) >= DUPLICATE_THRESHOLD
    ]

    # Build graph.
    graph = {}

    for pair in pairs:

        a = int(pair["clip_a"])
        b = int(pair["clip_b"])

        graph.setdefault(
            a,
            set(),
        ).add(b)

        graph.setdefault(
            b,
            set(),
        ).add(a)

    # Connected components.
    groups = []
    seen = set()

    for node in graph:

        if node in seen:
            continue

        stack = [node]
        component = set()

        while stack:

            current = stack.pop()

            if current in component:
                continue

            component.add(current)
            seen.add(current)

            for neighbor in graph.get(
                current,
                set(),
            ):
                if neighbor not in component:
                    stack.append(neighbor)

        groups.append(
            sorted(component)
        )

    return pairs, groups


# ============================================================
# PLAN USAGE
# ============================================================

def build_usage(records):

    usage = {}

    for record in records:

        clip = record["clip"]

        key = (
            clip["source_video"],
            clip["scene_id"],
        )

        usage[key] = (
            usage.get(key, 0)
            + 1
        )

    return usage


# ============================================================
# REPLACEMENT
# ============================================================

def find_replacement(
    target_record,
    records,
    scenes,
    current_usage,
    replaced_indices,
):

    clip = target_record["clip"]

    duration = float(
        clip["duration"]
    )

    current_quality = float(
        clip.get(
            "quality_score",
            0.0,
        )
    )

    minimum_quality = max(
        MIN_QUALITY,
        current_quality
        - QUALITY_DROP_ALLOWANCE,
    )

    current_key = (
        clip["source_video"],
        clip["scene_id"],
    )

    candidates = []

    # Existing scene keys in the current plan.
    used_keys = {
        (
            r["clip"]["source_video"],
            r["clip"]["scene_id"],
        )
        for r in records
        if r["clip"] is not clip
    }

    for scene in scenes:

        key = (
            scene["video"],
            scene["scene_id"],
        )

        # Prefer completely unused scenes.
        is_new_scene = (
            current_usage.get(
                key,
                0,
            )
            == 0
        )

        # Never replace with exactly the same scene.
        if key == current_key:
            continue

        window = get_window(
            scene,
            duration,
            minimum_quality,
        )

        if window is None:
            continue

        # ----------------------------------------------------
        # Candidate score
        # ----------------------------------------------------

        quality = float(
            window["quality"]
        )

        quality_delta = (
            quality
            - current_quality
        )

        # New scenes receive a meaningful diversity bonus.
        diversity_bonus = (
            0.18
            if is_new_scene
            else 0.0
        )

        # Avoid heavily reused scenes.
        reuse_penalty = (
            current_usage.get(
                key,
                0,
            )
            * 0.045
        )

        # Keep replacement close to original quality.
        quality_score = (
            min(
                1.0,
                quality,
            )
        )

        score = (
            quality_score * 0.72
            + diversity_bonus
            - reuse_penalty
            + quality_delta * 0.20
        )

        candidates.append(
            {
                "scene": scene,
                "window": window,
                "score": score,
                "is_new_scene":
                    is_new_scene,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x:
            x["score"],
        reverse=True,
    )

    return candidates[0]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "🎬 V10 — VISUAL DIVERSITY OPTIMIZER"
    )
    print("=" * 80)

    plan = load_json(
        PLAN_FILE
    )

    temporal = load_json(
        TEMPORAL_FILE
    )

    audit = load_json(
        AUDIT_FILE
    )

    scenes = collect_scenes(
        temporal
    )

    records = flatten_plan(
        plan
    )

    pairs, groups = build_duplicate_groups(
        audit
    )

    usage = build_usage(
        records
    )

    print(
        f"\nV9 clips              : "
        f"{len(records)}"
    )

    print(
        f"Hard duplicate pairs  : "
        f"{len(pairs)}"
    )

    print(
        f"Duplicate groups      : "
        f"{len(groups)}"
    )

    # --------------------------------------------------------
    # Decide which clip to replace in each duplicate group.
    #
    # Keep the highest-quality member whenever possible.
    # --------------------------------------------------------

    replacement_targets = set()

    for group in groups:

        candidates = []

        for clip_number in group:

            record = records[
                clip_number - 1
            ]

            q = float(
                record["clip"].get(
                    "quality_score",
                    0.0,
                )
            )

            candidates.append(
                (
                    q,
                    clip_number,
                )
            )

        # Keep strongest shot, replace the rest.
        candidates.sort(
            reverse=True
        )

        for _, clip_number in candidates[1:]:

            replacement_targets.add(
                clip_number
            )

    print(
        f"Initial replacements    : "
        f"{len(replacement_targets)}"
    )

    replaced = []
    failed = []

    # --------------------------------------------------------
    # Replace hard duplicates.
    # --------------------------------------------------------

    for clip_number in sorted(
        replacement_targets,
        reverse=True,
    ):

        record = records[
            clip_number - 1
        ]

        old_clip = record[
            "clip"
        ]

        replacement = find_replacement(
            record,
            records,
            scenes,
            usage,
            replacement_targets,
        )

        if replacement is None:

            failed.append(
                clip_number
            )

            continue

        scene = replacement[
            "scene"
        ]

        window = replacement[
            "window"
        ]

        old_key = (
            old_clip["source_video"],
            old_clip["scene_id"],
        )

        new_key = (
            scene["video"],
            scene["scene_id"],
        )

        # Update usage.
        usage[old_key] = max(
            0,
            usage.get(
                old_key,
                0,
            ) - 1,
        )

        usage[new_key] = (
            usage.get(
                new_key,
                0,
            )
            + 1
        )

        # Replace source selection only.
        old_clip["source_video"] = (
            scene["video"]
        )

        old_clip["scene_id"] = (
            scene["scene_id"]
        )

        old_clip["source_start"] = (
            round(
                window["start"],
                6,
            )
        )

        old_clip["source_end"] = (
            round(
                window["start"]
                + float(
                    old_clip["duration"]
                ),
                6,
            )
        )

        old_clip["quality_score"] = (
            round(
                window["quality"],
                4,
            )
        )

        old_clip["selection_mode"] = (
            "v10_diversity_replacement"
        )

        replaced.append(
            {
                "clip_number":
                    clip_number,

                "target_start":
                    round(
                        record[
                            "target_start"
                        ],
                        3,
                    ),

                "old_scene":
                    old_key,

                "new_scene":
                    new_key,

                "new_source_start":
                    round(
                        window["start"],
                        3,
                    ),

                "new_source_end":
                    round(
                        window["start"]
                        + float(
                            old_clip[
                                "duration"
                            ]
                        ),
                        3,
                    ),

                "quality":
                    round(
                        window["quality"],
                        3,
                    ),
            }
        )

    # --------------------------------------------------------
    # Reconstruct interval data from modified records.
    # --------------------------------------------------------

    by_interval = {}

    for record in records:

        by_interval.setdefault(
            record["interval_id"],
            [],
        ).append(record)

    new_intervals = []

    for interval in plan["intervals"]:

        interval_id = interval[
            "interval_id"
        ]

        interval_records = by_interval[
            interval_id
        ]

        interval_records.sort(
            key=lambda x:
                x["target_start"]
        )

        new_clips = []

        for record in interval_records:

            new_clips.append(
                record["clip"]
            )

        interval_copy = dict(
            interval
        )

        interval_copy["clips"] = (
            new_clips
        )

        interval_copy["planned_duration"] = (
            round(
                sum(
                    float(
                        c["duration"]
                    )
                    for c in new_clips
                ),
                6,
            )
        )

        interval_copy["complete"] = (
            abs(
                interval_copy[
                    "planned_duration"
                ]
                - float(
                    interval_copy[
                        "reference_duration"
                    ]
                )
            )
            <= 0.01
        )

        new_intervals.append(
            interval_copy
        )

    # --------------------------------------------------------
    # Final metrics.
    # --------------------------------------------------------

    all_clips = [
        c
        for interval in new_intervals
        for c in interval["clips"]
    ]

    total_duration = sum(
        float(
            c["duration"]
        )
        for c in all_clips
    )

    unique_scenes = len(
        {
            (
                c["source_video"],
                c["scene_id"],
            )
            for c in all_clips
        }
    )

    average_quality = (
        sum(
            float(
                c.get(
                    "quality_score",
                    0.0,
                )
            )
            for c in all_clips
        )
        / len(all_clips)
        if all_clips
        else 0.0
    )

    low_quality = sum(
        1
        for c in all_clips
        if float(
            c.get(
                "quality_score",
                0.0,
            )
        ) < MIN_QUALITY
    )

    output = dict(
        plan
    )

    output["planner_version"] = (
        "V10"
    )

    output["reference_duration"] = (
        round(
            plan[
                "reference_duration"
            ],
            6,
        )
    )

    output["actual_source_duration"] = (
        round(
            total_duration,
            6,
        )
    )

    output["missing_duration"] = (
        round(
            max(
                0.0,
                plan[
                    "reference_duration"
                ]
                - total_duration,
            ),
            6,
        )
    )

    output["unique_source_scenes"] = (
        unique_scenes
    )

    output["reused_scene_count"] = sum(
        1
        for count in build_usage(
            records
        ).values()
        if count > 1
    )

    output["average_quality"] = (
        round(
            average_quality,
            4,
        )
    )

    output["low_quality_clip_count"] = (
        low_quality
    )

    output["v10_replacements"] = (
        replaced
    )

    output["v10_failed_replacements"] = (
        failed
    )

    output["intervals"] = (
        new_intervals
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
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
    print("=" * 80)
    print("✅ V10 OPTIMIZATION COMPLETE")
    print("=" * 80)

    print(
        f"Reference duration : "
        f"{output['reference_duration']:.3f}s"
    )

    print(
        f"Actual duration    : "
        f"{output['actual_source_duration']:.3f}s"
    )

    print(
        f"Missing            : "
        f"{output['missing_duration']:.3f}s"
    )

    print(
        f"Clips              : "
        f"{output['clip_count']}"
    )

    print(
        f"Unique scenes      : "
        f"{output['unique_source_scenes']}"
    )

    print(
        f"Average quality    : "
        f"{output['average_quality']:.3f}"
    )

    print(
        f"Low-quality clips  : "
        f"{output['low_quality_clip_count']}"
    )

    print(
        f"Replaced           : "
        f"{len(replaced)}"
    )

    print(
        f"Failed replacements: "
        f"{len(failed)}"
    )

    if replaced:

        print()
        print(
            "🔄 REPLACEMENTS"
        )

        for item in replaced:

            print(
                f"  #{item['clip_number']:02d} "
                f"at "
                f"{item['target_start']:.3f}s | "
                f"{item['old_scene'][0]} "
                f"S{item['old_scene'][1]:02d} "
                f"→ "
                f"{item['new_scene'][0]} "
                f"S{item['new_scene'][1]:02d} "
                f"q={item['quality']:.3f}"
            )

    print()
    print("Saved:")
    print(OUTPUT_FILE)
    print("=" * 80)


if __name__ == "__main__":
    main()
