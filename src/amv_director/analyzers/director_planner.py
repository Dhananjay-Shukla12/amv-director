import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

REFERENCE_SCHEDULE = ROOT / "data/outputs/reference_edit_schedule.json"
TEMPORAL_JSON = ROOT / "data/outputs/clip_analysis/temporal_window_analysis.json"

OUTPUT = ROOT / "data/outputs/director_edit_plan_v9.json"

QUALITY_THRESHOLD = 0.30

# Controlled reuse:
# 1 = old V8 behavior
# 2 = preferred V9 maximum
# 3 = emergency capacity
MAX_SCENE_USES = 3


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_first(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


# ============================================================
# REFERENCE INTERVALS
# ============================================================

def normalize_interval(item: dict, idx: int):

    start = get_first(
        item,
        [
            "reference_start",
            "start",
            "target_start",
            "edit_start",
        ],
    )

    end = get_first(
        item,
        [
            "reference_end",
            "end",
            "target_end",
            "edit_end",
        ],
    )

    if start is None or end is None:
        return None

    start = float(start)
    end = float(end)

    if end <= start:
        return None

    duration = float(
        get_first(
            item,
            [
                "duration",
                "reference_duration",
                "target_duration",
            ],
            end - start,
        )
    )

    style = str(
        get_first(
            item,
            [
                "style",
                "rhythm_class",
                "tempo_class",
                "motion_class",
                "type",
            ],
            "medium",
        )
    ).lower()

    return {
        "interval_id": idx,
        "reference_start": start,
        "reference_end": end,
        "reference_duration": duration,
        "style": style,
    }


def find_intervals(obj: Any):

    if isinstance(obj, dict):

        preferred = [
            "intervals",
            "final_intervals",
            "edit_intervals",
            "schedule",
        ]

        for key in preferred:

            value = obj.get(key)

            if isinstance(value, list):

                result = []

                for i, item in enumerate(value, 1):

                    if isinstance(item, dict):

                        normalized = normalize_interval(
                            item,
                            i,
                        )

                        if normalized:
                            result.append(normalized)

                if result:
                    return result

        for value in obj.values():

            result = find_intervals(value)

            if result:
                return result

    elif isinstance(obj, list):

        result = []

        for i, item in enumerate(obj, 1):

            if isinstance(item, dict):

                normalized = normalize_interval(
                    item,
                    i,
                )

                if normalized:
                    result.append(normalized)

        if result:
            return result

        for item in obj:

            result = find_intervals(item)

            if result:
                return result

    return []


# ============================================================
# TEMPORAL SOURCE DATA
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

                windows.append(
                    {
                        "duration": duration,
                        "start": float(
                            window["start"]
                        ),
                        "end": float(
                            window["end"]
                        ),
                        "quality": float(
                            window.get(
                                "quality_score",
                                0.0,
                            )
                        ),
                        "usable": bool(
                            window.get(
                                "usable",
                                False,
                            )
                        ),
                        "reason": str(
                            window.get(
                                "reason",
                                "unknown",
                            )
                        ),
                    }
                )

            windows.sort(
                key=lambda x: x["duration"]
            )

            best_quality = max(
                (
                    w["quality"]
                    for w in windows
                ),
                default=0.0,
            )

            longest = (
                max(
                    windows,
                    key=lambda x: x["duration"],
                )
                if windows
                else None
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
                    "coverage_quality": (
                        longest["quality"]
                        if longest
                        else 0.0
                    ),
                }
            )

    return scenes


# ============================================================
# STYLE
# ============================================================

def desired_motion(style: str):

    s = style.lower()

    if "very_fast" in s:
        return 0.90

    if "fast" in s:
        return 0.82

    if "action" in s:
        return 0.82

    if "impact" in s:
        return 0.72

    if "medium" in s:
        return 0.55

    if "slow" in s:
        return 0.28

    if "cinematic" in s:
        return 0.25

    return 0.50


# ============================================================
# WINDOW UTILITIES
# ============================================================

def overlap_ratio(
    a_start,
    a_end,
    b_start,
    b_end,
):

    overlap = max(
        0.0,
        min(a_end, b_end)
        - max(a_start, b_start),
    )

    a_duration = max(
        0.000001,
        a_end - a_start,
    )

    return overlap / a_duration


def crop_window(
    start,
    end,
    requested,
):

    available = end - start

    if available <= requested:
        return start, end

    extra = available - requested

    new_start = start + extra / 2.0

    return (
        new_start,
        new_start + requested,
    )


def is_good_window(window):

    quality = float(
        window["quality"]
    )

    reason = str(
        window["reason"]
    ).lower()

    return (
        quality >= QUALITY_THRESHOLD
        and reason not in {
            "mostly_black",
            "black",
        }
    )


# ============================================================
# CANDIDATE WINDOWS
# ============================================================

def generate_window_candidates(
    scene,
    remaining,
    history,
):

    """
    Generate several possible windows from the same scene.

    V8 effectively gave us one choice per scene.
    V9 gives the same scene several possible temporal choices
    and penalizes overlap with previous uses.
    """

    candidates = []

    used_windows = history.get(
        (
            scene["video"],
            scene["scene_id"],
        ),
        [],
    )

    # --------------------------------------------------------
    # Measured temporal windows
    # --------------------------------------------------------

    for window in scene["windows"]:

        available = window["duration"]

        if available <= 0:
            continue

        # A window can provide a smaller take through cropping.
        take = min(
            remaining,
            available,
        )

        source_start, source_end = crop_window(
            window["start"],
            window["end"],
            take,
        )

        quality = window["quality"]

        if not is_good_window(
            window
        ):
            continue

        max_overlap = 0.0

        for previous in used_windows:

            ov = overlap_ratio(
                source_start,
                source_end,
                previous[0],
                previous[1],
            )

            max_overlap = max(
                max_overlap,
                ov,
            )

        candidates.append(
            {
                "start": source_start,
                "end": source_end,
                "duration": take,
                "quality": quality,
                "reason": window["reason"],
                "mode": "measured_window",
                "overlap": max_overlap,
            }
        )

    # --------------------------------------------------------
    # Expanded full-scene window
    # --------------------------------------------------------

    if (
        remaining <= scene["duration"]
        and scene["coverage_quality"] >= 0.34
        and scene["best_quality"] >= 0.45
    ):

        # Strongest measured window.
        strong_windows = [
            w
            for w in scene["windows"]
            if w["quality"] >= 0.45
        ]

        if strong_windows:

            core = max(
                strong_windows,
                key=lambda x: x["quality"],
            )

            center = (
                core["start"]
                + core["end"]
            ) / 2.0

            start = center - remaining / 2.0
            end = start + remaining

            scene_start = scene["start"]
            scene_end = scene["end"]

            if start < scene_start:
                start = scene_start
                end = start + remaining

            if end > scene_end:
                end = scene_end
                start = end - remaining

            overlap = 0.0

            for previous in used_windows:

                ov = overlap_ratio(
                    start,
                    end,
                    previous[0],
                    previous[1],
                )

                overlap = max(
                    overlap,
                    ov,
                )

            estimated_quality = (
                core["quality"] * 0.65
                + scene["coverage_quality"]
                * 0.35
            )

            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "duration": remaining,
                    "quality": estimated_quality,
                    "reason":
                        "expanded_from_best_window",
                    "mode": "expanded_window",
                    "overlap": overlap,
                }
            )

    return candidates


# ============================================================
# SCENE CANDIDATE
# ============================================================

def evaluate_candidate(
    scene,
    window,
    requested,
    style,
    reuse_count,
):

    quality = float(
        window["quality"]
    )

    duration = float(
        window["duration"]
    )

    # --------------------------------------------------------
    # Prefer larger useful takes.
    # --------------------------------------------------------

    duration_fit = min(
        1.0,
        duration
        / max(requested, 0.001),
    )

    # --------------------------------------------------------
    # Reuse penalty.
    # --------------------------------------------------------

    if reuse_count == 0:
        reuse_penalty = 0.0

    elif reuse_count == 1:
        reuse_penalty = 0.10

    else:
        reuse_penalty = 0.22

    # --------------------------------------------------------
    # Temporal overlap penalty.
    # --------------------------------------------------------

    overlap_penalty = (
        window["overlap"] * 0.28
    )

    # Stronger penalty for exact repetition.
    if window["overlap"] >= 0.95:
        overlap_penalty += 0.20

    # --------------------------------------------------------
    # Slight style preference.
    # --------------------------------------------------------

    style_bonus = 0.0

    if (
        "slow" in style
        or "cinematic" in style
    ):
        if duration >= min(
            requested,
            1.5,
        ):
            style_bonus += 0.04

    if (
        "fast" in style
        or "action" in style
        or "impact" in style
    ):
        if duration <= 1.5:
            style_bonus += 0.04

    score = (
        quality * 0.70
        + duration_fit * 0.20
        + style_bonus
        - reuse_penalty
        - overlap_penalty
    )

    return score


# ============================================================
# BEST SOURCE SELECTION
# ============================================================

def choose_source(
    scenes,
    remaining,
    style,
    usage,
    history,
):

    candidates = []

    for scene in scenes:

        key = (
            scene["video"],
            scene["scene_id"],
        )

        reuse_count = usage.get(
            key,
            0,
        )

        if reuse_count >= MAX_SCENE_USES:
            continue

        windows = generate_window_candidates(
            scene,
            remaining,
            history,
        )

        for window in windows:

            score = evaluate_candidate(
                scene,
                window,
                remaining,
                style,
                reuse_count,
            )

            candidates.append(
                {
                    "scene": scene,
                    "window": window,
                    "score": score,
                }
            )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda x: x["score"],
    )


# ============================================================
# DARK TRANSITION FALLBACK
# ============================================================

def choose_transition(
    scenes,
    remaining,
    usage,
    history,
):

    """
    Black/dark footage is only allowed as a tiny transition.

    This prevents it from contaminating long cinematic/action
    sections.
    """

    if remaining > 0.25:
        return None

    candidates = []

    for scene in scenes:

        key = (
            scene["video"],
            scene["scene_id"],
        )

        if usage.get(
            key,
            0,
        ) >= MAX_SCENE_USES:
            continue

        for window in scene["windows"]:

            if window["duration"] < remaining:
                continue

            reason = str(
                window["reason"]
            ).lower()

            if reason not in {
                "mostly_black",
                "very_dark",
            }:
                continue

            start, end = crop_window(
                window["start"],
                window["end"],
                remaining,
            )

            overlap = 0.0

            for previous in history.get(
                key,
                [],
            ):

                overlap = max(
                    overlap,
                    overlap_ratio(
                        start,
                        end,
                        previous[0],
                        previous[1],
                    ),
                )

            candidates.append(
                {
                    "scene": scene,
                    "start": start,
                    "end": end,
                    "quality": window["quality"],
                    "overlap": overlap,
                }
            )

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda x: (
            x["overlap"],
            -x["quality"],
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 84)
    print(
        "🎬 DIRECTOR V9 — REUSE-AWARE TEMPORAL PLANNER"
    )
    print("=" * 84)

    reference = load_json(
        REFERENCE_SCHEDULE
    )

    temporal = load_json(
        TEMPORAL_JSON
    )

    intervals = find_intervals(
        reference
    )

    scenes = collect_scenes(
        temporal
    )

    if not intervals:
        raise RuntimeError(
            "Reference intervals not found."
        )

    if not scenes:
        raise RuntimeError(
            "Temporal source data not found."
        )

    reference_duration = max(
        x["reference_end"]
        for x in intervals
    )

    print(
        f"\nReference intervals : "
        f"{len(intervals)}"
    )

    print(
        f"Reference duration  : "
        f"{reference_duration:.3f}s"
    )

    print(
        f"Source scenes       : "
        f"{len(scenes)}"
    )

    # --------------------------------------------------------
    # Long intervals first.
    # --------------------------------------------------------

    planning_order = sorted(
        intervals,
        key=lambda x: (
            -x["reference_duration"],
            x["reference_start"],
        ),
    )

    usage = {}
    history = {}

    planned = []

    fallback_transitions = 0

    # ========================================================
    # PLAN
    # ========================================================

    for interval in planning_order:

        remaining = interval[
            "reference_duration"
        ]

        interval_clips = []

        while remaining > 0.0005:

            requested = min(
                remaining,
                7.0,
            )

            candidate = choose_source(
                scenes,
                requested,
                interval["style"],
                usage,
                history,
            )

            # ------------------------------------------------
            # If nothing can provide requested duration,
            # progressively request a smaller take.
            # ------------------------------------------------

            if candidate is None:

                attempts = [
                    5.0,
                    4.0,
                    3.0,
                    2.0,
                    1.5,
                    1.25,
                    1.0,
                    0.75,
                    0.5,
                    0.4,
                ]

                for smaller in attempts:

                    if smaller >= requested:
                        continue

                    candidate = choose_source(
                        scenes,
                        smaller,
                        interval["style"],
                        usage,
                        history,
                    )

                    if candidate:
                        requested = smaller
                        break

            # ------------------------------------------------
            # Tiny transition fallback.
            # ------------------------------------------------

            if candidate is None:

                transition = choose_transition(
                    scenes,
                    remaining,
                    usage,
                    history,
                )

                if transition:

                    scene = transition[
                        "scene"
                    ]

                    key = (
                        scene["video"],
                        scene["scene_id"],
                    )

                    start = transition[
                        "start"
                    ]

                    end = transition[
                        "end"
                    ]

                    duration = end - start

                    clip = {
                        "source_video":
                            scene["video"],

                        "scene_id":
                            scene["scene_id"],

                        "source_start":
                            round(start, 6),

                        "source_end":
                            round(end, 6),

                        "duration":
                            round(duration, 6),

                        "quality_score":
                            round(
                                transition[
                                    "quality"
                                ],
                                4,
                            ),

                        "selection_mode":
                            "dark_transition",
                    }

                    interval_clips.append(
                        clip
                    )

                    usage[key] = (
                        usage.get(
                            key,
                            0,
                        ) + 1
                    )

                    history.setdefault(
                        key,
                        [],
                    ).append(
                        (
                            start,
                            end,
                        )
                    )

                    remaining -= duration
                    fallback_transitions += 1

                    continue

            # ------------------------------------------------
            # No source available.
            # ------------------------------------------------

            if candidate is None:
                break

            scene = candidate["scene"]
            window = candidate["window"]

            key = (
                scene["video"],
                scene["scene_id"],
            )

            start = float(
                window["start"]
            )

            end = float(
                window["end"]
            )

            duration = end - start

            duration = min(
                duration,
                remaining,
            )

            # Center crop when needed.
            if (
                end - start
                > duration + 1e-6
            ):

                start, end = crop_window(
                    start,
                    end,
                    duration,
                )

            duration = end - start

            if duration <= 0.000001:
                break

            clip = {
                "source_video":
                    scene["video"],

                "scene_id":
                    scene["scene_id"],

                "source_start":
                    round(
                        start,
                        6,
                    ),

                "source_end":
                    round(
                        end,
                        6,
                    ),

                "duration":
                    round(
                        duration,
                        6,
                    ),

                "quality_score":
                    round(
                        window["quality"],
                        4,
                    ),

                "selection_score":
                    round(
                        candidate["score"],
                        4,
                    ),

                "selection_mode":
                    window["mode"],
            }

            interval_clips.append(
                clip
            )

            usage[key] = (
                usage.get(
                    key,
                    0,
                ) + 1
            )

            history.setdefault(
                key,
                [],
            ).append(
                (
                    start,
                    end,
                )
            )

            remaining -= duration

        planned.append(
            {
                "interval_id":
                    interval[
                        "interval_id"
                    ],

                "reference_start":
                    round(
                        interval[
                            "reference_start"
                        ],
                        6,
                    ),

                "reference_end":
                    round(
                        interval[
                            "reference_end"
                        ],
                        6,
                    ),

                "reference_duration":
                    round(
                        interval[
                            "reference_duration"
                        ],
                        6,
                    ),

                "style":
                    interval["style"],

                "planned_duration":
                    round(
                        sum(
                            c["duration"]
                            for c
                            in interval_clips
                        ),
                        6,
                    ),

                "complete":
                    abs(
                        sum(
                            c["duration"]
                            for c
                            in interval_clips
                        )
                        - interval[
                            "reference_duration"
                        ]
                    ) <= 0.01,

                "clips":
                    interval_clips,
            }
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    planned.sort(
        key=lambda x:
            x["reference_start"]
    )

    all_clips = [
        clip
        for interval in planned
        for clip in interval["clips"]
    ]

    actual_duration = sum(
        c["duration"]
        for c in all_clips
    )

    missing = max(
        0.0,
        reference_duration
        - actual_duration,
    )

    unique_scene_count = len(
        {
            (
                c["source_video"],
                c["scene_id"],
            )
            for c in all_clips
        }
    )

    reused_scene_count = sum(
        1
        for count in usage.values()
        if count > 1
    )

    low_quality = [
        c
        for c in all_clips
        if (
            c["quality_score"]
            < QUALITY_THRESHOLD
            and c["selection_mode"]
            != "dark_transition"
        )
    ]

    average_quality = (
        sum(
            c["quality_score"]
            for c in all_clips
        )
        / len(all_clips)
        if all_clips
        else 0.0
    )

    complete_intervals = sum(
        1
        for x in planned
        if x["complete"]
    )

    output = {
        "project":
            "AMV Director",

        "planner_version":
            "V9",

        "reference_duration":
            round(
                reference_duration,
                6,
            ),

        "actual_source_duration":
            round(
                actual_duration,
                6,
            ),

        "missing_duration":
            round(
                missing,
                6,
            ),

        "interval_count":
            len(planned),

        "complete_intervals":
            complete_intervals,

        "clip_count":
            len(all_clips),

        "unique_source_scenes":
            unique_scene_count,

        "reused_scene_count":
            reused_scene_count,

        "average_quality":
            round(
                average_quality,
                4,
            ),

        "low_quality_clip_count":
            len(low_quality),

        "dark_transition_count":
            fallback_transitions,

        "scene_usage":
            {
                f"{k[0]}::S{k[1]:02d}":
                    v
                for k, v
                in usage.items()
            },

        "intervals":
            planned,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 84)
    print("✅ DIRECTOR V9 COMPLETE")
    print("=" * 84)

    print(
        f"Reference duration : "
        f"{reference_duration:.3f}s"
    )

    print(
        f"Actual source      : "
        f"{actual_duration:.3f}s"
    )

    print(
        f"Missing            : "
        f"{missing:.3f}s"
    )

    print(
        f"Intervals          : "
        f"{len(planned)}"
    )

    print(
        f"Complete intervals : "
        f"{complete_intervals}/"
        f"{len(planned)}"
    )

    print(
        f"Clips              : "
        f"{len(all_clips)}"
    )

    print(
        f"Unique scenes      : "
        f"{unique_scene_count}"
    )

    print(
        f"Reused scenes      : "
        f"{reused_scene_count}"
    )

    print(
        f"Average quality    : "
        f"{average_quality:.3f}"
    )

    print(
        f"Low quality clips  : "
        f"{len(low_quality)}"
    )

    print(
        f"Dark transitions   : "
        f"{fallback_transitions}"
    )

    if low_quality:
        print()
        print(
            "⚠️ LOW QUALITY MATERIAL:"
        )

        for clip in low_quality:
            print(
                f"   {clip['source_video']} "
                f"S{clip['scene_id']:02d} "
                f"{clip['source_start']:.3f}"
                f"→"
                f"{clip['source_end']:.3f} "
                f"q="
                f"{clip['quality_score']:.3f}"
            )

    else:
        print()
        print(
            "🎉 No ordinary low-quality "
            "clips selected."
        )

    print()
    print("Saved:")
    print(OUTPUT)

    print("=" * 84)


if __name__ == "__main__":
    main()
