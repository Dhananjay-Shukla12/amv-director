from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]

PLAN_FILE = (
    ROOT
    / "data"
    / "outputs"
    / "director_edit_plan_v10.json"
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
    / "visual_diversity_audit_v10.json"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "outputs"
    / "director_edit_plan_v11.json"
)

MIN_QUALITY = 0.30
HARD_DUPLICATE = 0.97

MAX_SCENE_USES = 2

CACHE = {}


# ============================================================
# LOAD
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frame(video_path, timestamp):

    key = (
        str(video_path),
        round(timestamp, 3),
    )

    if key in CACHE:
        return CACHE[key]

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        return None

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 30.0

    frame_no = max(
        0,
        int(round(timestamp * fps)),
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_no,
    )

    ok, frame = cap.read()

    cap.release()

    if not ok:
        return None

    CACHE[key] = frame

    return frame


# ============================================================
# VISUAL FEATURES
# ============================================================

def features(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    small_gray = cv2.resize(
        gray,
        (64, 36),
        interpolation=cv2.INTER_AREA,
    )

    # Normalized grayscale structure.
    gray_float = (
        small_gray.astype(
            np.float32
        ) / 255.0
    )

    gray_mean = float(
        gray_float.mean()
    )

    gray_std = float(
        gray_float.std()
    )

    normalized_gray = (
        (
            gray_float
            - gray_mean
        )
        /
        max(
            gray_std,
            1e-5,
        )
    )

    # HSV color histogram.
    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [18, 8],
        [0, 180, 0, 256],
    )

    hist = cv2.normalize(
        hist,
        hist,
    ).flatten()

    # Edge structure.
    edges = cv2.Canny(
        gray,
        70,
        150,
    )

    edge_small = cv2.resize(
        edges,
        (32, 18),
        interpolation=cv2.INTER_AREA,
    )

    edge_float = (
        edge_small.astype(
            np.float32
        ) / 255.0
    )

    # dHash.
    dh = cv2.resize(
        gray,
        (17, 16),
        interpolation=cv2.INTER_AREA,
    )

    dhash = (
        dh[:, 1:]
        >
        dh[:, :-1]
    ).flatten()

    return {
        "gray": normalized_gray,
        "hist": hist,
        "edges": edge_float,
        "dhash": dhash,
    }


def similarity(a, b):

    # Structural similarity approximation.
    gray_a = a["gray"].flatten()
    gray_b = b["gray"].flatten()

    corr = np.corrcoef(
        gray_a,
        gray_b,
    )[0, 1]

    if not np.isfinite(corr):
        corr = 0.0

    structural = (
        float(corr) + 1.0
    ) / 2.0

    # Histogram similarity.
    bhatta = cv2.compareHist(
        a["hist"].astype(
            np.float32
        ),
        b["hist"].astype(
            np.float32
        ),
        cv2.HISTCMP_BHATTACHARYYA,
    )

    color = max(
        0.0,
        1.0 - float(bhatta),
    )

    # Edge similarity.
    edge_distance = float(
        np.mean(
            np.abs(
                a["edges"]
                - b["edges"]
            )
        )
    )

    edge_sim = max(
        0.0,
        1.0 - edge_distance,
    )

    # dHash similarity.
    dhash_equal = np.count_nonzero(
        a["dhash"] == b["dhash"]
    )

    dhash_sim = (
        dhash_equal
        /
        max(
            len(a["dhash"]),
            1,
        )
    )

    score = (
        structural * 0.35
        +
        color * 0.25
        +
        edge_sim * 0.15
        +
        dhash_sim * 0.25
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


# ============================================================
# SOURCE SCENES
# ============================================================

def collect_scenes(data):

    scenes = []

    for video in data["videos"]:

        video_name = video["video"]

        for scene in video["scenes"]:

            windows = []

            for key, window in scene.get(
                "best_windows",
                {},
            ).items():

                try:
                    duration = float(
                        key.replace(
                            "s",
                            "",
                        )
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

                if (
                    quality >= MIN_QUALITY
                    and reason not in {
                        "mostly_black",
                        "black",
                    }
                ):

                    windows.append(
                        {
                            "duration":
                                duration,
                            "start":
                                float(
                                    window[
                                        "start"
                                    ]
                                ),
                            "end":
                                float(
                                    window[
                                        "end"
                                    ]
                                ),
                            "quality":
                                quality,
                        }
                    )

            scenes.append(
                {
                    "video":
                        video_name,
                    "scene_id":
                        int(
                            scene[
                                "scene_id"
                            ]
                        ),
                    "start":
                        float(
                            scene[
                                "start"
                            ]
                        ),
                    "end":
                        float(
                            scene[
                                "end"
                            ]
                        ),
                    "duration":
                        float(
                            scene[
                                "duration"
                            ]
                        ),
                    "windows":
                        windows,
                }
            )

    return scenes


# ============================================================
# PLAN FLATTEN
# ============================================================

def flatten(plan):

    records = []

    for interval in plan["intervals"]:

        cursor = float(
            interval[
                "reference_start"
            ]
        )

        for clip in interval["clips"]:

            duration = float(
                clip["duration"]
            )

            records.append(
                {
                    "interval_id":
                        interval[
                            "interval_id"
                        ],
                    "style":
                        interval.get(
                            "style",
                            "medium",
                        ),
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
# FRAME FOR CLIP
# ============================================================

def clip_features(
    clip,
    scenes_by_key,
):

    key = (
        clip["source_video"],
        clip["scene_id"],
    )

    timestamp = (
        float(
            clip["source_start"]
        )
        +
        float(
            clip["source_end"]
        )
    ) / 2.0

    frame = extract_frame(
        ROOT
        / "data"
        / "clips"
        / clip["source_video"],
        timestamp,
    )

    if frame is None:
        return None

    return features(frame)


# ============================================================
# CANDIDATE WINDOW
# ============================================================

def candidate_window(
    scene,
    duration,
):

    fitting = [
        w
        for w in scene["windows"]
        if w["duration"] >= duration
    ]

    if fitting:

        # Best quality sufficient window.
        fitting.sort(
            key=lambda w: (
                -w["quality"],
                w["duration"],
            )
        )

        w = fitting[0]

        extra = (
            w["duration"]
            - duration
        )

        start = (
            w["start"]
            + extra / 2.0
        )

        end = (
            start
            + duration
        )

        return {
            "start": start,
            "end": end,
            "quality": w["quality"],
        }

    # Expand strong windows if needed.
    strong = [
        w
        for w in scene["windows"]
        if w["quality"] >= 0.45
    ]

    if (
        strong
        and duration <= scene["duration"]
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
            - duration / 2.0
        )

        end = start + duration

        if start < scene["start"]:

            start = scene["start"]
            end = (
                start
                + duration
            )

        if end > scene["end"]:

            end = scene["end"]
            start = (
                end
                - duration
            )

        if (
            end - start
            >= duration - 1e-6
        ):

            quality = (
                core["quality"]
                * 0.65
                +
                max(
                    w["quality"]
                    for w in scene[
                        "windows"
                    ]
                )
                * 0.35
            )

            return {
                "start": start,
                "end": end,
                "quality": quality,
            }

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(
        "🎬 V11 — FRAME-AWARE DIVERSITY OPTIMIZER"
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

    records = flatten(
        plan
    )

    scenes_by_key = {
        (
            s["video"],
            s["scene_id"],
        ): s
        for s in scenes
    }

    # --------------------------------------------------------
    # Identify hard duplicate components.
    # --------------------------------------------------------

    graph = {}

    for pair in audit.get(
        "top_pairs",
        [],
    ):

        if float(
            pair["similarity"]
        ) < HARD_DUPLICATE:
            continue

        a = int(
            pair["clip_a"]
        )

        b = int(
            pair["clip_b"]
        )

        graph.setdefault(
            a,
            set(),
        ).add(b)

        graph.setdefault(
            b,
            set(),
        ).add(a)

    groups = []
    visited = set()

    for node in graph:

        if node in visited:
            continue

        stack = [node]
        group = set()

        while stack:

            current = stack.pop()

            if current in group:
                continue

            group.add(current)
            visited.add(current)

            for neighbor in graph.get(
                current,
                set(),
            ):

                if neighbor not in group:
                    stack.append(
                        neighbor
                    )

        groups.append(
            sorted(group)
        )

    print(
        f"\nHard duplicate groups : "
        f"{len(groups)}"
    )

    # --------------------------------------------------------
    # Initial feature extraction.
    # --------------------------------------------------------

    print(
        "Extracting current-frame features..."
    )

    feature_cache = {}

    for index, record in enumerate(
        records,
        start=1,
    ):

        feature_cache[index] = (
            clip_features(
                record["clip"],
                scenes_by_key,
            )
        )

        if index % 10 == 0:
            print(
                f"  {index}/{len(records)}"
            )

    # --------------------------------------------------------
    # Current usage.
    # --------------------------------------------------------

    usage = {}

    for record in records:

        clip = record["clip"]

        key = (
            clip["source_video"],
            clip["scene_id"],
        )

        usage[key] = (
            usage.get(
                key,
                0,
            )
            + 1
        )

    replacements = []

    # --------------------------------------------------------
    # Process groups.
    # --------------------------------------------------------

    for group in groups:

        if len(group) <= 1:
            continue

        # Keep highest quality.
        keep = max(
            group,
            key=lambda idx:
                float(
                    records[
                        idx - 1
                    ]["clip"].get(
                        "quality_score",
                        0.0,
                    )
                )
        )

        for idx in group:

            if idx == keep:
                continue

            record = records[
                idx - 1
            ]

            old_clip = record[
                "clip"
            ]

            duration = float(
                old_clip[
                    "duration"
                ]
            )

            old_quality = float(
                old_clip.get(
                    "quality_score",
                    0.0,
                )
            )

            old_key = (
                old_clip[
                    "source_video"
                ],
                old_clip[
                    "scene_id"
                ],
            )

            # ------------------------------------------------
            # Candidate scenes.
            # ------------------------------------------------

            candidates = []

            for scene in scenes:

                new_key = (
                    scene["video"],
                    scene["scene_id"],
                )

                if new_key == old_key:
                    continue

                if (
                    usage.get(
                        new_key,
                        0,
                    )
                    >= MAX_SCENE_USES
                ):
                    continue

                window = candidate_window(
                    scene,
                    duration,
                )

                if window is None:
                    continue

                quality = float(
                    window["quality"]
                )

                # Don't sacrifice too much quality.
                if (
                    quality
                    <
                    max(
                        MIN_QUALITY,
                        old_quality
                        - 0.10,
                    )
                ):
                    continue

                timestamp = (
                    window["start"]
                    +
                    window["end"]
                ) / 2.0

                frame = extract_frame(
                    ROOT
                    / "data"
                    / "clips"
                    / scene[
                        "video"
                    ],
                    timestamp,
                )

                if frame is None:
                    continue

                candidate_feature = (
                    features(frame)
                )

                # ------------------------------------------------
                # Compare against all OTHER clips.
                # ------------------------------------------------

                similarities = []

                for other_idx, other_feature in (
                    feature_cache.items()
                ):

                    if other_idx == idx:
                        continue

                    if other_feature is None:
                        continue

                    similarities.append(
                        similarity(
                            candidate_feature,
                            other_feature,
                        )
                    )

                if not similarities:
                    max_sim = 0.0
                    mean_sim = 0.0
                else:
                    max_sim = max(
                        similarities
                    )

                    mean_sim = float(
                        np.mean(
                            similarities
                        )
                    )

                # ------------------------------------------------
                # Score candidate.
                #
                # Diversity is deliberately dominant.
                # ------------------------------------------------

                diversity_score = (
                    1.0 - max_sim
                )

                mean_diversity = (
                    1.0 - mean_sim
                )

                quality_score = min(
                    1.0,
                    quality,
                )

                usage_penalty = (
                    usage.get(
                        new_key,
                        0,
                    )
                    * 0.12
                )

                score = (
                    diversity_score
                    * 0.52
                    +
                    mean_diversity
                    * 0.20
                    +
                    quality_score
                    * 0.28
                    -
                    usage_penalty
                )

                candidates.append(
                    {
                        "scene":
                            scene,
                        "window":
                            window,
                        "feature":
                            candidate_feature,
                        "max_similarity":
                            max_sim,
                        "mean_similarity":
                            mean_sim,
                        "score":
                            score,
                    }
                )

            if not candidates:
                continue

            best = max(
                candidates,
                key=lambda x:
                    x["score"],
            )

            scene = best[
                "scene"
            ]

            window = best[
                "window"
            ]

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
                )
                - 1,
            )

            usage[new_key] = (
                usage.get(
                    new_key,
                    0,
                )
                + 1
            )

            # Update the record.
            old_clip[
                "source_video"
            ] = scene[
                "video"
            ]

            old_clip[
                "scene_id"
            ] = scene[
                "scene_id"
            ]

            old_clip[
                "source_start"
            ] = round(
                window["start"],
                6,
            )

            old_clip[
                "source_end"
            ] = round(
                window["end"],
                6,
            )

            old_clip[
                "quality_score"
            ] = round(
                window["quality"],
                4,
            )

            old_clip[
                "selection_mode"
            ] = (
                "v11_frame_diversity"
            )

            feature_cache[
                idx
            ] = best[
                "feature"
            ]

            replacements.append(
                {
                    "clip":
                        idx,
                    "target":
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
                    "quality":
                        round(
                            window[
                                "quality"
                            ],
                            3,
                        ),
                    "max_similarity":
                        round(
                            best[
                                "max_similarity"
                            ],
                            3,
                        ),
                }
            )

    # --------------------------------------------------------
    # Rebuild plan.
    # --------------------------------------------------------

    new_intervals = []

    by_interval = {}

    for record in records:

        by_interval.setdefault(
            record[
                "interval_id"
            ],
            [],
        ).append(record)

    for interval in plan[
        "intervals"
    ]:

        records_for_interval = (
            by_interval[
                interval[
                    "interval_id"
                ]
            ]
        )

        records_for_interval.sort(
            key=lambda x:
                x["target_start"]
        )

        copy_interval = dict(
            interval
        )

        copy_interval[
            "clips"
        ] = [
            r["clip"]
            for r
            in records_for_interval
        ]

        copy_interval[
            "planned_duration"
        ] = round(
            sum(
                float(
                    c["duration"]
                )
                for c
                in copy_interval[
                    "clips"
                ]
            ),
            6,
        )

        copy_interval[
            "complete"
        ] = (
            abs(
                copy_interval[
                    "planned_duration"
                ]
                -
                float(
                    copy_interval[
                        "reference_duration"
                    ]
                )
            )
            <= 0.01
        )

        new_intervals.append(
            copy_interval
        )

    all_clips = [
        c
        for interval in new_intervals
        for c in interval[
            "clips"
        ]
    ]

    total_duration = sum(
        float(
            c["duration"]
        )
        for c in all_clips
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
        /
        len(all_clips)
    )

    unique_scenes = len(
        {
            (
                c[
                    "source_video"
                ],
                c[
                    "scene_id"
                ],
            )
            for c in all_clips
        }
    )

    low_quality = sum(
        1
        for c in all_clips
        if float(
            c.get(
                "quality_score",
                0.0,
            )
        )
        < MIN_QUALITY
    )

    output = dict(
        plan
    )

    output[
        "planner_version"
    ] = "V11"

    output[
        "actual_source_duration"
    ] = round(
        total_duration,
        6,
    )

    output[
        "missing_duration"
    ] = round(
        max(
            0.0,
            float(
                plan[
                    "reference_duration"
                ]
            )
            - total_duration,
        ),
        6,
    )

    output[
        "unique_source_scenes"
    ] = unique_scenes

    output[
        "reused_scene_count"
    ] = sum(
        1
        for count in usage.values()
        if count > 1
    )

    output[
        "average_quality"
    ] = round(
        average_quality,
        4,
    )

    output[
        "low_quality_clip_count"
    ] = low_quality

    output[
        "v11_replacements"
    ] = replacements

    output[
        "intervals"
    ] = new_intervals

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

    print()
    print("=" * 80)
    print("✅ V11 COMPLETE")
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
        f"Reused scenes      : "
        f"{output['reused_scene_count']}"
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
        f"Replacements       : "
        f"{len(replacements)}"
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)
    print("=" * 80)


if __name__ == "__main__":
    main()
