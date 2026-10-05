from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]

PLAN_FILE = (
    ROOT
    / "data"
    / "outputs"
    / "director_edit_plan_v11.json"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "outputs"
    / "visual_diversity_audit_v11.json"
)

CLIPS_DIR = (
    ROOT
    / "data"
    / "clips"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_frame(video_path, timestamp):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    frame_no = max(
        0,
        int(round(timestamp * fps))
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_no
    )

    ok, frame = cap.read()

    cap.release()

    if not ok:
        raise RuntimeError(
            f"Could not read frame at "
            f"{timestamp:.3f}s from "
            f"{video_path.name}"
        )

    return frame


def dhash(frame):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    small = cv2.resize(
        gray,
        (9, 8),
        interpolation=cv2.INTER_AREA
    )

    diff = small[:, 1:] > small[:, :-1]

    bits = diff.flatten()

    return bits


def color_histogram(frame):
    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 8],
        [0, 180, 0, 256]
    )

    hist = cv2.normalize(
        hist,
        hist
    ).flatten()

    return hist


def edge_profile(frame):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        80,
        160
    )

    small = cv2.resize(
        edges,
        (16, 9),
        interpolation=cv2.INTER_AREA
    )

    return small.astype(
        np.float32
    ).flatten() / 255.0


def frame_features(frame):
    return {
        "dhash": dhash(frame),
        "color": color_histogram(frame),
        "edges": edge_profile(frame),
    }


def compare(a, b):

    dhash_distance = int(
        np.count_nonzero(
            a["dhash"] != b["dhash"]
        )
    )

    # Bhattacharyya distance -> similarity.
    bhatta = cv2.compareHist(
        a["color"].astype(np.float32),
        b["color"].astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA
    )

    color_similarity = max(
        0.0,
        1.0 - float(bhatta)
    )

    edge_distance = float(
        np.mean(
            np.abs(
                a["edges"]
                - b["edges"]
            )
        )
    )

    edge_similarity = max(
        0.0,
        1.0 - edge_distance
    )

    # Combined perceptual similarity.
    similarity = (
        (1.0 - dhash_distance / 64.0)
        * 0.45
        +
        color_similarity * 0.35
        +
        edge_similarity * 0.20
    )

    return {
        "dhash_distance": dhash_distance,
        "color_similarity":
            round(color_similarity, 4),
        "edge_similarity":
            round(edge_similarity, 4),
        "similarity":
            round(similarity, 4),
    }


def main():

    print()
    print("=" * 80)
    print("🔎 V10 — VISUAL DIVERSITY CRITIC")
    print("=" * 80)

    plan = load_json(
        PLAN_FILE
    )

    all_clips = []

    for interval in plan["intervals"]:

        cursor = float(
            interval["reference_start"]
        )

        for clip in interval["clips"]:

            duration = float(
                clip["duration"]
            )

            target_start = cursor
            target_end = (
                cursor + duration
            )

            center = (
                float(
                    clip["source_start"]
                )
                +
                float(
                    clip["source_end"]
                )
            ) / 2.0

            all_clips.append(
                {
                    "interval_id":
                        interval["interval_id"],
                    "target_start":
                        target_start,
                    "target_end":
                        target_end,
                    "source_video":
                        clip["source_video"],
                    "scene_id":
                        clip["scene_id"],
                    "source_start":
                        clip["source_start"],
                    "source_end":
                        clip["source_end"],
                    "quality_score":
                        clip["quality_score"],
                    "center":
                        center,
                }
            )

            cursor = target_end

    print(
        f"Clips to inspect : "
        f"{len(all_clips)}"
    )

    features = []
    cache = {}

    for i, clip in enumerate(
        all_clips,
        start=1
    ):

        video_name = clip[
            "source_video"
        ]

        timestamp = float(
            clip["center"]
        )

        cache_key = (
            video_name,
            round(timestamp, 3)
        )

        if cache_key not in cache:

            frame = extract_frame(
                CLIPS_DIR / video_name,
                timestamp
            )

            cache[cache_key] = (
                frame_features(frame)
            )

        features.append(
            cache[cache_key]
        )

        if i % 10 == 0 or i == len(all_clips):
            print(
                f"  analyzed {i}/{len(all_clips)}"
            )

    # --------------------------------------------------------
    # Pairwise similarity.
    # --------------------------------------------------------

    pairs = []

    for i in range(
        len(all_clips)
    ):

        for j in range(
            i + 1,
            len(all_clips)
        ):

            result = compare(
                features[i],
                features[j]
            )

            pairs.append(
                {
                    "clip_a": i + 1,
                    "clip_b": j + 1,
                    **result,
                    "same_scene":
                        (
                            all_clips[i][
                                "source_video"
                            ]
                            ==
                            all_clips[j][
                                "source_video"
                            ]
                            and
                            all_clips[i][
                                "scene_id"
                            ]
                            ==
                            all_clips[j][
                                "scene_id"
                            ]
                        ),
                    "target_a":
                        round(
                            all_clips[i][
                                "target_start"
                            ],
                            3
                        ),
                    "target_b":
                        round(
                            all_clips[j][
                                "target_start"
                            ],
                            3
                        ),
                }
            )

    pairs.sort(
        key=lambda x:
            x["similarity"],
        reverse=True
    )

    # --------------------------------------------------------
    # Flag potentially repetitive pairs.
    # --------------------------------------------------------

    flagged = [
        p
        for p in pairs
        if p["similarity"] >= 0.86
    ]

    # Only count meaningful temporal separation.
    meaningful_flagged = []

    for p in flagged:

        clip_a = all_clips[
            p["clip_a"] - 1
        ]

        clip_b = all_clips[
            p["clip_b"] - 1
        ]

        target_gap = abs(
            clip_a["target_start"]
            -
            clip_b["target_start"]
        )

        # Adjacent cuts can naturally resemble each other.
        # Strong similarity becomes more concerning when
        # separated by at least 1.5 seconds.
        if (
            target_gap >= 1.5
            or p["same_scene"]
        ):
            meaningful_flagged.append(
                p
            )

    # --------------------------------------------------------
    # Per-clip repetition score.
    # --------------------------------------------------------

    max_similarity = {
        i + 1: 0.0
        for i in range(
            len(all_clips)
        )
    }

    duplicate_partner = {}

    for p in meaningful_flagged:

        for side in [
            "clip_a",
            "clip_b"
        ]:

            idx = p[side]

            if (
                p["similarity"]
                > max_similarity[idx]
            ):
                max_similarity[idx] = (
                    p["similarity"]
                )

                duplicate_partner[idx] = (
                    p[
                        "clip_b"
                    ]
                    if side == "clip_a"
                    else
                    p[
                        "clip_a"
                    ]
                )

    flagged_clips = []

    for idx, similarity in sorted(
        max_similarity.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        if similarity >= 0.86:

            clip = all_clips[
                idx - 1
            ]

            partner_idx = (
                duplicate_partner
                .get(idx)
            )

            partner = (
                all_clips[
                    partner_idx - 1
                ]
                if partner_idx
                else None
            )

            flagged_clips.append(
                {
                    "clip_index":
                        idx,
                    "similarity":
                        round(
                            similarity,
                            4
                        ),
                    "source_video":
                        clip[
                            "source_video"
                        ],
                    "scene_id":
                        clip[
                            "scene_id"
                        ],
                    "target_start":
                        round(
                            clip[
                                "target_start"
                            ],
                            3
                        ),
                    "quality_score":
                        clip[
                            "quality_score"
                        ],
                    "partner_clip":
                        partner_idx,
                    "partner_target":
                        (
                            round(
                                partner[
                                    "target_start"
                                ],
                                3
                            )
                            if partner
                            else None
                        ),
                }
            )

    output = {
        "project":
            "AMV Director",

        "version":
            "V10-audit",

        "clip_count":
            len(all_clips),

        "flagged_pair_count":
            len(meaningful_flagged),

        "flagged_clip_count":
            len(flagged_clips),

        "top_pairs":
            pairs[:30],

        "flagged_clips":
            flagged_clips,

        "clips":
            all_clips,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2
        )

    print()
    print("=" * 80)
    print("✅ V10 VISUAL DIVERSITY AUDIT COMPLETE")
    print("=" * 80)

    print(
        f"Clips analyzed       : "
        f"{len(all_clips)}"
    )

    print(
        f"Suspicious pairs     : "
        f"{len(meaningful_flagged)}"
    )

    print(
        f"Flagged clips        : "
        f"{len(flagged_clips)}"
    )

    print()

    if flagged_clips:

        print(
            "⚠️ MOST REPETITIVE SHOTS"
        )

        for item in flagged_clips[:20]:

            print(
                f"  #{item['clip_index']:02d} "
                f"{item['source_video']} "
                f"S{item['scene_id']:02d} "
                f"at {item['target_start']:.3f}s "
                f"<-> "
                f"#{item['partner_clip']} "
                f"at {item['partner_target']:.3f}s "
                f"similarity="
                f"{item['similarity']:.3f}"
            )

    else:

        print(
            "🎉 No strong visual repetitions detected."
        )

    print()
    print("Saved:")
    print(OUTPUT_FILE)
    print("=" * 80)


if __name__ == "__main__":
    main()
