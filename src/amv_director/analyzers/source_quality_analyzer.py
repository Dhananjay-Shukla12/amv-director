from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CLIPS_DIR = (
    PROJECT_ROOT
    / "data"
    / "clips"
)

CLIP_ANALYSIS_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "clip_analysis"
    / "all_clips_analysis.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "clip_analysis"
    / "source_quality_analysis.json"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_FPS = 5.0

BLACK_THRESHOLD = 18
DARK_THRESHOLD = 35
BRIGHT_THRESHOLD = 245

LOW_CONTRAST_THRESHOLD = 18.0
LOW_EDGE_THRESHOLD = 0.015

# These don't mean "bad" by themselves.
# They contribute to a suspicion score.
TITLE_CARD_EDGE_MIN = 0.035
TITLE_CARD_COLOR_MAX = 45.0
TITLE_CARD_BRIGHT_RATIO_MIN = 0.015
TITLE_CARD_DARK_RATIO_MIN = 0.25

# Final quality weights.
QUALITY_WEIGHTS = {
    "information": 0.28,
    "sharpness": 0.22,
    "contrast": 0.15,
    "color": 0.12,
    "temporal_stability": 0.08,
    "dark_penalty": 0.10,
    "card_penalty": 0.05,
}


# ============================================================
# HELPERS
# ============================================================

def clamp(value, low=0.0, high=1.0):
    return max(
        low,
        min(high, float(value)),
    )


def safe_float(value, default=0.0):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return default


def robust_normalize(
    value,
    low,
    high,
):
    if high - low < 1e-9:
        return 0.5

    return clamp(
        (value - low) /
        (high - low)
    )


# ============================================================
# FRAME METRICS
# ============================================================

def frame_metrics(frame):
    """
    Calculate objective visual properties.

    These features are deliberately independent of semantic
    understanding. We are measuring the quality/information
    available in the image.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    sharpness_raw = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    edges = cv2.Canny(
        gray,
        80,
        180,
    )

    edge_density = float(
        np.mean(
            edges > 0
        )
    )

    saturation = hsv[:, :, 1]

    mean_saturation = float(
        np.mean(saturation)
    )

    color_fraction = float(
        np.mean(
            saturation > 35
        )
    )

    dark_ratio = float(
        np.mean(
            gray < DARK_THRESHOLD
        )
    )

    black_ratio = float(
        np.mean(
            gray < BLACK_THRESHOLD
        )
    )

    bright_ratio = float(
        np.mean(
            gray > BRIGHT_THRESHOLD
        )
    )

    # Rough spatial information estimate.
    #
    # A completely blank image will have low contrast,
    # low edges and low information.
    resized = cv2.resize(
        gray,
        (64, 36),
        interpolation=cv2.INTER_AREA,
    )

    histogram = cv2.calcHist(
        [resized],
        [0],
        None,
        [32],
        [0, 256],
    )

    histogram = histogram.flatten()

    histogram_sum = (
        float(
            np.sum(histogram)
        )
        or 1.0
    )

    histogram = (
        histogram /
        histogram_sum
    )

    entropy = float(
        -np.sum(
            histogram *
            np.log2(
                histogram + 1e-10
            )
        )
    )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness_raw": sharpness_raw,
        "edge_density": edge_density,
        "mean_saturation": mean_saturation,
        "color_fraction": color_fraction,
        "dark_ratio": dark_ratio,
        "black_ratio": black_ratio,
        "bright_ratio": bright_ratio,
        "entropy": entropy,
    }


# ============================================================
# VIDEO SAMPLING
# ============================================================

def sample_video(
    video_path,
    scenes,
):
    """
    Sequentially decode the video.

    We intentionally do NOT repeatedly seek H264 frames because
    earlier random seeking caused decoder warnings.
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = safe_float(
        cap.get(
            cv2.CAP_PROP_FPS
        ),
        30.0,
    )

    if fps <= 0:
        fps = 30.0

    frame_step = max(
        1,
        int(
            round(
                fps /
                SAMPLE_FPS
            )
        ),
    )

    scene_buffers = {
        int(scene["scene_id"]): []
        for scene in scenes
    }

    scene_index = 0

    current_scene = (
        scenes[scene_index]
        if scenes
        else None
    )

    frame_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        timestamp = (
            frame_index /
            fps
        )

        frame_index += 1

        if (
            current_scene is None
            or
            frame_index % frame_step != 0
        ):
            continue

        # Move through scenes as time advances.
        while (
            current_scene is not None
            and
            timestamp >
            float(
                current_scene["end"]
            )
            and
            scene_index <
            len(scenes) - 1
        ):
            scene_index += 1
            current_scene = scenes[
                scene_index
            ]

        if current_scene is None:
            continue

        start = float(
            current_scene["start"]
        )

        end = float(
            current_scene["end"]
        )

        if (
            timestamp < start
            or
            timestamp >= end
        ):
            continue

        metrics = frame_metrics(
            frame
        )

        scene_buffers[
            int(
                current_scene[
                    "scene_id"
                ]
            )
        ].append(
            metrics
        )

    cap.release()

    return scene_buffers


# ============================================================
# TEMPORAL STABILITY
# ============================================================

def temporal_stability(metrics):

    if len(metrics) < 2:
        return 0.5

    brightness = np.array(
        [
            item["brightness"]
            for item in metrics
        ],
        dtype=float,
    )

    contrast = np.array(
        [
            item["contrast"]
            for item in metrics
        ],
        dtype=float,
    )

    brightness_variation = (
        np.std(
            brightness
        )
        /
        max(
            np.mean(brightness),
            1.0,
        )
    )

    contrast_variation = (
        np.std(
            contrast
        )
        /
        max(
            np.mean(contrast),
            1.0,
        )
    )

    variation = (
        brightness_variation
        +
        contrast_variation
    ) / 2.0

    # This is NOT saying stable = always good.
    # It prevents a wildly unstable/blank source from getting
    # treated as a consistently usable scene.
    return clamp(
        1.0 -
        variation
    )


# ============================================================
# TITLE CARD / GRAPHIC SUSPICION
# ============================================================

def graphic_card_suspicion(
    metrics_mean,
):
    """
    Heuristic only.

    This is intentionally called "suspicion" rather than
    "is_title_card" because anime title cards, subtitles,
    dark scenes and stylized frames can overlap visually.
    """

    edge = metrics_mean[
        "edge_density"
    ]

    color = metrics_mean[
        "mean_saturation"
    ]

    bright = metrics_mean[
        "bright_ratio"
    ]

    dark = metrics_mean[
        "dark_ratio"
    ]

    if (
        edge >= TITLE_CARD_EDGE_MIN
        and
        color <= TITLE_CARD_COLOR_MAX
        and
        bright >= TITLE_CARD_BRIGHT_RATIO_MIN
        and
        dark >= TITLE_CARD_DARK_RATIO_MIN
    ):
        return 0.85

    # Secondary weaker suspicion.
    if (
        color <= 30
        and
        bright >= 0.025
        and
        dark >= 0.35
    ):
        return 0.55

    return 0.0


# ============================================================
# AGGREGATE SCENE
# ============================================================

def aggregate_scene(
    scene,
    metrics,
):

    if not metrics:

        return {
            "quality_score": 0.0,
            "usable": False,
            "reason": "no_frames_sampled",
        }

    keys = [
        "brightness",
        "contrast",
        "sharpness_raw",
        "edge_density",
        "mean_saturation",
        "color_fraction",
        "dark_ratio",
        "black_ratio",
        "bright_ratio",
        "entropy",
    ]

    means = {}

    for key in keys:

        means[key] = float(
            np.mean(
                [
                    item[key]
                    for item in metrics
                ]
            )
        )

    # Log-transform sharpness because Laplacian variance
    # can span several orders of magnitude.
    sharpness_log = math.log1p(
        means["sharpness_raw"]
    )

    # Information combines entropy, edges and contrast.
    entropy_score = clamp(
        means["entropy"] /
        5.0
    )

    edge_score = clamp(
        means["edge_density"] /
        0.20
    )

    contrast_score = clamp(
        means["contrast"] /
        70.0
    )

    information_score = (
        entropy_score * 0.40
        +
        edge_score * 0.30
        +
        contrast_score * 0.30
    )

    sharpness_score = clamp(
        sharpness_log /
        math.log1p(500.0)
    )

    color_score = clamp(
        means["mean_saturation"] /
        100.0
    )

    contrast_score = clamp(
        means["contrast"] /
        70.0
    )

    dark_penalty = clamp(
        (
            means["dark_ratio"]
            -
            0.55
        )
        /
        0.35
    )

    black_penalty = clamp(
        (
            means["black_ratio"]
            -
            0.40
        )
        /
        0.50
    )

    dark_penalty = max(
        dark_penalty,
        black_penalty,
    )

    card_suspicion = (
        graphic_card_suspicion(
            means
        )
    )

    stability = temporal_stability(
        metrics
    )

    # Combine into quality.
    quality = (
        information_score
        * QUALITY_WEIGHTS[
            "information"
        ]
        +

        sharpness_score
        * QUALITY_WEIGHTS[
            "sharpness"
        ]
        +

        contrast_score
        * QUALITY_WEIGHTS[
            "contrast"
        ]
        +

        color_score
        * QUALITY_WEIGHTS[
            "color"
        ]
        +

        stability
        * QUALITY_WEIGHTS[
            "temporal_stability"
        ]
    )

    quality -= (
        dark_penalty
        *
        QUALITY_WEIGHTS[
            "dark_penalty"
        ]
    )

    quality -= (
        card_suspicion
        *
        QUALITY_WEIGHTS[
            "card_penalty"
        ]
    )

    quality = clamp(
        quality
    )

    # A quality score isn't an absolute semantic judgment.
    # This threshold only identifies obviously weak candidates.
    usable = quality >= 0.30

    if (
        means["black_ratio"] >= 0.70
    ):
        reason = "mostly_black"

    elif (
        means["contrast"] <
        LOW_CONTRAST_THRESHOLD
        and
        means["edge_density"] <
        LOW_EDGE_THRESHOLD
    ):
        reason = "low_information"

    elif card_suspicion >= 0.75:
        reason = "possible_graphic_card"

    elif (
        means["dark_ratio"] >= 0.75
    ):
        reason = "very_dark"

    elif (
        sharpness_score < 0.15
    ):
        reason = "soft"

    else:
        reason = "usable"

    return {
        "quality_score": round(
            quality,
            4,
        ),

        "usable": usable,

        "reason": reason,

        "samples": len(
            metrics
        ),

        "metrics": {
            "brightness": round(
                means["brightness"],
                3,
            ),

            "contrast": round(
                means["contrast"],
                3,
            ),

            "sharpness_raw": round(
                means["sharpness_raw"],
                3,
            ),

            "edge_density": round(
                means["edge_density"],
                4,
            ),

            "mean_saturation": round(
                means["mean_saturation"],
                3,
            ),

            "color_fraction": round(
                means["color_fraction"],
                4,
            ),

            "dark_ratio": round(
                means["dark_ratio"],
                4,
            ),

            "black_ratio": round(
                means["black_ratio"],
                4,
            ),

            "bright_ratio": round(
                means["bright_ratio"],
                4,
            ),

            "entropy": round(
                means["entropy"],
                4,
            ),
        },

        "subscores": {
            "information": round(
                information_score,
                4,
            ),

            "sharpness": round(
                sharpness_score,
                4,
            ),

            "contrast": round(
                contrast_score,
                4,
            ),

            "color": round(
                color_score,
                4,
            ),

            "temporal_stability": round(
                stability,
                4,
            ),

            "dark_penalty": round(
                dark_penalty,
                4,
            ),

            "graphic_card_suspicion": round(
                card_suspicion,
                4,
            ),
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("🔎 SOURCE QUALITY ANALYZER")
    print()

    if not CLIP_ANALYSIS_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{CLIP_ANALYSIS_FILE}\n\n"
            "Run clip_analyzer.py first."
        )

    with CLIP_ANALYSIS_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        clip_data = json.load(f)

    all_results = []

    total_scenes = 0
    usable_scenes = 0

    for video_data in clip_data:

        video_name = video_data[
            "video"
        ]

        scenes = video_data.get(
            "scenes",
            [],
        )

        video_path = (
            CLIPS_DIR /
            video_name
        )

        print(
            f"🎬 {video_name}"
        )

        print(
            f"   Scenes: {len(scenes)}"
        )

        if not video_path.exists():
            print(
                f"   ❌ Missing: {video_path}"
            )
            continue

        buffers = sample_video(
            video_path,
            scenes,
        )

        video_result = {
            "video": video_name,
            "scenes": [],
        }

        for scene in scenes:

            scene_id = int(
                scene["scene_id"]
            )

            metrics = buffers.get(
                scene_id,
                [],
            )

            result = aggregate_scene(
                scene,
                metrics,
            )

            item = {
                "scene_id": scene_id,
                "start": scene["start"],
                "end": scene["end"],
                "duration": scene["duration"],
                **result,
            }

            video_result[
                "scenes"
            ].append(item)

            total_scenes += 1

            if result.get(
                "usable",
                False,
            ):
                usable_scenes += 1

        all_results.append(
            video_result
        )

    # --------------------------------------------------------
    # Rankings
    # --------------------------------------------------------

    flat = []

    for video in all_results:

        for scene in video["scenes"]:

            item = dict(scene)

            item["video"] = video[
                "video"
            ]

            flat.append(item)

    flat.sort(
        key=lambda x: x.get(
            "quality_score",
            0.0,
        ),
        reverse=True,
    )

    weakest = sorted(
        flat,
        key=lambda x: x.get(
            "quality_score",
            0.0,
        ),
    )

    output = {
        "project": "AMV Director",
        "version": "source_quality_v1",

        "summary": {
            "videos": len(
                all_results
            ),
            "scenes": total_scenes,
            "usable_scenes": usable_scenes,
            "rejected_or_weak": (
                total_scenes -
                usable_scenes
            ),
        },

        "videos": all_results,

        "top_quality_scenes": flat[:15],

        "weakest_scenes": weakest[:15],
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 84)
    print("🔎 SOURCE QUALITY REPORT")
    print("=" * 84)

    print(
        f"Scenes analyzed : "
        f"{total_scenes}"
    )

    print(
        f"Usable          : "
        f"{usable_scenes}"
    )

    print(
        f"Weak/suspect    : "
        f"{total_scenes - usable_scenes}"
    )

    print()
    print("TOP QUALITY SCENES:")

    for index, item in enumerate(
        flat[:10],
        start=1,
    ):

        print(
            f"  #{index:02d} "
            f"{item['video']} "
            f"S{item['scene_id']:02d} | "
            f"{item['quality_score']:.3f} | "
            f"{item['duration']:.3f}s | "
            f"{item['reason']}"
        )

    print()
    print("WEAKEST / SUSPICIOUS:")

    for index, item in enumerate(
        weakest[:15],
        start=1,
    ):

        print(
            f"  #{index:02d} "
            f"{item['video']} "
            f"S{item['scene_id']:02d} | "
            f"{item['quality_score']:.3f} | "
            f"{item['duration']:.3f}s | "
            f"{item['reason']} | "
            f"card={item['subscores']['graphic_card_suspicion']:.2f} | "
            f"black={item['metrics']['black_ratio']:.2f}"
        )

    print()
    print("=" * 84)
    print("✅ Saved:")
    print(OUTPUT_FILE)
    print("=" * 84)


if __name__ == "__main__":
    main()