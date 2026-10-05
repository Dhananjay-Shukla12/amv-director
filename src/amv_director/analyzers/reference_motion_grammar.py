import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

REFERENCE_VIDEO = (
    BASE_DIR
    / "data"
    / "references"
    / "reference_video.mp4"
)

SHOT_MAP_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "reference_shot_map.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "reference_motion_grammar.json"
)


# ============================================================
# SETTINGS
# ============================================================

ANALYSIS_WIDTH = 320
ANALYSIS_HEIGHT = 180

SAMPLES_PER_SECOND = 8.0

FLOW_PYR_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 15
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2

# Global normalization percentiles.
MOTION_LOW_PERCENTILE = 15
MOTION_HIGH_PERCENTILE = 90

FLASH_LOW_PERCENTILE = 25
FLASH_HIGH_PERCENTILE = 98

EPSILON = 1e-8


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        if isinstance(
            value,
            (list, tuple, np.ndarray)
        ):

            if len(value) == 0:
                return default

            value = value[0]

        return float(value)

    except Exception:

        return default


def clamp(
    value,
    low=0.0,
    high=1.0
):

    return max(
        low,
        min(high, value)
    )


def normalize_robust(
    value,
    low,
    high
):

    if high <= low:

        return 0.0

    return clamp(
        (
            value - low
        )
        /
        (
            high - low
        )
    )


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


def resize_gray(frame):

    small = cv2.resize(
        frame,
        (
            ANALYSIS_WIDTH,
            ANALYSIS_HEIGHT
        ),
        interpolation=cv2.INTER_AREA
    )

    return cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )


# ============================================================
# OPTICAL FLOW
# ============================================================

def calculate_flow_metrics(
    prev_gray,
    curr_gray
):

    flow = cv2.calcOpticalFlowFarneback(

        prev_gray,
        curr_gray,
        None,

        FLOW_PYR_SCALE,
        FLOW_LEVELS,
        FLOW_WINSIZE,
        FLOW_ITERATIONS,
        FLOW_POLY_N,
        FLOW_POLY_SIGMA,

        0
    )

    fx = flow[..., 0]
    fy = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(
        fx,
        fy,
        angleInDegrees=False
    )

    # --------------------------------------------------------
    # Robust motion measurements
    # --------------------------------------------------------

    magnitude_flat = (
        magnitude
        .reshape(-1)
    )

    mean_magnitude = float(
        np.median(
            magnitude_flat
        )
    )

    p90_magnitude = float(
        np.percentile(
            magnitude_flat,
            90
        )
    )

    p95_magnitude = float(
        np.percentile(
            magnitude_flat,
            95
        )
    )

    # --------------------------------------------------------
    # Global flow vector
    # --------------------------------------------------------

    mean_fx = float(
        np.median(fx)
    )

    mean_fy = float(
        np.median(fy)
    )

    vector_magnitude = float(
        np.sqrt(
            mean_fx ** 2
            + mean_fy ** 2
        )
    )

    coherence = (
        vector_magnitude
        /
        max(
            mean_magnitude,
            EPSILON
        )
    )

    coherence = clamp(
        coherence
    )

    # --------------------------------------------------------
    # Horizontal / vertical components
    # --------------------------------------------------------

    horizontal = float(
        np.median(
            np.abs(fx)
        )
    )

    vertical = float(
        np.median(
            np.abs(fy)
        )
    )

    # --------------------------------------------------------
    # Radial flow around image centre
    # --------------------------------------------------------

    h, w = magnitude.shape

    yy, xx = np.mgrid[
        0:h,
        0:w
    ]

    cx = (
        w - 1
    ) / 2.0

    cy = (
        h - 1
    ) / 2.0

    dx = xx - cx
    dy = yy - cy

    radius = np.sqrt(
        dx ** 2
        + dy ** 2
    )

    radius = np.maximum(
        radius,
        1.0
    )

    unit_x = (
        dx / radius
    )

    unit_y = (
        dy / radius
    )

    radial = (
        fx * unit_x
        + fy * unit_y
    )

    radial_mean = float(
        np.median(radial)
    )

    # --------------------------------------------------------
    # Tangential flow
    # --------------------------------------------------------

    tangential = (
        -fx * unit_y
        + fy * unit_x
    )

    tangential_mean = float(
        np.median(tangential)
    )

    return {

        "mean_magnitude":
            mean_magnitude,

        "p90_magnitude":
            p90_magnitude,

        "p95_magnitude":
            p95_magnitude,

        "mean_fx":
            mean_fx,

        "mean_fy":
            mean_fy,

        "vector_magnitude":
            vector_magnitude,

        "coherence":
            coherence,

        "horizontal":
            horizontal,

        "vertical":
            vertical,

        "radial_mean":
            radial_mean,

        "tangential_mean":
            tangential_mean,
    }


# ============================================================
# SAMPLE TIMES
# ============================================================

def sample_times(
    start,
    end
):

    duration = (
        end - start
    )

    if duration <= 0:

        return []

    count = max(
        3,
        int(
            np.ceil(
                duration
                * SAMPLES_PER_SECOND
            )
        )
        + 1
    )

    values = np.linspace(
        start,
        end,
        count
    )

    return [
        float(x)
        for x in values
    ]


# ============================================================
# SHOT ANALYSIS
# ============================================================

def analyze_shot(
    cap,
    shot
):

    start = safe_float(
        shot.get(
            "start",
            0
        )
    )

    end = safe_float(
        shot.get(
            "end",
            start
        )
    )

    times = sample_times(
        start,
        end
    )

    if len(times) < 2:

        return {
            "raw": {},
            "derived": {
                "mean_motion":
                    0.0,

                "peak_motion":
                    0.0,

                "coherence":
                    0.0,

                "mean_dx":
                    0.0,

                "mean_dy":
                    0.0,

                "horizontal":
                    0.0,

                "vertical":
                    0.0,

                "radial":
                    0.0,

                "tangential":
                    0.0,

                "shake_score":
                    0.0,

                "brightness_change":
                    0.0,

                "max_brightness_change":
                    0.0,

                "motion_burst":
                    0.0,
            }
        }

    previous_gray = None
    previous_brightness = None

    flow_records = []
    brightness_changes = []

    for timestamp in times:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000.0
        )

        ok, frame = cap.read()

        if not ok:
            continue

        gray = resize_gray(
            frame
        )

        brightness = (
            float(
                np.mean(gray)
            )
            / 255.0
        )

        if previous_gray is not None:

            flow_metrics = (
                calculate_flow_metrics(
                    previous_gray,
                    gray
                )
            )

            flow_records.append(
                (
                    timestamp,
                    flow_metrics
                )
            )

            if previous_brightness is not None:

                brightness_changes.append(
                    abs(
                        brightness
                        - previous_brightness
                    )
                )

        previous_gray = gray
        previous_brightness = brightness

    if not flow_records:

        return {
            "raw": {},
            "derived": {
                "mean_motion":
                    0.0,

                "peak_motion":
                    0.0,

                "coherence":
                    0.0,

                "mean_dx":
                    0.0,

                "mean_dy":
                    0.0,

                "horizontal":
                    0.0,

                "vertical":
                    0.0,

                "radial":
                    0.0,

                "tangential":
                    0.0,

                "shake_score":
                    0.0,

                "brightness_change":
                    0.0,

                "max_brightness_change":
                    0.0,

                "motion_burst":
                    0.0,
            }
        }

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    mean_motion_values = np.array(
        [
            item[1]["mean_magnitude"]
            for item in flow_records
        ],
        dtype=float
    )

    p90_values = np.array(
        [
            item[1]["p90_magnitude"]
            for item in flow_records
        ],
        dtype=float
    )

    dx_values = np.array(
        [
            item[1]["mean_fx"]
            for item in flow_records
        ],
        dtype=float
    )

    dy_values = np.array(
        [
            item[1]["mean_fy"]
            for item in flow_records
        ],
        dtype=float
    )

    coherence_values = np.array(
        [
            item[1]["coherence"]
            for item in flow_records
        ],
        dtype=float
    )

    horizontal_values = np.array(
        [
            item[1]["horizontal"]
            for item in flow_records
        ],
        dtype=float
    )

    vertical_values = np.array(
        [
            item[1]["vertical"]
            for item in flow_records
        ],
        dtype=float
    )

    radial_values = np.array(
        [
            item[1]["radial_mean"]
            for item in flow_records
        ],
        dtype=float
    )

    tangential_values = np.array(
        [
            item[1]["tangential_mean"]
            for item in flow_records
        ],
        dtype=float
    )

    # --------------------------------------------------------
    # Aggregates
    # --------------------------------------------------------

    mean_motion = float(
        np.median(
            mean_motion_values
        )
    )

    peak_motion = float(
        np.percentile(
            p90_values,
            85
        )
    )

    mean_dx = float(
        np.median(
            dx_values
        )
    )

    mean_dy = float(
        np.median(
            dy_values
        )
    )

    coherence = float(
        np.median(
            coherence_values
        )
    )

    horizontal = float(
        np.median(
            horizontal_values
        )
    )

    vertical = float(
        np.median(
            vertical_values
        )
    )

    radial = float(
        np.median(
            radial_values
        )
    )

    tangential = float(
        np.median(
            tangential_values
        )
    )

    # --------------------------------------------------------
    # Motion burst
    # --------------------------------------------------------

    median_motion = float(
        np.median(
            mean_motion_values
        )
    )

    p95_motion = float(
        np.percentile(
            mean_motion_values,
            95
        )
    )

    burst_ratio = (
        p95_motion
        /
        max(
            median_motion,
            EPSILON
        )
    )

    motion_burst = clamp(
        (
            burst_ratio - 1.0
        )
        / 5.0
    )

    # --------------------------------------------------------
    # Shake
    #
    # Count genuine directional reversals.
    # Tiny noisy values are ignored relative to shot motion.
    # --------------------------------------------------------

    shake_score = 0.0

    if len(dx_values) >= 4:

        motion_scale = max(
            float(
                np.percentile(
                    mean_motion_values,
                    75
                )
            ),
            EPSILON
        )

        x_threshold = (
            motion_scale
            * 0.20
        )

        y_threshold = (
            motion_scale
            * 0.20
        )

        reversal_x = 0
        reversal_y = 0

        possible = (
            len(dx_values) - 1
        )

        for i in range(
            1,
            len(dx_values)
        ):

            if (
                abs(dx_values[i])
                >= x_threshold
                and
                abs(dx_values[i - 1])
                >= x_threshold
                and
                np.sign(
                    dx_values[i]
                )
                !=
                np.sign(
                    dx_values[i - 1]
                )
            ):

                reversal_x += 1

            if (
                abs(dy_values[i])
                >= y_threshold
                and
                abs(dy_values[i - 1])
                >= y_threshold
                and
                np.sign(
                    dy_values[i]
                )
                !=
                np.sign(
                    dy_values[i - 1]
                )
            ):

                reversal_y += 1

        reversal_ratio = (
            reversal_x
            + reversal_y
        ) / max(
            2.0 * possible,
            1.0
        )

        shake_score = clamp(
            reversal_ratio
            * 2.0
        )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    if brightness_changes:

        mean_brightness_change = float(
            np.median(
                brightness_changes
            )
        )

        max_brightness_change = float(
            np.percentile(
                brightness_changes,
                90
            )
        )

    else:

        mean_brightness_change = 0.0
        max_brightness_change = 0.0

    return {

        "raw": {

            "mean_magnitude":
                mean_motion,

            "peak_p90_magnitude":
                peak_motion,

            "mean_dx":
                mean_dx,

            "mean_dy":
                mean_dy,

            "coherence":
                coherence,

            "horizontal":
                horizontal,

            "vertical":
                vertical,

            "radial":
                radial,

            "tangential":
                tangential,

            "motion_burst":
                motion_burst,

            "shake_score":
                shake_score,

            "brightness_change":
                mean_brightness_change,

            "max_brightness_change":
                max_brightness_change,
        },

        "derived": {
            "mean_motion":
                mean_motion,

            "peak_motion":
                peak_motion,

            "coherence":
                coherence,

            "mean_dx":
                mean_dx,

            "mean_dy":
                mean_dy,

            "horizontal":
                horizontal,

            "vertical":
                vertical,

            "radial":
                radial,

            "tangential":
                tangential,

            "shake_score":
                shake_score,

            "brightness_change":
                mean_brightness_change,

            "max_brightness_change":
                max_brightness_change,

            "motion_burst":
                motion_burst,
        }
    }


# ============================================================
# GLOBAL CALIBRATION
# ============================================================

def calibrate_profiles(
    analyzed_shots
):

    motion_values = [
        shot["motion_data"]["raw"]["mean_magnitude"]
        for shot in analyzed_shots
    ]

    peak_values = [
        shot["motion_data"]["raw"]["peak_p90_magnitude"]
        for shot in analyzed_shots
    ]

    brightness_values = [
        shot["motion_data"]["raw"]["max_brightness_change"]
        for shot in analyzed_shots
    ]

    motion_low = float(
        np.percentile(
            motion_values,
            MOTION_LOW_PERCENTILE
        )
    )

    motion_high = float(
        np.percentile(
            motion_values,
            MOTION_HIGH_PERCENTILE
        )
    )

    peak_low = float(
        np.percentile(
            peak_values,
            MOTION_LOW_PERCENTILE
        )
    )

    peak_high = float(
        np.percentile(
            peak_values,
            MOTION_HIGH_PERCENTILE
        )
    )

    flash_low = float(
        np.percentile(
            brightness_values,
            FLASH_LOW_PERCENTILE
        )
    )

    flash_high = float(
        np.percentile(
            brightness_values,
            FLASH_HIGH_PERCENTILE
        )
    )

    # Avoid collapsed ranges.
    if (
        motion_high
        <= motion_low
    ):

        motion_high = (
            motion_low
            + EPSILON
        )

    if (
        peak_high
        <= peak_low
    ):

        peak_high = (
            peak_low
            + EPSILON
        )

    if (
        flash_high
        <= flash_low
    ):

        flash_high = (
            flash_low
            + EPSILON
        )

    return {

        "motion": {
            "low":
                motion_low,

            "high":
                motion_high,
        },

        "peak_motion": {
            "low":
                peak_low,

            "high":
                peak_high,
        },

        "flash": {
            "low":
                flash_low,

            "high":
                flash_high,
        }
    }


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_shot(
    motion_data,
    calibration
):

    raw = motion_data["raw"]

    mean_motion = raw[
        "mean_magnitude"
    ]

    peak_motion = raw[
        "peak_p90_magnitude"
    ]

    shake = raw[
        "shake_score"
    ]

    brightness = raw[
        "max_brightness_change"
    ]

    coherence = raw[
        "coherence"
    ]

    dx = abs(
        raw[
            "mean_dx"
        ]
    )

    dy = abs(
        raw[
            "mean_dy"
        ]
    )

    radial = abs(
        raw[
            "radial"
        ]
    )

    # --------------------------------------------------------
    # Robust normalization
    # --------------------------------------------------------

    motion_score = (
        normalize_robust(
            mean_motion,
            calibration[
                "motion"
            ]["low"],
            calibration[
                "motion"
            ]["high"]
        )
    )

    peak_score = (
        normalize_robust(
            peak_motion,
            calibration[
                "peak_motion"
            ]["low"],
            calibration[
                "peak_motion"
            ]["high"]
        )
    )

    flash_score = (
        normalize_robust(
            brightness,
            calibration[
                "flash"
            ]["low"],
            calibration[
                "flash"
            ]["high"]
        )
    )

    # --------------------------------------------------------
    # Relative direction
    # --------------------------------------------------------

    total_direction = (
        dx + dy
    )

    horizontal_ratio = (
        dx
        /
        max(
            dx + dy,
            EPSILON
        )
    )

    radial_ratio = (
        radial
        /
        max(
            mean_motion,
            EPSILON
        )
    )

    # --------------------------------------------------------
    # Direction label
    # --------------------------------------------------------

    direction = "none"

    if motion_score >= 0.25:

        if (
            radial_ratio >= 0.45
            and coherence >= 0.18
        ):

            # Preserve radial sign.
            if raw["radial"] < 0:
                direction = "zoom_in"
            else:
                direction = "zoom_out"

        elif (
            horizontal_ratio >= 0.68
            and coherence >= 0.20
        ):

            if raw["mean_dx"] < 0:
                direction = "left"
            else:
                direction = "right"

        elif (
            horizontal_ratio <= 0.32
            and coherence >= 0.20
        ):

            if raw["mean_dy"] < 0:
                direction = "up"
            else:
                direction = "down"

        else:

            direction = "mixed"

    # --------------------------------------------------------
    # Pattern classification
    # --------------------------------------------------------

    if (
        flash_score >= 0.80
        and peak_score >= 0.70
    ):

        pattern = (
            "flash_impact"
        )

    elif (
        shake >= 0.65
        and peak_score >= 0.55
    ):

        pattern = (
            "shake_burst"
        )

    elif (
        peak_score >= 0.75
        and raw["motion_burst"] >= 0.45
    ):

        pattern = (
            "impact_burst"
        )

    elif motion_score < 0.12:

        pattern = (
            "static_hold"
        )

    elif motion_score < 0.30:

        pattern = (
            "subtle_drift"
        )

    elif direction == "zoom_in":

        pattern = (
            "zoom_in"
        )

    elif direction == "zoom_out":

        pattern = (
            "zoom_out"
        )

    elif direction in {
        "left",
        "right",
        "up",
        "down"
    }:

        pattern = (
            "directional_motion"
        )

    else:

        pattern = (
            "mixed_motion"
        )

    return {

        "motion_score":
            round(
                motion_score,
                4
            ),

        "peak_motion_score":
            round(
                peak_score,
                4
            ),

        "shake_score":
            round(
                shake,
                4
            ),

        "flash_score":
            round(
                flash_score,
                4
            ),

        "directional_coherence":
            round(
                coherence,
                4
            ),

        "dominant_direction":
            direction,

        "motion_pattern":
            pattern,
    }


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    shots
):

    patterns = {}

    for shot in shots:

        pattern = (
            shot[
                "motion_profile"
            ][
                "motion_pattern"
            ]
        )

        patterns[
            pattern
        ] = (
            patterns.get(
                pattern,
                0
            )
            + 1
        )

    motion = [
        safe_float(
            s["motion_profile"][
                "motion_score"
            ]
        )
        for s in shots
    ]

    shake = [
        safe_float(
            s["motion_profile"][
                "shake_score"
            ]
        )
        for s in shots
    ]

    flash = [
        safe_float(
            s["motion_profile"][
                "flash_score"
            ]
        )
        for s in shots
    ]

    return {

        "shot_count":
            len(shots),

        "motion_pattern_distribution":
            patterns,

        "average_motion":
            round(
                float(
                    np.mean(
                        motion
                    )
                ),
                4
            )
            if motion else 0.0,

        "average_shake":
            round(
                float(
                    np.mean(
                        shake
                    )
                ),
                4
            )
            if shake else 0.0,

        "average_flash":
            round(
                float(
                    np.mean(
                        flash
                    )
                ),
                4
            )
            if flash else 0.0,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 82)
    print("🎥 REFERENCE MOTION GRAMMAR")
    print("=" * 82)

    if not REFERENCE_VIDEO.exists():

        raise FileNotFoundError(
            f"Reference video not found:\n"
            f"{REFERENCE_VIDEO}"
        )

    shot_map = load_json(
        SHOT_MAP_PATH
    )

    shots = shot_map.get(
        "shots",
        []
    )

    if not shots:

        raise RuntimeError(
            "No reference shots found."
        )

    cap = cv2.VideoCapture(
        str(
            REFERENCE_VIDEO
        )
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open reference video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    print()
    print(
        f"Reference FPS    : "
        f"{fps:.3f}"
    )

    print(
        f"Reference frames : "
        f"{frame_count}"
    )

    print(
        f"Duration         : "
        f"{duration:.3f}s"
    )

    print(
        f"Shots            : "
        f"{len(shots)}"
    )

    # --------------------------------------------------------
    # First pass
    # --------------------------------------------------------

    analyzed_shots = []

    print()
    print("PASS 1: RAW MOTION MEASUREMENT")

    for index, shot in enumerate(
        shots,
        start=1
    ):

        start = safe_float(
            shot.get(
                "start",
                0
            )
        )

        end = safe_float(
            shot.get(
                "end",
                start
            )
        )

        motion_data = analyze_shot(
            cap,
            shot
        )

        analyzed_shots.append(
            {
                "source_shot":
                    index,

                "start":
                    start,

                "end":
                    end,

                "duration":
                    end - start,

                "motion_data":
                    motion_data,
            }
        )

        raw = motion_data[
            "raw"
        ]

        print(
            f"  #{index:02d} "
            f"motion="
            f"{raw.get('mean_magnitude', 0):.5f} "
            f"peak="
            f"{raw.get('peak_p90_magnitude', 0):.5f} "
            f"shake="
            f"{raw.get('shake_score', 0):.2f} "
            f"brightness="
            f"{raw.get('max_brightness_change', 0):.4f}"
        )

    cap.release()

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    calibration = calibrate_profiles(
        analyzed_shots
    )

    print()
    print("GLOBAL CALIBRATION")

    print(
        f"Motion range : "
        f"{calibration['motion']['low']:.5f}"
        f" → "
        f"{calibration['motion']['high']:.5f}"
    )

    print(
        f"Peak range   : "
        f"{calibration['peak_motion']['low']:.5f}"
        f" → "
        f"{calibration['peak_motion']['high']:.5f}"
    )

    print(
        f"Flash range  : "
        f"{calibration['flash']['low']:.5f}"
        f" → "
        f"{calibration['flash']['high']:.5f}"
    )

    # --------------------------------------------------------
    # Second pass
    # --------------------------------------------------------

    output_shots = []

    print()
    print("PASS 2: MOTION CLASSIFICATION")

    for item in analyzed_shots:

        profile = classify_shot(
            item[
                "motion_data"
            ],
            calibration
        )

        source_shot = item[
            "source_shot"
        ]

        print(
            f"  #{source_shot:02d} "
            f"{profile['motion_pattern']:<20} "
            f"motion="
            f"{profile['motion_score']:.2f} "
            f"shake="
            f"{profile['shake_score']:.2f} "
            f"flash="
            f"{profile['flash_score']:.2f} "
            f"dir="
            f"{profile['dominant_direction']}"
        )

        output_shots.append(
            {
                "shot_number":
                    source_shot,

                "start":
                    item[
                        "start"
                    ],

                "end":
                    item[
                        "end"
                    ],

                "duration":
                    item[
                        "duration"
                    ],

                "motion_profile":
                    profile,

                "raw_motion":
                    item[
                        "motion_data"
                    ],
            }
        )

    summary = build_summary(
        output_shots
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {

        "meta": {

            "name":
                "reference_motion_grammar",

            "version":
                "2.0",

            "purpose":
                (
                    "Measure motion behavior in the reference "
                    "using robust, reference-relative calibration."
                ),
        },

        "video": {

            "path":
                str(
                    REFERENCE_VIDEO
                ),

            "fps":
                fps,

            "duration":
                duration,
        },

        "calibration":
            calibration,

        "summary":
            summary,

        "shots":
            output_shots,

        "director_rules": {

            "reference_motion_is_behavior":
                True,

            "do_not_copy_reference_subject":
                True,

            "zoom_is_not_default":
                True,

            "shake_is_not_default":
                True,

            "flash_is_not_default":
                True,

            "use_reference_direction_when_relevant":
                True,

            "use_reference_motion_intensity":
                True,

            "preserve_source_composition":
                True,

            "source_motion_can_override_reference":
                True,
        },
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

    print()
    print("=" * 82)
    print("✅ REFERENCE MOTION GRAMMAR COMPLETE")
    print("=" * 82)

    print(
        f"Shots analyzed : "
        f"{summary['shot_count']}"
    )

    print()
    print(
        "MOTION PATTERNS"
    )

    for pattern, count in sorted(
        summary[
            "motion_pattern_distribution"
        ].items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"  {pattern:<20}: "
            f"{count}"
        )

    print()

    print(
        f"Average motion : "
        f"{summary['average_motion']:.3f}"
    )

    print(
        f"Average shake  : "
        f"{summary['average_shake']:.3f}"
    )

    print(
        f"Average flash  : "
        f"{summary['average_flash']:.3f}"
    )

    print()
    print("Saved:")
    print(
        OUTPUT_PATH
    )

    print(
        "=" * 82
    )


if __name__ == "__main__":
    main()