from __future__ import annotations

import json
from pathlib import Path

import cv2
import librosa
import numpy as np
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[3]

REFERENCE_VIDEO = (
    ROOT
    / "data"
    / "references"
    / "reference_video.mp4"
)

REFERENCE_AUDIO = (
    ROOT
    / "data"
    / "outputs"
    / "reference_audio.wav"
)

OUTPUT = (
    ROOT
    / "data"
    / "outputs"
    / "reference_shot_map.json"
)


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

SAMPLE_WIDTH = 320
SAMPLE_HEIGHT = 180

MIN_SHOT_DURATION = 0.12
MIN_EVENT_DISTANCE = 0.12

# Adaptive percentiles for this reference.
CUT_PERCENTILE = 94
EVENT_PERCENTILE = 90


# ------------------------------------------------------------
# Audio
# ------------------------------------------------------------

def load_audio():

    y, sr = librosa.load(
        str(REFERENCE_AUDIO),
        sr=None,
        mono=True,
    )

    return y, sr


# def audio_features(y, sr):

#     hop = 512

#     onset_env = librosa.onset.onset_strength(
#         y=y,
#         sr=sr,
#         hop_length=hop,
#         aggregate="mean",
#     )

#     times = librosa.times_like(
#         onset_env,
#         sr=sr,
#         hop_length=hop,
#     )

#     beat_times, _ = librosa.beat.beat_track(
#         y=y,
#         sr=sr,
#         hop_length=hop,
#     )

#     beat_times = np.asarray(
#         beat_times,
#         dtype=float,
#     ).reshape(-1)

#     # --------------------------------------------------------
#     # Detect stronger musical onsets.
#     # --------------------------------------------------------

#     threshold = np.percentile(
#         onset_env,
#         85,
#     )

#     peaks, _ = find_peaks(
#         onset_env,
#         height=threshold,
#         distance=max(
#             1,
#             int(
#                 0.12
#                 * sr
#                 / hop
#             ),
#         ),
#     )

#     strong_onsets = [
#         float(times[i])
#         for i in peaks
#         if i < len(times)
#     ]

#     return {
#         "onset_env": onset_env,
#         "onset_times": times,
#         "beats": beat_times,
#         "strong_onsets": strong_onsets,
#     }

# def audio_features(y, sr):
#     """
#     Analyze reference audio.

#     Returns:
#         bpm                  -> detected tempo
#         beat_times           -> musical beat timestamps
#         strong_onset_times   -> strong musical/energy onsets
#         onset_strength       -> onset envelope
#         onset_times          -> timestamps corresponding to onset envelope
#     """

#     # ---------------------------------------------------------
#     # 1. ONSET STRENGTH
#     # ---------------------------------------------------------
#     onset_env = librosa.onset.onset_strength(
#         y=y,
#         sr=sr,
#         hop_length=512
#     )

#     # ---------------------------------------------------------
#     # 2. BEAT TRACKING
#     # ---------------------------------------------------------
#     tempo, beat_frames = librosa.beat.beat_track(
#         y=y,
#         sr=sr,
#         onset_envelope=onset_env,
#         hop_length=512
#     )

#     # librosa versions may return tempo as an array
#     tempo_arr = np.asarray(tempo).reshape(-1)

#     if len(tempo_arr) > 0:
#         bpm = float(tempo_arr[0])
#     else:
#         bpm = 0.0

#     beat_frames = np.asarray(beat_frames).reshape(-1)

#     beat_times = librosa.frames_to_time(
#         beat_frames,
#         sr=sr,
#         hop_length=512
#     )

#     # ---------------------------------------------------------
#     # 3. ONSET TIMESTAMPS
#     # ---------------------------------------------------------
#     onset_times = librosa.frames_to_time(
#         np.arange(len(onset_env)),
#         sr=sr,
#         hop_length=512
#     )

#     # ---------------------------------------------------------
#     # 4. STRONG ONSETS
#     # ---------------------------------------------------------
#     onset_threshold = np.percentile(
#         onset_env,
#         85
#     )

#     min_distance_frames = max(
#         1,
#         int(MIN_EVENT_DISTANCE * sr / 512)
#     )

#     strong_onset_frames, _ = find_peaks(
#         onset_env,
#         height=onset_threshold,
#         distance=min_distance_frames
#     )

#     strong_onset_times = onset_times[
#         strong_onset_frames
#     ]

#     return {
#         "y": y,
#         "sr": sr,
#         "bpm": bpm,
#         "beat_times": beat_times,
#         "strong_onset_times": strong_onset_times,
#         "onset_strength": onset_env,
#         "onset_times": onset_times,
#     }

def audio_features(y, sr):
    """
    Analyze reference audio.

    Keeps compatibility with the existing reference shot mapper.
    """

    # =========================================================
    # 1. ONSET STRENGTH
    # =========================================================

    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=512
    )

    # =========================================================
    # 2. BEAT TRACKING
    # =========================================================

    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        onset_envelope=onset_env,
        hop_length=512
    )

    # librosa may return tempo as numpy array
    tempo_arr = np.asarray(tempo).reshape(-1)

    if len(tempo_arr) > 0:
        bpm = float(tempo_arr[0])
    else:
        bpm = 0.0

    beat_frames = np.asarray(beat_frames).reshape(-1)

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=512
    )

    # =========================================================
    # 3. ONSET TIMELINE
    # =========================================================

    onset_times = librosa.frames_to_time(
        np.arange(len(onset_env)),
        sr=sr,
        hop_length=512
    )

    # =========================================================
    # 4. STRONG ONSETS
    # =========================================================

    onset_threshold = np.percentile(
        onset_env,
        85
    )

    min_distance_frames = max(
        1,
        int(MIN_EVENT_DISTANCE * sr / 512)
    )

    strong_onset_frames, properties = find_peaks(
        onset_env,
        height=onset_threshold,
        distance=min_distance_frames
    )

    strong_onset_times = onset_times[
        strong_onset_frames
    ]

    # =========================================================
    # 5. RETURN
    # =========================================================
    #
    # We intentionally expose both naming conventions because
    # different parts of the current pipeline use different keys.
    #

    return {
        # Raw audio
        "y": y,
        "sr": sr,

        # Tempo
        "bpm": bpm,

        # Beats
        "beat_times": beat_times,
        "beats": beat_times,

        # Strong onsets
        "strong_onset_times": strong_onset_times,
        "strong_onsets": strong_onset_times,

        # Full onset envelope
        "onset_strength": onset_env,
        "onset_times": onset_times,
        "onsets": onset_times,

        # Optional peak metadata
        "strong_onset_frames": strong_onset_frames,
        "strong_onset_heights": properties.get(
            "peak_heights",
            np.array([])
        ),
    }

def nearest_time(
    value,
    times,
):

    if len(times) == 0:
        return None

    arr = np.asarray(
        times,
        dtype=float,
    )

    idx = int(
        np.argmin(
            np.abs(
                arr - value
            )
        )
    )

    return float(
        arr[idx]
    )


# ------------------------------------------------------------
# Frame metrics
# ------------------------------------------------------------

def frame_metrics(frame):

    small = cv2.resize(
        frame,
        (
            SAMPLE_WIDTH,
            SAMPLE_HEIGHT,
        ),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY,
    )

    hsv = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2HSV,
    )

    brightness = float(
        gray.mean()
        / 255.0
    )

    contrast = float(
        gray.std()
        / 255.0
    )

    saturation = float(
        hsv[:, :, 1].mean()
        / 255.0
    )

    edges = cv2.Canny(
        gray,
        70,
        150,
    )

    edge_density = float(
        np.mean(
            edges > 0
        )
    )

    lap = cv2.Laplacian(
        gray,
        cv2.CV_32F,
    )

    sharpness = float(
        min(
            1.0,
            lap.var()
            / 1000.0,
        )
    )

    return {
        "gray": gray,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "edge_density": edge_density,
        "sharpness": sharpness,
    }


def histogram(frame):

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
    )

    return hist


# ------------------------------------------------------------
# Scan reference video
# ------------------------------------------------------------

def scan_reference():

    print()
    print("🎬 SCANNING REFERENCE VIDEO")

    cap = cv2.VideoCapture(
        str(REFERENCE_VIDEO)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open reference:\n"
            f"{REFERENCE_VIDEO}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 30.0

    frames = []
    metrics = []
    histograms = []

    frame_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        timestamp = (
            frame_index
            / fps
        )

        m = frame_metrics(
            frame
        )

        frames.append(
            timestamp
        )

        metrics.append(
            m
        )

        histograms.append(
            histogram(frame)
        )

        frame_index += 1

        if frame_index % 300 == 0:

            print(
                f"  decoded "
                f"{frame_index} frames "
                f"({timestamp:.1f}s)"
            )

    cap.release()

    if len(frames) < 2:
        raise RuntimeError(
            "Reference video produced too few frames."
        )

    return (
        fps,
        frames,
        metrics,
        histograms,
    )


# ------------------------------------------------------------
# Visual changes
# ------------------------------------------------------------

def build_visual_changes(
    metrics,
    histograms,
):

    motion = []
    hist_change = []
    brightness_change = []
    contrast_change = []

    for i in range(
        1,
        len(metrics),
    ):

        previous = metrics[
            i - 1
        ]

        current = metrics[
            i
        ]

        gray_a = previous[
            "gray"
        ].astype(
            np.float32
        )

        gray_b = current[
            "gray"
        ].astype(
            np.float32
        )

        # Frame-to-frame difference.
        frame_diff = float(
            np.mean(
                np.abs(
                    gray_a
                    - gray_b
                )
            )
            / 255.0
        )

        motion.append(
            frame_diff
        )

        # Histogram change.
        bhatta = cv2.compareHist(
            histograms[
                i - 1
            ],
            histograms[
                i
            ],
            cv2.HISTCMP_BHATTACHARYYA,
        )

        hist_change.append(
            float(
                max(
                    0.0,
                    min(
                        1.0,
                        bhatta,
                    ),
                )
            )
        )

        brightness_change.append(
            abs(
                current[
                    "brightness"
                ]
                -
                previous[
                    "brightness"
                ]
            )
        )

        contrast_change.append(
            abs(
                current[
                    "contrast"
                ]
                -
                previous[
                    "contrast"
                ]
            )
        )

    motion = np.asarray(
        motion,
        dtype=np.float32,
    )

    hist_change = np.asarray(
        hist_change,
        dtype=np.float32,
    )

    brightness_change = np.asarray(
        brightness_change,
        dtype=np.float32,
    )

    contrast_change = np.asarray(
        contrast_change,
        dtype=np.float32,
    )

    def robust_norm(x):

        lo = np.percentile(
            x,
            10,
        )

        hi = np.percentile(
            x,
            99,
        )

        return np.clip(
            (
                x - lo
            )
            /
            max(
                hi - lo,
                1e-8,
            ),
            0.0,
            1.0,
        )

    motion_n = robust_norm(
        motion
    )

    hist_n = robust_norm(
        hist_change
    )

    brightness_n = robust_norm(
        brightness_change
    )

    # Combined visual-event score.
    visual_event = (
        motion_n * 0.40
        +
        hist_n * 0.40
        +
        brightness_n * 0.20
    )

    return {
        "motion": motion,
        "hist_change": hist_change,
        "brightness_change":
            brightness_change,
        "contrast_change":
            contrast_change,
        "motion_norm":
            motion_n,
        "hist_norm":
            hist_n,
        "brightness_norm":
            brightness_n,
        "visual_event":
            visual_event,
    }


# ------------------------------------------------------------
# Detect editorial cuts
# ------------------------------------------------------------

def detect_cuts(
    timestamps,
    changes,
):

    score = changes[
        "visual_event"
    ]

    if len(score) < 3:
        return []

    threshold = np.percentile(
        score,
        CUT_PERCENTILE,
    )

    distance = max(
        1,
        int(
            MIN_SHOT_DURATION
            * 29.97
        ),
    )

    peaks, _ = find_peaks(
        score,
        height=threshold,
        prominence=0.08,
        distance=distance,
    )

    cuts = []

    for peak in peaks:

        frame_index = (
            peak + 1
        )

        if frame_index >= len(
            timestamps
        ):
            continue

        cuts.append(
            {
                "time":
                    float(
                        timestamps[
                            frame_index
                        ]
                    ),

                "strength":
                    float(
                        score[
                            peak
                        ]
                    ),

                "motion":
                    float(
                        changes[
                            "motion_norm"
                        ][
                            peak
                        ]
                    ),

                "hist_change":
                    float(
                        changes[
                            "hist_norm"
                        ][
                            peak
                        ]
                    ),

                "brightness_change":
                    float(
                        changes[
                            "brightness_norm"
                        ][
                            peak
                        ]
                    ),
            }
        )

    return cuts


# ------------------------------------------------------------
# Detect visual events
# ------------------------------------------------------------

def detect_events(
    timestamps,
    changes,
):

    score = changes[
        "visual_event"
    ]

    if len(score) == 0:
        return []

    threshold = np.percentile(
        score,
        EVENT_PERCENTILE,
    )

    distance = max(
        1,
        int(
            MIN_EVENT_DISTANCE
            * 29.97
        ),
    )

    peaks, _ = find_peaks(
        score,
        height=threshold,
        prominence=0.045,
        distance=distance,
    )

    events = []

    for peak in peaks:

        frame_index = (
            peak + 1
        )

        if frame_index >= len(
            timestamps
        ):
            continue

        brightness = float(
            changes[
                "brightness_norm"
            ][
                peak
            ]
        )

        motion = float(
            changes[
                "motion_norm"
            ][
                peak
            ]
        )

        hist = float(
            changes[
                "hist_norm"
            ][
                peak
            ]
        )

        if (
            brightness >= 0.75
            and motion >= 0.55
        ):

            event_type = (
                "flash_or_impact"
            )

        elif hist >= 0.75:

            event_type = (
                "structural_change"
            )

        elif motion >= 0.80:

            event_type = (
                "motion_spike"
            )

        else:

            event_type = (
                "visual_change"
            )

        events.append(
            {
                "time":
                    float(
                        timestamps[
                            frame_index
                        ]
                    ),

                "score":
                    float(
                        score[
                            peak
                        ]
                    ),

                "type":
                    event_type,

                "motion":
                    motion,

                "hist_change":
                    hist,

                "brightness_change":
                    brightness,
            }
        )

    return events


# ------------------------------------------------------------
# Build reference shots
# ------------------------------------------------------------

def build_shots(
    timestamps,
    cuts,
    metrics,
    audio,
    fps,
):

    duration = float(
        timestamps[-1]
        + 1.0 / fps
    )

    boundaries = [
        0.0
    ]

    for cut in cuts:

        t = float(
            cut["time"]
        )

        if (
            0.05
            < t
            < duration - 0.05
        ):
            boundaries.append(
                t
            )

    boundaries.append(
        duration
    )

    boundaries = sorted(
        set(
            round(
                x,
                4,
            )
            for x in boundaries
        )
    )

    shots = []

    for i in range(
        len(boundaries) - 1
    ):

        start = boundaries[i]
        end = boundaries[i + 1]

        if (
            end - start
            < MIN_SHOT_DURATION
        ):
            continue

        center = (
            start + end
        ) / 2.0

        frame_index = int(
            min(
                len(metrics) - 1,
                max(
                    0,
                    round(
                        center
                        * fps
                    ),
                ),
            )
        )

        m = metrics[
            frame_index
        ]

        nearest_onset = nearest_time(
            center,
            audio[
                "strong_onsets"
            ],
        )

        nearest_beat = nearest_time(
            center,
            audio[
                "beats"
            ],
        )

        onset_distance = (
            abs(
                center
                - nearest_onset
            )
            if nearest_onset
            is not None
            else None
        )

        beat_distance = (
            abs(
                center
                - nearest_beat
            )
            if nearest_beat
            is not None
            else None
        )

        cut_info = None

        for cut in cuts:

            if abs(
                cut["time"]
                - start
            ) < 0.05:

                cut_info = cut
                break

        shots.append(
            {
                "shot_id":
                    len(shots) + 1,

                "start":
                    round(
                        start,
                        4,
                    ),

                "end":
                    round(
                        end,
                        4,
                    ),

                "duration":
                    round(
                        end - start,
                        4,
                    ),

                "brightness":
                    round(
                        m[
                            "brightness"
                        ],
                        4,
                    ),

                "contrast":
                    round(
                        m[
                            "contrast"
                        ],
                        4,
                    ),

                "saturation":
                    round(
                        m[
                            "saturation"
                        ],
                        4,
                    ),

                "edge_density":
                    round(
                        m[
                            "edge_density"
                        ],
                        4,
                    ),

                "sharpness":
                    round(
                        m[
                            "sharpness"
                        ],
                        4,
                    ),

                "cut_strength":
                    round(
                        (
                            cut_info[
                                "strength"
                            ]
                            if cut_info
                            else 0.0
                        ),
                        4,
                    ),

                "cut_motion":
                    round(
                        (
                            cut_info[
                                "motion"
                            ]
                            if cut_info
                            else 0.0
                        ),
                        4,
                    ),

                "cut_hist_change":
                    round(
                        (
                            cut_info[
                                "hist_change"
                            ]
                            if cut_info
                            else 0.0
                        ),
                        4,
                    ),

                "cut_brightness_change":
                    round(
                        (
                            cut_info[
                                "brightness_change"
                            ]
                            if cut_info
                            else 0.0
                        ),
                        4,
                    ),

                "nearest_strong_onset":
                    (
                        round(
                            nearest_onset,
                            4,
                        )
                        if nearest_onset
                        is not None
                        else None
                    ),

                "onset_distance":
                    (
                        round(
                            onset_distance,
                            4,
                        )
                        if onset_distance
                        is not None
                        else None
                    ),

                "nearest_beat":
                    (
                        round(
                            nearest_beat,
                            4,
                        )
                        if nearest_beat
                        is not None
                        else None
                    ),

                "beat_distance":
                    (
                        round(
                            beat_distance,
                            4,
                        )
                        if beat_distance
                        is not None
                        else None
                    ),
            }
        )

    return shots


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print()
    print("=" * 82)
    print(
        "🎬 REFERENCE SHOT MAPPER"
    )
    print("=" * 82)

    if not REFERENCE_VIDEO.exists():

        raise FileNotFoundError(
            f"Missing reference video:\n"
            f"{REFERENCE_VIDEO}"
        )

    if not REFERENCE_AUDIO.exists():

        raise FileNotFoundError(
            f"Missing reference audio:\n"
            f"{REFERENCE_AUDIO}"
        )

    (
        fps,
        timestamps,
        metrics,
        histograms,
    ) = scan_reference()

    print()
    print(
        f"FPS                 : "
        f"{fps:.3f}"
    )

    print(
        f"Frames              : "
        f"{len(timestamps)}"
    )

    print(
        f"Duration            : "
        f"{timestamps[-1]:.3f}s"
    )

    # --------------------------------------------------------
    # Visual analysis
    # --------------------------------------------------------

    changes = build_visual_changes(
        metrics,
        histograms,
    )

    cuts = detect_cuts(
        timestamps,
        changes,
    )

    events = detect_events(
        timestamps,
        changes,
    )

    # --------------------------------------------------------
    # Audio analysis
    # --------------------------------------------------------

    y, sr = load_audio()

    audio = audio_features(
        y,
        sr,
    )

    # librosa current API.
    tempo_values = librosa.feature.tempo(
        y=y,
        sr=sr,
    )

    if np.ndim(
        tempo_values
    ) == 0:

        bpm = float(
            tempo_values
        )

    else:

        bpm = float(
            np.asarray(
                tempo_values
            ).reshape(-1)[0]
        )

    # --------------------------------------------------------
    # Build shots.
    # --------------------------------------------------------

    shots = build_shots(
        timestamps,
        cuts,
        metrics,
        audio,
        fps,
    )

    # --------------------------------------------------------
    # Attach visual events to shots.
    # --------------------------------------------------------

    for shot in shots:

        relevant_events = [
            event
            for event in events
            if (
                shot["start"]
                <= event["time"]
                <= shot["end"]
            )
        ]

        shot[
            "visual_event_count"
        ] = len(
            relevant_events
        )

        if relevant_events:

            strongest = max(
                relevant_events,
                key=lambda x:
                    x["score"],
            )

            shot[
                "strongest_visual_event"
            ] = round(
                strongest["time"],
                4,
            )

            shot[
                "strongest_event_type"
            ] = strongest[
                "type"
            ]

            shot[
                "strongest_event_score"
            ] = round(
                strongest["score"],
                4,
            )

        else:

            shot[
                "strongest_visual_event"
            ] = None

            shot[
                "strongest_event_type"
            ] = None

            shot[
                "strongest_event_score"
            ] = 0.0

    # --------------------------------------------------------
    # Summary statistics.
    # --------------------------------------------------------

    shot_durations = np.asarray(
        [
            shot["duration"]
            for shot in shots
        ],
        dtype=float,
    )

    if len(
        shot_durations
    ):

        median_shot = float(
            np.median(
                shot_durations
            )
        )

        mean_shot = float(
            np.mean(
                shot_durations
            )
        )

        shortest_shot = float(
            np.min(
                shot_durations
            )
        )

        longest_shot = float(
            np.max(
                shot_durations
            )
        )

    else:

        median_shot = 0.0
        mean_shot = 0.0
        shortest_shot = 0.0
        longest_shot = 0.0

    result = {

        "project":
            "AMV Director",

        "version":
            "reference-shot-map-v1",

        "reference_video":
            str(
                REFERENCE_VIDEO
            ),

        "reference_audio":
            str(
                REFERENCE_AUDIO
            ),

        "duration":
            round(
                timestamps[-1],
                4,
            ),

        "fps":
            round(
                fps,
                4,
            ),

        "frame_count":
            len(
                timestamps
            ),

        "cut_count":
            len(cuts),

        "event_count":
            len(events),

        "shot_count":
            len(shots),

        "shot_statistics":
            {
                "mean_duration":
                    round(
                        mean_shot,
                        4,
                    ),

                "median_duration":
                    round(
                        median_shot,
                        4,
                    ),

                "shortest_duration":
                    round(
                        shortest_shot,
                        4,
                    ),

                "longest_duration":
                    round(
                        longest_shot,
                        4,
                    ),
            },

        "audio":
            {
                "bpm":
                    round(
                        bpm,
                        4,
                    ),

                "beat_count":
                    len(
                        audio[
                            "beats"
                        ]
                    ),

                "strong_onset_count":
                    len(
                        audio[
                            "strong_onsets"
                        ]
                    ),
            },

        "cuts":
            cuts,

        "events":
            events,

        "shots":
            shots,
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
            result,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 82)
    print(
        "✅ REFERENCE SHOT MAP COMPLETE"
    )
    print("=" * 82)

    print(
        f"Visual cuts detected : "
        f"{len(cuts)}"
    )

    print(
        f"Visual events        : "
        f"{len(events)}"
    )

    print(
        f"Editorial shots      : "
        f"{len(shots)}"
    )

    print(
        f"Audio BPM            : "
        f"{bpm:.2f}"
    )

    print(
        f"Audio beats          : "
        f"{len(audio['beats'])}"
    )

    print(
        f"Strong audio onsets  : "
        f"{len(audio['strong_onsets'])}"
    )

    print(
        f"Mean shot duration   : "
        f"{mean_shot:.3f}s"
    )

    print(
        f"Median shot duration : "
        f"{median_shot:.3f}s"
    )

    print(
        f"Shortest shot        : "
        f"{shortest_shot:.3f}s"
    )

    print(
        f"Longest shot         : "
        f"{longest_shot:.3f}s"
    )

    print()
    print(
        "FIRST 25 REFERENCE SHOTS"
    )

    for shot in shots[:25]:

        print(
            f"  "
            f"#{shot['shot_id']:02d} "
            f"{shot['start']:.3f}"
            f"→"
            f"{shot['end']:.3f}s "
            f"dur="
            f"{shot['duration']:.3f}s "
            f"cut="
            f"{shot['cut_strength']:.2f} "
            f"events="
            f"{shot['visual_event_count']} "
            f"onset="
            f"{shot['onset_distance']}"
        )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT
    )

    print(
        "=" * 82
    )


if __name__ == "__main__":
    main()
