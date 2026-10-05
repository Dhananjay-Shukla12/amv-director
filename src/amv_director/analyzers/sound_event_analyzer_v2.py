from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Any

import librosa
import numpy as np
import torch

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

AUDIO_PATH = (
    BASE_DIR
    / "data/outputs/reference_audio.wav"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data/outputs/reference_sound_events_v2.json"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = (
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

SAMPLE_RATE = 16000


# ============================================================
# EVENT WINDOWS
# ============================================================

WINDOW_SECONDS = 0.96

# Minimum target score for a semantic SFX candidate.
MIN_TARGET_SCORE = 0.035

# Stronger threshold for high-confidence event.
STRONG_TARGET_SCORE = 0.15

# Nearby detections of the same semantic event are merged.
MERGE_DISTANCE = 0.35


# ============================================================
# TARGET AUDIO EVENT VOCABULARY
# ============================================================

TARGET_EVENTS = {

    "gunshot": [
        "gunshot",
        "gun fire",
        "gunfire",
        "firearm",
        "pistol",
        "rifle",
        "shotgun",
        "machine gun",
        "shooting",
        "artillery fire",
        "gun",
    ],

    "explosion": [
        "explosion",
        "blast",
        "boom",
        "burst",
        "firecracker",
    ],

    "weapon": [
        "weapon",
        "sword",
        "blade",
        "knife",
        "dagger",
        "saber",
    ],

    "weapon_clash": [
        "sword clash",
        "metal",
        "clang",
        "clash",
        "clanking",
    ],

    "impact": [
        "impact",
        "thump",
        "thud",
        "slam",
        "smash",
        "punch",
        "knock",
        "hit",
        "crash",
    ],

    "whoosh": [
        "whoosh",
        "swish",
        "whiz",
        "whizz",
        "woosh",
    ],

    "scream": [
        "scream",
        "screaming",
        "shriek",
        "shout",
        "yell",
        "crying",
    ],

    "footsteps": [
        "footstep",
        "footsteps",
        "walking",
        "running",
        "run",
    ],

    "glass": [
        "glass",
        "breaking",
        "shatter",
        "smash",
    ],

    "door": [
        "door",
        "door slam",
        "knocking",
    ],

    "crowd": [
        "crowd",
        "cheering",
        "applause",
    ],

    "alarm": [
        "alarm",
        "siren",
        "beep",
        "ringtone",
    ],
}


# ============================================================
# LABELS THAT SHOULD NEVER BECOME SFX EVENTS
# ============================================================

IGNORED_LABEL_KEYWORDS = [
    "music",
    "singing",
    "musical instrument",
    "background music",
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:

        return default


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:

    return max(
        low,
        min(high, value),
    )


def get_device() -> torch.device:

    if torch.backends.mps.is_available():

        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio():

    if not AUDIO_PATH.exists():

        raise FileNotFoundError(
            f"Audio not found:\n{AUDIO_PATH}"
        )

    print()
    print("Loading reference audio...")

    audio, sr = librosa.load(
        str(AUDIO_PATH),
        sr=SAMPLE_RATE,
        mono=True,
    )

    audio = audio.astype(
        np.float32
    )

    print(
        f"✅ Duration: "
        f"{len(audio) / SAMPLE_RATE:.3f}s"
    )

    return audio


# ============================================================
# AUDIO TRANSIENTS
# ============================================================

def detect_transients(
    audio: np.ndarray,
) -> List[float]:

    print()
    print(
        "Detecting transient candidates..."
    )

    # Separate harmonic music from percussive material.
    _, percussive = librosa.effects.hpss(
        audio
    )

    onset_env = librosa.onset.onset_strength(
        y=percussive,
        sr=SAMPLE_RATE,
        hop_length=256,
    )

    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=SAMPLE_RATE,
        hop_length=256,
        units="frames",
        backtrack=False,
        pre_max=5,
        post_max=5,
        pre_avg=10,
        post_avg=10,
        delta=0.20,
        wait=3,
    )

    times = librosa.frames_to_time(
        frames,
        sr=SAMPLE_RATE,
        hop_length=256,
    )

    raw = [
        float(x)
        for x in times
        if x >= 0
    ]

    # Merge very close transient detections.
    merged = []

    for timestamp in raw:

        if not merged:

            merged.append(
                timestamp
            )

            continue

        if (
            timestamp
            - merged[-1]
            < 0.10
        ):

            # Keep the stronger/later event center.
            merged[-1] = (
                merged[-1]
                + timestamp
            ) / 2.0

        else:

            merged.append(
                timestamp
            )

    print(
        f"Raw transients    : {len(raw)}"
    )

    print(
        f"Merged candidates : {len(merged)}"
    )

    return merged


# ============================================================
# MODEL
# ============================================================

def load_model():

    device = get_device()

    print()
    print("=" * 72)
    print("🔊 SOUND EVENT CLASSIFIER V2")
    print("=" * 72)

    print()
    print(
        f"Model  : {MODEL_NAME}"
    )

    print(
        f"Device : {device}"
    )

    print()
    print(
        "Loading AudioSet AST..."
    )

    extractor = (
        AutoFeatureExtractor
        .from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModelForAudioClassification
        .from_pretrained(
            MODEL_NAME
        )
    )

    model.to(device)
    model.eval()

    print(
        "✅ Model loaded"
    )

    return (
        model,
        extractor,
        device,
    )


# ============================================================
# LABEL MAP
# ============================================================

def get_label_map(
    model,
) -> Dict[int, str]:

    result = {}

    for key, value in (
        model.config.id2label.items()
    ):

        try:

            key = int(key)

        except Exception:

            pass

        result[key] = str(
            value
        )

    return result


# ============================================================
# FIND AUDIOSET LABELS
# ============================================================

def build_target_label_map(
    label_map: Dict[int, str],
) -> Dict[str, List[int]]:

    result = {}

    print()
    print(
        "Searching AudioSet vocabulary..."
    )

    for event_name, keywords in (
        TARGET_EVENTS.items()
    ):

        matches = []

        for index, label in (
            label_map.items()
        ):

            label_lower = (
                label.lower()
            )

            for keyword in keywords:

                if (
                    keyword.lower()
                    in label_lower
                ):

                    matches.append(
                        index
                    )

                    break

        result[event_name] = sorted(
            set(matches)
        )

        print(
            f"  {event_name:15s}: "
            f"{len(result[event_name])} labels"
        )

    return result


# ============================================================
# WINDOW
# ============================================================

def extract_window(
    audio: np.ndarray,
    timestamp: float,
) -> np.ndarray:

    half = (
        WINDOW_SECONDS / 2.0
    )

    start = max(
        0.0,
        timestamp - half,
    )

    end = min(
        len(audio) / SAMPLE_RATE,
        timestamp + half,
    )

    start_sample = int(
        start * SAMPLE_RATE
    )

    end_sample = int(
        end * SAMPLE_RATE
    )

    chunk = audio[
        start_sample:end_sample
    ]

    if len(chunk) < 1000:

        return np.zeros(
            int(
                SAMPLE_RATE
                * WINDOW_SECONDS
            ),
            dtype=np.float32,
        )

    return chunk.astype(
        np.float32
    )


# ============================================================
# AUDIOSET PROBABILITY
# ============================================================

def get_probabilities(
    logits: torch.Tensor,
    model,
) -> torch.Tensor:

    problem_type = (
        model.config.problem_type
    )

    if (
        problem_type
        == "multi_label_classification"
    ):

        return torch.sigmoid(
            logits
        )

    return torch.softmax(
        logits,
        dim=-1,
    )


# ============================================================
# MODEL ANALYSIS
# ============================================================

def predict(
    audio_chunk: np.ndarray,
    model,
    extractor,
    device,
) -> torch.Tensor:

    inputs = extractor(
        audio_chunk,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(
            **inputs
        )

        probs = get_probabilities(
            outputs.logits,
            model,
        )[0]

    return probs.detach()


# ============================================================
# SCORE TARGET EVENTS
# ============================================================

def score_target_events(
    probabilities: torch.Tensor,
    target_labels: Dict[str, List[int]],
    label_map: Dict[int, str],
) -> Dict:

    event_scores = {}

    evidence = {}

    for event_name, indices in (
        target_labels.items()
    ):

        if not indices:

            event_scores[
                event_name
            ] = 0.0

            evidence[
                event_name
            ] = []

            continue

        values = []

        for index in indices:

            if (
                index < 0
                or index
                >= len(probabilities)
            ):

                continue

            score = float(
                probabilities[
                    index
                ].item()
            )

            values.append(
                (
                    score,
                    index,
                )
            )

        values.sort(
            reverse=True
        )

        best_score = (
            values[0][0]
            if values
            else 0.0
        )

        event_scores[
            event_name
        ] = best_score

        evidence[
            event_name
        ] = [

            {
                "label": label_map.get(
                    index,
                    f"class_{index}",
                ),

                "confidence": round(
                    score,
                    5,
                ),
            }

            for score, index
            in values[:5]
        ]

    return {
        "scores": event_scores,
        "evidence": evidence,
    }


# ============================================================
# MUSIC / VOICE BACKGROUND SCORES
# ============================================================

def background_scores(
    probabilities: torch.Tensor,
    label_map: Dict[int, str],
) -> Dict[str, float]:

    music_score = 0.0
    voice_score = 0.0

    for index, label in (
        label_map.items()
    ):

        if (
            index < 0
            or index >= len(probabilities)
        ):
            continue

        score = float(
            probabilities[index].item()
        )

        label_lower = (
            label.lower()
        )

        if (
            "music"
            in label_lower
            or "singing"
            in label_lower
        ):

            music_score = max(
                music_score,
                score,
            )

        if (
            "speech"
            in label_lower
            or "voice"
            in label_lower
            or "conversation"
            in label_lower
        ):

            voice_score = max(
                voice_score,
                score,
            )

    return {
        "music": music_score,
        "voice": voice_score,
    }


# ============================================================
# ANALYZE ONE TRANSIENT
# ============================================================

def analyze_transient(
    timestamp: float,
    full_audio: np.ndarray,
    percussive_audio: np.ndarray,
    model,
    extractor,
    device,
    target_labels,
    label_map,
) -> Dict:

    full_chunk = extract_window(
        full_audio,
        timestamp,
    )

    percussive_chunk = extract_window(
        percussive_audio,
        timestamp,
    )

    full_probs = predict(
        full_chunk,
        model,
        extractor,
        device,
    )

    percussive_probs = predict(
        percussive_chunk,
        model,
        extractor,
        device,
    )

    full_targets = (
        score_target_events(
            full_probs,
            target_labels,
            label_map,
        )
    )

    percussive_targets = (
        score_target_events(
            percussive_probs,
            target_labels,
            label_map,
        )
    )

    full_background = (
        background_scores(
            full_probs,
            label_map,
        )
    )

    percussive_background = (
        background_scores(
            percussive_probs,
            label_map,
        )
    )

    # --------------------------------------------------------
    # Combine full-mix and percussive evidence.
    #
    # The percussive track gets extra weight because we are
    # specifically looking for short SFX hidden underneath
    # music.
    # --------------------------------------------------------

    combined_scores = {}

    combined_evidence = {}

    for event_name in TARGET_EVENTS:

        full_score = (
            full_targets[
                "scores"
            ].get(
                event_name,
                0.0,
            )
        )

        perc_score = (
            percussive_targets[
                "scores"
            ].get(
                event_name,
                0.0,
            )
        )

        combined = max(
            full_score,
            perc_score * 1.20,
        )

        combined_scores[
            event_name
        ] = clamp(
            combined
        )

        combined_evidence[
            event_name
        ] = {
            "combined_score":
                round(
                    combined,
                    5,
                ),

            "full_mix_score":
                round(
                    full_score,
                    5,
                ),

            "percussive_score":
                round(
                    perc_score,
                    5,
                ),

            "full_mix_evidence":
                full_targets[
                    "evidence"
                ].get(
                    event_name,
                    [],
                ),

            "percussive_evidence":
                percussive_targets[
                    "evidence"
                ].get(
                    event_name,
                    [],
                ),
        }

    # --------------------------------------------------------
    # Best event
    # --------------------------------------------------------

    best_event = None
    best_score = 0.0

    for name, score in (
        combined_scores.items()
    ):

        if score > best_score:

            best_event = name
            best_score = score

    # --------------------------------------------------------
    # Only call it an event above threshold.
    # --------------------------------------------------------

    detected = (
        best_event is not None
        and best_score
        >= MIN_TARGET_SCORE
    )

    if not detected:

        best_event = None

    return {

        "timestamp": round(
            timestamp,
            4,
        ),

        "window_start": round(
            max(
                0.0,
                timestamp
                - WINDOW_SECONDS / 2,
            ),
            4,
        ),

        "window_end": round(
            timestamp
            + WINDOW_SECONDS / 2,
            4,
        ),

        "detected_event":
            best_event,

        "event_confidence":
            round(
                best_score,
                5,
            ),

        "strength":
            (
                "strong"
                if best_score
                >= STRONG_TARGET_SCORE
                else
                "candidate"
                if detected
                else
                "none"
            ),

        "background": {

            "music":
                round(
                    max(
                        full_background[
                            "music"
                        ],
                        percussive_background[
                            "music"
                        ],
                    ),
                    5,
                ),

            "voice":
                round(
                    max(
                        full_background[
                            "voice"
                        ],
                        percussive_background[
                            "voice"
                        ],
                    ),
                    5,
                ),
        },

        "event_scores": {
            name: round(
                score,
                5,
            )

            for name, score
            in combined_scores.items()
        },

        "evidence":
            combined_evidence,
    }


# ============================================================
# MERGE EVENTS
# ============================================================

def merge_events(
    analyses: List[Dict],
) -> List[Dict]:

    candidates = [

        item
        for item in analyses
        if item.get(
            "detected_event"
        ) is not None

    ]

    candidates.sort(
        key=lambda item: (
            item["detected_event"],
            item["timestamp"],
        )
    )

    merged = []

    for candidate in candidates:

        if not merged:

            merged.append(
                candidate
            )

            continue

        previous = merged[-1]

        same_type = (
            candidate[
                "detected_event"
            ]
            ==
            previous[
                "detected_event"
            ]
        )

        close = (
            candidate[
                "timestamp"
            ]
            -
            previous[
                "timestamp"
            ]
            < MERGE_DISTANCE
        )

        if same_type and close:

            if (
                candidate[
                    "event_confidence"
                ]
                >
                previous[
                    "event_confidence"
                ]
            ):

                merged[-1] = (
                    candidate
                )

        else:

            merged.append(
                candidate
            )

    merged.sort(
        key=lambda item: item[
            "timestamp"
        ]
    )

    return merged


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("🔊 SOUND EVENT ANALYZER V2")
    print("=" * 72)

    audio = load_audio()

    # --------------------------------------------------------
    # HPSS
    # --------------------------------------------------------

    print()
    print(
        "Separating percussive audio..."
    )

    _, percussive = (
        librosa.effects.hpss(
            audio
        )
    )

    print(
        "✅ Percussive component ready"
    )

    # --------------------------------------------------------
    # Transients
    # --------------------------------------------------------

    timestamps = (
        detect_transients(
            audio
        )
    )

    if not timestamps:

        raise RuntimeError(
            "No transient candidates detected."
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, extractor, device = (
        load_model()
    )

    label_map = get_label_map(
        model
    )

    print()
    print(
        f"AudioSet labels: "
        f"{len(label_map)}"
    )

    # --------------------------------------------------------
    # Build target label lookup
    # --------------------------------------------------------

    target_labels = (
        build_target_label_map(
            label_map
        )
    )

    print()
    print(
        "Analyzing SFX candidates..."
    )

    analyses = []

    total = len(
        timestamps
    )

    for index, timestamp in enumerate(
        timestamps,
        start=1,
    ):

        print(
            f"[{index:03d}/{total:03d}] "
            f"{timestamp:7.3f}s ... ",
            end="",
            flush=True,
        )

        try:

            result = (
                analyze_transient(
                    timestamp=timestamp,
                    full_audio=audio,
                    percussive_audio=percussive,
                    model=model,
                    extractor=extractor,
                    device=device,
                    target_labels=target_labels,
                    label_map=label_map,
                )
            )

            analyses.append(
                result
            )

            event = result[
                "detected_event"
            ]

            confidence = result[
                "event_confidence"
            ]

            if event:

                print(
                    f"{event:15s} "
                    f"{confidence:.3f}"
                )

            else:

                print(
                    "no target SFX"
                )

        except Exception as exc:

            print(
                f"ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    events = merge_events(
        analyses
    )

    # --------------------------------------------------------
    # Event counts
    # --------------------------------------------------------

    event_counts = {}

    for event in events:

        name = event[
            "detected_event"
        ]

        event_counts[name] = (
            event_counts.get(
                name,
                0,
            )
            + 1
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {

        "metadata": {

            "engine":
                "sound_event_analyzer_v2",

            "model":
                MODEL_NAME,

            "device":
                str(device),

            "duration":
                round(
                    len(audio)
                    / SAMPLE_RATE,
                    4,
                ),

            "transient_candidates":
                len(timestamps),

            "windows_analyzed":
                len(analyses),

            "semantic_events":
                len(events),

            "target_events":
                list(
                    TARGET_EVENTS.keys()
                ),
        },

        "summary": {

            "event_counts":
                event_counts,

            "strong_events":
                sum(
                    1
                    for item in events
                    if item[
                        "strength"
                    ]
                    == "strong"
                ),
        },

        "events":
            events,

        "window_analyses":
            analyses,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
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
    print("=" * 72)
    print(
        "✅ SOUND EVENT ANALYSIS V2 COMPLETE"
    )
    print("=" * 72)

    print()

    print(
        f"Transient candidates : "
        f"{len(timestamps)}"
    )

    print(
        f"Windows analyzed    : "
        f"{len(analyses)}"
    )

    print(
        f"Semantic SFX events : "
        f"{len(events)}"
    )

    print()

    print(
        "Detected events:"
    )

    if event_counts:

        for name, count in sorted(
            event_counts.items(),
            key=lambda x: -x[1],
        ):

            print(
                f"  {name:15s}: "
                f"{count}"
            )

    else:

        print(
            "  NONE"
        )

    print()

    print(
        "Strong events:"
    )

    for event in [
        x
        for x in events
        if x[
            "strength"
        ] == "strong"
    ][:25]:

        print(
            f"  "
            f"{event['timestamp']:7.3f}s | "
            f"{event['detected_event']:15s} | "
            f"{event['event_confidence']:.3f}"
        )

    print()

    print(
        "Output:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()