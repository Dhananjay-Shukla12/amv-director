from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

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
    / "data/outputs/reference_sound_events.json"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = (
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

SAMPLE_RATE = 16000

# Analyze short windows around detected audio transients.
WINDOW_SECONDS = 0.96
HOP_SECONDS = 0.48

# Only keep reasonably strong model predictions.
MIN_CONFIDENCE = 0.20

# Number of AudioSet labels to preserve per analyzed window.
TOP_K = 12


# ============================================================
# SEMANTIC EVENT GROUPS
# ============================================================

# We search the model's actual AudioSet labels by keywords.
# This avoids hardcoding ontology wording too aggressively.

EVENT_KEYWORDS = {

    "gunshot": [
        "gun",
        "gunshot",
        "gunfire",
        "firearm",
        "machine gun",
        "artillery",
        "rifle",
        "pistol",
        "shotgun",
    ],

    "explosion": [
        "explosion",
        "blast",
        "boom",
    ],

    "scream": [
        "scream",
        "shriek",
        "yell",
        "shout",
    ],

    "footsteps": [
        "footstep",
        "walking",
        "running",
        "walk",
    ],

    "impact": [
        "thump",
        "thud",
        "impact",
        "slam",
        "smash",
        "punch",
        "knock",
        "hit",
    ],

    "sword": [
        "sword",
        "blade",
        "knife",
        "weapon",
        "clash",
    ],

    "whoosh": [
        "whoosh",
        "swish",
        "whiz",
        "woosh",
    ],

    "glass": [
        "glass",
        "breaking",
        "shatter",
    ],

    "door": [
        "door",
        "knock",
        "slam",
    ],

    "crowd": [
        "crowd",
        "applause",
        "cheering",
    ],

    "voice": [
        "speech",
        "conversation",
        "voice",
        "male speech",
        "female speech",
    ],

    "music": [
        "music",
        "musical",
        "singing",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def safe_float(
    value,
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


# ============================================================
# DEVICE
# ============================================================

def get_device() -> torch.device:

    if torch.backends.mps.is_available():

        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio() -> np.ndarray:

    if not AUDIO_PATH.exists():

        raise FileNotFoundError(
            f"Reference audio not found:\n{AUDIO_PATH}"
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
        f"✅ Audio loaded"
    )

    print(
        f"   Duration: "
        f"{len(audio) / SAMPLE_RATE:.3f}s"
    )

    print(
        f"   Sample rate: "
        f"{SAMPLE_RATE}"
    )

    return audio


# ============================================================
# AUDIO TRANSIENT DETECTION
# ============================================================

def detect_transient_times(
    audio: np.ndarray,
) -> List[float]:

    print()
    print(
        "Detecting audio transients..."
    )

    # Harmonic/percussive separation makes this stage more
    # sensitive to short percussive sound effects.
    _, percussive = librosa.effects.hpss(
        audio
    )

    onset_env = librosa.onset.onset_strength(
        y=percussive,
        sr=SAMPLE_RATE,
        hop_length=256,
    )

    # Conservative onset detection.
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=SAMPLE_RATE,
        hop_length=256,
        backtrack=False,
        units="frames",
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

    raw_times = [
        float(x)
        for x in times
        if x >= 0
    ]

    # --------------------------------------------------------
    # Merge nearby detections.
    #
    # A gunshot/explosion can create multiple nearby spectral
    # peaks. We want one candidate event rather than 4.
    # --------------------------------------------------------

    merged = []

    for timestamp in raw_times:

        if not merged:

            merged.append(
                timestamp
            )

            continue

        if (
            timestamp
            - merged[-1]
            < 0.12
        ):

            # Keep midpoint of cluster.
            merged[-1] = (
                merged[-1]
                + timestamp
            ) * 0.5

        else:

            merged.append(
                timestamp
            )

    print(
        f"✅ Raw transients : "
        f"{len(raw_times)}"
    )

    print(
        f"✅ Merged events  : "
        f"{len(merged)}"
    )

    return merged


# ============================================================
# MODEL
# ============================================================

def load_model():

    device = get_device()

    print()
    print("=" * 72)
    print("🔊 AUDIO EVENT CLASSIFIER")
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
# LABEL ACCESS
# ============================================================

def get_label_map(
    model,
) -> Dict[int, str]:

    id2label = (
        model.config.id2label
    )

    result = {}

    for key, value in id2label.items():

        try:
            index = int(key)
        except Exception:
            index = key

        result[index] = str(
            value
        )

    return result


# ============================================================
# MATCH LABEL TO SEMANTIC GROUP
# ============================================================

def semantic_groups_for_label(
    label: str,
) -> List[str]:

    label_lower = label.lower()

    matches = []

    for group, keywords in (
        EVENT_KEYWORDS.items()
    ):

        for keyword in keywords:

            if keyword in label_lower:

                matches.append(
                    group
                )

                break

    return matches


# ============================================================
# MODEL PROBABILITY
# ============================================================

def model_probabilities(
    logits: torch.Tensor,
    model,
) -> torch.Tensor:

    problem_type = (
        model.config.problem_type
    )

    if problem_type == (
        "multi_label_classification"
    ):

        return torch.sigmoid(
            logits
        )

    return torch.softmax(
        logits,
        dim=-1,
    )


# ============================================================
# ANALYZE AUDIO WINDOW
# ============================================================

def analyze_window(
    audio: np.ndarray,
    timestamp: float,
    model,
    extractor,
    device,
    label_map,
) -> Dict:

    half_window = (
        WINDOW_SECONDS / 2
    )

    start = max(
        0.0,
        timestamp - half_window,
    )

    end = min(
        len(audio) / SAMPLE_RATE,
        timestamp + half_window,
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

        return {
            "timestamp": timestamp,
            "status": "too_short",
            "predictions": [],
        }

    inputs = extractor(
        chunk,
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

        probabilities = (
            model_probabilities(
                outputs.logits,
                model,
            )[0]
        )

    k = min(
        TOP_K,
        probabilities.shape[-1],
    )

    top_values, top_indices = (
        torch.topk(
            probabilities,
            k=k,
        )
    )

    predictions = []

    for value, index in zip(
        top_values.tolist(),
        top_indices.tolist(),
    ):

        label = label_map.get(
            int(index),
            f"class_{index}",
        )

        value = float(value)

        groups = (
            semantic_groups_for_label(
                label
            )
        )

        predictions.append(
            {
                "label": label,
                "confidence": round(
                    value,
                    5,
                ),
                "semantic_groups": groups,
            }
        )

    # --------------------------------------------------------
    # Aggregate event groups.
    # --------------------------------------------------------

    group_scores = {}

    for prediction in predictions:

        confidence = (
            prediction["confidence"]
        )

        for group in prediction[
            "semantic_groups"
        ]:

            group_scores[group] = max(
                group_scores.get(
                    group,
                    0.0,
                ),
                confidence,
            )

    return {
        "timestamp": round(
            timestamp,
            4,
        ),

        "window_start": round(
            start,
            4,
        ),

        "window_end": round(
            end,
            4,
        ),

        "status": "success",

        "group_scores": {
            key: round(
                value,
                5,
            )
            for key, value
            in group_scores.items()
        },

        "predictions": predictions,
    }


# ============================================================
# EVENT CREATION
# ============================================================

def create_semantic_events(
    analyses: List[Dict],
) -> List[Dict]:

    candidates = []

    for analysis in analyses:

        if (
            analysis.get("status")
            != "success"
        ):
            continue

        group_scores = analysis.get(
            "group_scores",
            {},
        )

        for event_name, confidence in (
            group_scores.items()
        ):

            if (
                confidence
                < MIN_CONFIDENCE
            ):

                continue

            candidates.append(
                {
                    "event": event_name,
                    "time": safe_float(
                        analysis[
                            "timestamp"
                        ]
                    ),
                    "confidence": safe_float(
                        confidence
                    ),
                    "window_start":
                        analysis[
                            "window_start"
                        ],
                    "window_end":
                        analysis[
                            "window_end"
                        ],
                }
            )

    # --------------------------------------------------------
    # Merge nearby detections belonging to same event.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: (
            x["event"],
            x["time"],
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

        same_event = (
            previous["event"]
            == candidate["event"]
        )

        close = (
            candidate["time"]
            - previous["time"]
            < 0.30
        )

        if same_event and close:

            # Keep strongest detection.
            if (
                candidate[
                    "confidence"
                ]
                > previous[
                    "confidence"
                ]
            ):

                merged[-1] = candidate

        else:

            merged.append(
                candidate
            )

    # --------------------------------------------------------
    # Final ordering by time.
    # --------------------------------------------------------

    merged.sort(
        key=lambda x: x["time"]
    )

    return merged


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("🔊 SOUND EVENT ANALYZER")
    print("=" * 72)

    audio = load_audio()

    transient_times = (
        detect_transient_times(
            audio
        )
    )

    if not transient_times:

        raise RuntimeError(
            "No audio transients detected."
        )

    model, extractor, device = (
        load_model()
    )

    label_map = get_label_map(
        model
    )

    print()
    print(
        f"AudioSet labels loaded: "
        f"{len(label_map)}"
    )

    print()
    print(
        "Analyzing transient windows..."
    )

    analyses = []

    for index, timestamp in enumerate(
        transient_times,
        start=1,
    ):

        print(
            f"[{index:03d}/"
            f"{len(transient_times):03d}] "
            f"{timestamp:7.3f}s ... ",
            end="",
            flush=True,
        )

        try:

            result = analyze_window(
                audio=audio,
                timestamp=timestamp,
                model=model,
                extractor=extractor,
                device=device,
                label_map=label_map,
            )

            analyses.append(
                result
            )

            groups = result.get(
                "group_scores",
                {},
            )

            if groups:

                top_group = max(
                    groups,
                    key=groups.get,
                )

                score = groups[
                    top_group
                ]

                print(
                    f"{top_group:12s} "
                    f"{score:.2f}"
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
    # Semantic events
    # --------------------------------------------------------

    events = create_semantic_events(
        analyses
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    event_counts = {}

    for event in events:

        name = event[
            "event"
        ]

        event_counts[name] = (
            event_counts.get(
                name,
                0,
            )
            + 1
        )

    output = {

        "metadata": {

            "model": MODEL_NAME,

            "sample_rate": SAMPLE_RATE,

            "duration": round(
                len(audio)
                / SAMPLE_RATE,
                4,
            ),

            "transient_candidates": len(
                transient_times
            ),

            "windows_analyzed": len(
                analyses
            ),

            "semantic_events": len(
                events
            ),

            "device": str(
                device
            ),
        },

        "summary": {
            "event_counts": event_counts,
        },

        "events": events,

        "window_analyses": analyses,
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
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "✅ SOUND EVENT ANALYSIS COMPLETE"
    )
    print("=" * 72)

    print()
    print(
        f"Transient candidates : "
        f"{len(transient_times)}"
    )

    print(
        f"Windows analyzed    : "
        f"{len(analyses)}"
    )

    print(
        f"Semantic events     : "
        f"{len(events)}"
    )

    print()

    print(
        "Detected event types:"
    )

    if event_counts:

        for event_name, count in sorted(
            event_counts.items(),
            key=lambda x: -x[1],
        ):

            print(
                f"  {event_name:12s}: "
                f"{count}"
            )

    else:

        print(
            "  No target sound events "
            "above confidence threshold."
        )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_PATH
    )

    print()

    print(
        "Strongest semantic events:"
    )

    for event in sorted(
        events,
        key=lambda x: x["confidence"],
        reverse=True,
    )[:20]:

        print(
            f"  {event['time']:7.3f}s | "
            f"{event['event']:12s} | "
            f"{event['confidence']:.3f}"
        )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()