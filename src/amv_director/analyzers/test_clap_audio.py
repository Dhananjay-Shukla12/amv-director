from __future__ import annotations

import math
from pathlib import Path
from typing import List

import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

AUDIO_PATH = (
    BASE_DIR
    / "data/outputs/reference_audio.wav"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "laion/clap-htsat-fused"

SAMPLE_RATE = 48000

# Use CPU for this first test.
# Your M1 can run it, but CPU keeps memory pressure lower.
DEVICE = "cpu"


# ============================================================
# TEST WINDOWS
# ============================================================

# These are moments already identified as interesting by
# our previous audio analysis.
#
# We deliberately include several different parts of the AMV.
TEST_TIMES = [
    9.440,
    12.720,
    31.520,
    35.488,
    36.592,
    41.040,
    54.112,
    57.552,
]


# ============================================================
# AUDIO DESCRIPTIONS
# ============================================================

PROMPTS = [
    "the sound of a gunshot",
    "the sound of a pistol firing",
    "the sound of an explosion",
    "the sound of a sword clash",
    "the sound of a punch or physical impact",
    "the sound of a heavy impact",
    "the sound of a fast whoosh",
    "the sound of footsteps",
    "the sound of glass breaking",
    "the sound of someone screaming",
    "the sound of human speech",
    "music playing in the background",
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:
        return default


def load_audio() -> np.ndarray:

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

    print(
        f"✅ Duration: "
        f"{len(audio) / SAMPLE_RATE:.3f}s"
    )

    return audio.astype(
        np.float32
    )


def extract_window(
    audio: np.ndarray,
    center_time: float,
    duration: float = 6.0,
) -> np.ndarray:

    half = duration / 2.0

    start = max(
        0.0,
        center_time - half,
    )

    end = min(
        len(audio) / SAMPLE_RATE,
        center_time + half,
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

    # CLAP supports padding through the processor,
    # but keeping a valid non-empty array here is safer.
    if len(chunk) < 100:

        chunk = np.zeros(
            int(
                SAMPLE_RATE
                * duration
            ),
            dtype=np.float32,
        )

    return chunk.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("🎧 CLAP AUDIO ↔ TEXT TEST")
    print("=" * 72)

    print()
    print(
        f"Model  : {MODEL_NAME}"
    )

    print(
        f"Device : {DEVICE}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print(
        "Loading CLAP..."
    )

    processor = ClapProcessor.from_pretrained(
        MODEL_NAME
    )

    model = ClapModel.from_pretrained(
        MODEL_NAME
    )

    model.to(DEVICE)
    model.eval()

    print(
        "✅ CLAP loaded"
    )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio = load_audio()

    # --------------------------------------------------------
    # Test each timestamp
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TESTING AUDIO WINDOWS")
    print("=" * 72)

    for index, timestamp in enumerate(
        TEST_TIMES,
        start=1,
    ):

        print()
        print(
            "-" * 72
        )

        print(
            f"[{index}/{len(TEST_TIMES)}] "
            f"Timestamp: {timestamp:.3f}s"
        )

        chunk = extract_window(
            audio,
            timestamp,
        )

        # ----------------------------------------------------
        # Process audio + text together.
        # ----------------------------------------------------

        inputs = processor(
            text=PROMPTS,
            audio=chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # CLAP audio-text similarity
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = model(
                **inputs
            )

            logits = (
                outputs.logits_per_audio
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )[0]

        values = probabilities.tolist()

        ranked = sorted(
            zip(
                PROMPTS,
                values,
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        print()
        print(
            "TOP AUDIO DESCRIPTIONS:"
        )

        for rank, (
            prompt,
            score,
        ) in enumerate(
            ranked[:8],
            start=1,
        ):

            print(
                f"  {rank:02d}. "
                f"{score:.4f}  "
                f"{prompt}"
            )

    print()
    print("=" * 72)
    print(
        "✅ CLAP TEST COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()