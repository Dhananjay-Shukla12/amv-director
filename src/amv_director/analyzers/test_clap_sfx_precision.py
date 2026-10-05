from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor


BASE_DIR = Path(__file__).resolve().parents[3]

AUDIO_PATH = (
    BASE_DIR
    / "data/outputs/reference_audio.wav"
)

MODEL_NAME = "laion/clap-htsat-fused"

SAMPLE_RATE = 48000
DEVICE = "cpu"

# These are transient locations from our previous analysis.
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

PROMPTS = [
    "a gunshot",
    "a pistol firing",
    "a rifle firing",
    "a firearm shooting",
    "an explosion",
    "a sword clash",
    "a metal weapon clash",
    "a punch impact",
    "a heavy physical impact",
    "a whoosh sound",
    "footsteps",
    "glass breaking",
    "a scream",
    "human speech",
    "background music",
    "silence or ambient sound",
]


def extract_window(
    audio: np.ndarray,
    center: float,
    duration: float = 1.5,
) -> np.ndarray:

    half = duration / 2

    start = max(
        0,
        int(
            (center - half)
            * SAMPLE_RATE
        ),
    )

    end = min(
        len(audio),
        int(
            (center + half)
            * SAMPLE_RATE
        ),
    )

    chunk = audio[start:end]

    return chunk.astype(
        np.float32
    )


def main():

    print("=" * 72)
    print("🎯 CLAP SHORT-SFX PRECISION TEST")
    print("=" * 72)

    print()
    print("Loading audio...")

    audio, _ = librosa.load(
        str(AUDIO_PATH),
        sr=SAMPLE_RATE,
        mono=True,
    )

    print("✅ Audio loaded")

    print()
    print("Loading CLAP...")

    processor = ClapProcessor.from_pretrained(
        MODEL_NAME
    )

    model = ClapModel.from_pretrained(
        MODEL_NAME
    )

    model.to(DEVICE)
    model.eval()

    print("✅ CLAP loaded")

    for n, timestamp in enumerate(
        TEST_TIMES,
        start=1,
    ):

        print()
        print("-" * 72)
        print(
            f"[{n}/{len(TEST_TIMES)}] "
            f"{timestamp:.3f}s"
        )

        chunk = extract_window(
            audio,
            timestamp,
            duration=1.5,
        )

        inputs = processor(
            text=PROMPTS,
            audio=chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        with torch.inference_mode():

            outputs = model(
            **inputs
            )

            scores = torch.softmax(
            outputs.logits_per_audio,
            dim=-1,
            )[0]

        ranked = sorted(
            zip(
                PROMPTS,
                scores.tolist(),
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        for rank, (
            prompt,
            score,
        ) in enumerate(
            ranked[:8],
            start=1,
        ):

            print(
                f"{rank:02d}. "
                f"{score:.4f}  "
                f"{prompt}"
            )

    print()
    print("=" * 72)
    print("✅ PRECISION TEST COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()