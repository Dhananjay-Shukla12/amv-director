from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

AUDIO_PATH = (
    BASE_DIR
    / "data/outputs/reference_audio.wav"
)

PREVIOUS_ANALYSIS = (
    BASE_DIR
    / "data/outputs/reference_sound_events_v2.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data/outputs/reference_clap_sfx_scan.json"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "laion/clap-htsat-fused"

SAMPLE_RATE = 48000

DEVICE = "cpu"


# ============================================================
# WINDOW
# ============================================================

WINDOW_SECONDS = 1.5


# ============================================================
# PROMPTS
# ============================================================

PROMPTS = {

    "gunshot": [
        "a gunshot",
        "the sound of a gun firing",
        "a pistol firing",
        "a rifle firing",
        "a firearm shooting",
    ],

    "explosion": [
        "an explosion",
        "a large blast",
        "a bomb exploding",
    ],

    "sword": [
        "a sword clash",
        "two metal weapons clashing",
        "a blade striking another blade",
    ],

    "impact": [
        "a punch impact",
        "a heavy physical impact",
        "a person being hit",
        "a strong hit or thump",
    ],

    "whoosh": [
        "a fast whoosh",
        "a fast swishing sound",
        "something moving quickly through the air",
    ],

    "footsteps": [
        "footsteps",
        "someone running",
        "someone walking",
    ],

    "glass": [
        "glass breaking",
        "glass shattering",
        "something made of glass breaking",
    ],

    "scream": [
        "someone screaming",
        "someone shouting",
        "a loud human cry",
    ],

    "voice": [
        "human speech",
        "someone talking",
        "a human voice",
    ],

    "music": [
        "background music",
        "music playing",
        "an instrumental music track",
    ],

    "ambient": [
        "background ambient sound",
        "environmental background noise",
        "room or outdoor ambience",
    ],
}


# ============================================================
# LOAD TRANSIENT TIMES
# ============================================================

def load_transient_times():

    with open(
        PREVIOUS_ANALYSIS,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    analyses = data.get(
        "window_analyses",
        [],
    )

    times = []

    for item in analyses:

        timestamp = item.get(
            "timestamp"
        )

        if timestamp is not None:

            times.append(
                float(timestamp)
            )

    return sorted(
        set(
            round(x, 4)
            for x in times
        )
    )


# ============================================================
# AUDIO
# ============================================================

def load_audio():

    print()
    print("Loading audio...")

    audio, _ = librosa.load(
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
    audio,
    center_time,
):

    half = WINDOW_SECONDS / 2

    start = max(
        0,
        int(
            (center_time - half)
            * SAMPLE_RATE
        ),
    )

    end = min(
        len(audio),
        int(
            (center_time + half)
            * SAMPLE_RATE
        ),
    )

    chunk = audio[
        start:end
    ]

    return chunk.astype(
        np.float32
    )


# ============================================================
# MODEL
# ============================================================

def load_model():

    print()
    print(
        "Loading CLAP..."
    )

    processor = (
        ClapProcessor.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        ClapModel.from_pretrained(
            MODEL_NAME
        )
    )

    model.to(DEVICE)
    model.eval()

    print(
        "✅ CLAP loaded"
    )

    return (
        model,
        processor,
    )


# ============================================================
# TEXT PROMPTS
# ============================================================

def flatten_prompts():

    flat_prompts = []
    prompt_groups = []

    for group, prompts in PROMPTS.items():

        for prompt in prompts:

            prompt_groups.append(
                group
            )

            flat_prompts.append(
                prompt
            )

    return (
        flat_prompts,
        prompt_groups,
    )


# ============================================================
# ANALYZE WINDOW
# ============================================================

def analyze_window(
    audio,
    timestamp,
    model,
    processor,
    prompts,
    prompt_groups,
):

    chunk = extract_window(
        audio,
        timestamp,
    )

    inputs = processor(
        text=prompts,
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

    values = scores.tolist()

    # --------------------------------------------------------
    # Best prompt in each semantic group
    # --------------------------------------------------------

    group_scores = {}
    group_best_prompt = {}

    for group in PROMPTS:

        group_indices = [
            i
            for i, g in enumerate(
                prompt_groups
            )
            if g == group
        ]

        if not group_indices:
            continue

        best_index = max(
            group_indices,
            key=lambda i: values[i],
        )

        group_scores[group] = float(
            values[best_index]
        )

        group_best_prompt[group] = {
            "prompt": prompts[
                best_index
            ],
            "score": float(
                values[best_index]
            ),
        }

    # --------------------------------------------------------
    # Rank semantic groups
    # --------------------------------------------------------

    ranked_groups = sorted(
        group_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "timestamp": round(
            timestamp,
            4,
        ),
        "window_start": round(
            timestamp
            - WINDOW_SECONDS / 2,
            4,
        ),
        "window_end": round(
            timestamp
            + WINDOW_SECONDS / 2,
            4,
        ),
        "group_scores": {
            k: round(v, 5)
            for k, v in group_scores.items()
        },
        "best_prompt": {
            group: {
                "prompt": info[
                    "prompt"
                ],
                "score": round(
                    info["score"],
                    5,
                ),
            }
            for group, info
            in group_best_prompt.items()
        },
        "ranked_groups": [
            {
                "group": group,
                "score": round(
                    score,
                    5,
                ),
            }
            for group, score
            in ranked_groups
        ],
    }


# ============================================================
# BUILD EVENT CANDIDATES
# ============================================================

def build_candidates(
    analyses,
):

    candidates = []

    ignored = {
        "music",
        "ambient",
    }

    for analysis in analyses:

        groups = analysis[
            "group_scores"
        ]

        ranked = [
            item
            for item in analysis[
                "ranked_groups"
            ]
            if item[
                "group"
            ] not in ignored
        ]

        if not ranked:
            continue

        best = ranked[0]

        best_score = float(
            best["score"]
        )

        second_score = (
            float(
                ranked[1]["score"]
            )
            if len(ranked) > 1
            else 0.0
        )

        margin = (
            best_score
            - second_score
        )

        candidates.append(
            {
                "time": analysis[
                    "timestamp"
                ],
                "event": best[
                    "group"
                ],
                "score": round(
                    best_score,
                    5,
                ),
                "margin": round(
                    margin,
                    5,
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

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("🔊 CLAP FULL SFX SCANNER")
    print("=" * 72)

    times = load_transient_times()

    print()
    print(
        f"Transient candidates: "
        f"{len(times)}"
    )

    audio = load_audio()

    model, processor = load_model()

    prompts, prompt_groups = (
        flatten_prompts()
    )

    print()
    print(
        f"Prompt count: "
        f"{len(prompts)}"
    )

    analyses = []

    for index, timestamp in enumerate(
        times,
        start=1,
    ):

        print(
            f"[{index:03d}/{len(times):03d}] "
            f"{timestamp:7.3f}s ... ",
            end="",
            flush=True,
        )

        result = analyze_window(
            audio,
            timestamp,
            model,
            processor,
            prompts,
            prompt_groups,
        )

        analyses.append(
            result
        )

        ranked = result[
            "ranked_groups"
        ]

        # Find top non-music category.
        top = ranked[0]

        print(
            f"{top['group']:10s} "
            f"{top['score']:.3f}"
        )

    candidates = (
        build_candidates(
            analyses
        )
    )

    # --------------------------------------------------------
    # Sort strongest candidates
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x[
            "score"
        ],
        reverse=True,
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    counts = {}

    for candidate in candidates:

        event = candidate[
            "event"
        ]

        counts[event] = (
            counts.get(
                event,
                0,
            )
            + 1
        )

    output = {

        "metadata": {

            "engine":
                "clap_sfx_scanner_v3",

            "model":
                MODEL_NAME,

            "sample_rate":
                SAMPLE_RATE,

            "window_seconds":
                WINDOW_SECONDS,

            "transient_candidates":
                len(times),

            "windows_analyzed":
                len(analyses),

            "candidate_events":
                len(candidates),
        },

        "summary": {
            "candidate_counts":
                counts,
        },

        "candidates":
            candidates,

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

    print()
    print("=" * 72)
    print("✅ CLAP FULL SFX SCAN COMPLETE")
    print("=" * 72)

    print()
    print(
        "Candidate counts:"
    )

    for event, count in sorted(
        counts.items(),
        key=lambda x: -x[1],
    ):

        print(
            f"  {event:12s}: "
            f"{count}"
        )

    print()
    print(
        "STRONGEST SFX CANDIDATES:"
    )

    for candidate in candidates[:30]:

        print(
            f"  "
            f"{candidate['time']:7.3f}s | "
            f"{candidate['event']:12s} | "
            f"{candidate['score']:.3f} | "
            f"margin={candidate['margin']:.3f}"
        )

    print()
    print(
        f"Output:\n{OUTPUT_PATH}"
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()