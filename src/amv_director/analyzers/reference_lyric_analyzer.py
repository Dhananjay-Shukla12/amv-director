from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx_whisper


def transcribe_audio(
    audio_path: Path,
    model: str,
) -> dict:
    """
    Transcribe audio using MLX Whisper and preserve timestamped segments.
    """

    print(f"🎤 Transcribing: {audio_path}")
    print(f"🤖 Model: {model}")

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        verbose=False,
    )

    return result


def normalize_segments(raw_result: dict) -> list[dict]:
    """
    Convert Whisper output into a stable project-specific schema.
    """

    normalized = []

    for index, segment in enumerate(raw_result.get("segments", []), start=1):
        text = segment.get("text", "").strip()

        if not text:
            continue

        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))

        # Whisper exposes avg_logprob rather than a calibrated confidence.
        # Keep the raw value instead of pretending it is a probability.
        avg_logprob = segment.get("avg_logprob")

        normalized.append(
            {
                "phrase_id": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(max(0.0, end - start), 3),
                "text": text,
                "avg_logprob": (
                    round(float(avg_logprob), 4)
                    if avg_logprob is not None
                    else None
                ),
            }
        )

    return normalized


def build_output(
    audio_path: Path,
    model: str,
    raw_result: dict,
    segments: list[dict],
) -> dict:
    duration = None

    if segments:
        duration = max(segment["end"] for segment in segments)

    return {
        "meta": {
            "audio": str(audio_path),
            "model": model,
            "segment_count": len(segments),
            "estimated_transcribed_duration": (
                round(duration, 3) if duration is not None else 0.0
            ),
        },
        "segments": segments,
        "full_text": " ".join(segment["text"] for segment in segments),
    }


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe reference AMV lyrics with MLX Whisper."
    )

    parser.add_argument(
        "--audio",
        default="data/outputs/reference_audio.wav",
        help="Path to reference audio.",
    )

    parser.add_argument(
        "--output",
        default="data/outputs/reference_lyrics.json",
        help="Output JSON path.",
    )

    parser.add_argument(
        "--model",
        default="mlx-community/whisper-small-mlx",
        help="MLX Whisper model.",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    output_path = Path(args.output)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    raw_result = transcribe_audio(
        audio_path=audio_path,
        model=args.model,
    )

    segments = normalize_segments(raw_result)

    output = build_output(
        audio_path=audio_path,
        model=args.model,
        raw_result=raw_result,
        segments=segments,
    )

    save_json(output, output_path)

    print()
    print("===== LYRIC ANALYSIS =====")
    print(f"Segments: {len(segments)}")
    print(f"Output:   {output_path}")
    print("==========================")

    for segment in segments[:10]:
        print(
            f'{segment["start"]:7.3f} → '
            f'{segment["end"]:7.3f} | '
            f'{segment["text"]}'
        )


if __name__ == "__main__":
    main()