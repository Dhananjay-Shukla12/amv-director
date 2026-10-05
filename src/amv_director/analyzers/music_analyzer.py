from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import librosa
import numpy as np


VIDEO_PATH = "data/references/reference_full.mp4"
AUDIO_PATH = "data/outputs/reference_audio.wav"
OUTPUT_PATH = "data/outputs/music_analysis.json"


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract clean mono WAV audio from the reference video."""

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    Path(audio_path).parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        audio_path,
    ]

    print("Extracting audio...")

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed:\n" + result.stderr[-2000:]
        )

    print(f"✅ Audio extracted: {audio_path}")


def normalize(values: np.ndarray) -> np.ndarray:
    """Normalize an array to 0-1."""

    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if maximum - minimum < 1e-9:
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def detect_energy_peaks(
    energy: np.ndarray,
    times: np.ndarray,
    threshold: float = 0.80,
    min_gap: float = 0.35,
) -> list[dict[str, float]]:
    """Find strong local energy peaks."""

    peaks: list[dict[str, float]] = []

    for i in range(1, len(energy) - 1):

        local_peak = (
            energy[i] > energy[i - 1]
            and energy[i] >= energy[i + 1]
        )

        strong = energy[i] >= threshold

        if not (local_peak and strong):
            continue

        time = float(times[i])

        if peaks and time - peaks[-1]["time"] < min_gap:
            if energy[i] > peaks[-1]["energy"]:
                peaks[-1] = {
                    "time": round(time, 3),
                    "energy": round(float(energy[i]), 3),
                }
        else:
            peaks.append(
                {
                    "time": round(time, 3),
                    "energy": round(float(energy[i]), 3),
                }
            )

    return peaks


def analyze_music(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> dict[str, Any]:

    extract_audio(video_path, audio_path)

    print("Loading audio...")

    y, sr = librosa.load(
        audio_path,
        sr=44100,
        mono=True,
    )

    duration = librosa.get_duration(
        y=y,
        sr=sr,
    )

    print(f"Duration: {duration:.2f}s")

    # ---------------------------------------------------------
    # TEMPO / BEATS
    # ---------------------------------------------------------

    print("Detecting beats...")

    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        units="frames",
    )

    bpm = float(np.asarray(tempo).reshape(-1)[0])

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
    )

    beat_times = [
        round(float(t), 3)
        for t in beat_times
    ]

    # ---------------------------------------------------------
    # ONSETS
    # ---------------------------------------------------------

    print("Detecting onsets...")

    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="frames",
    )

    onset_times = librosa.frames_to_time(
        onset_frames,
        sr=sr,
    )

    onset_times = [
        round(float(t), 3)
        for t in onset_times
    ]

    # ---------------------------------------------------------
    # ENERGY
    # ---------------------------------------------------------

    print("Analyzing energy...")

    rms = librosa.feature.rms(y=y)[0]
    energy = normalize(rms)

    energy_times = librosa.frames_to_time(
        np.arange(len(energy)),
        sr=sr,
    )

    energy_peaks = detect_energy_peaks(
        energy,
        energy_times,
    )

    # Save a manageable-size energy curve.
    sample_count = min(300, len(energy))

    indices = np.linspace(
        0,
        len(energy) - 1,
        sample_count,
        dtype=int,
    )

    energy_curve = [
        {
            "time": round(float(energy_times[i]), 3),
            "energy": round(float(energy[i]), 3),
        }
        for i in indices
    ]

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------

    result = {
        "audio": {
            "source": video_path,
            "duration_seconds": round(float(duration), 3),
            "sample_rate": sr,
        },

        "tempo": {
            "bpm": round(bpm, 3),
        },

        "beats": [
            {
                "index": i,
                "time": time,
            }
            for i, time in enumerate(
                beat_times,
                start=1,
            )
        ],

        "onsets": [
            {
                "time": time,
            }
            for time in onset_times
        ],

        "energy_peaks": energy_peaks,

        "energy_curve": energy_curve,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n✅ Music analysis complete.")
    print(f"JSON: {output}")

    return result


if __name__ == "__main__":
    analyze_music(
        video_path=VIDEO_PATH,
        audio_path=AUDIO_PATH,
        output_path=OUTPUT_PATH,
    )