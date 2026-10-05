from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[2]

PLAN_FILE = (
    ROOT
    / "data"
    / "outputs"
    / "director_edit_plan_v11.json"
)

CLIPS_DIR = (
    ROOT
    / "data"
    / "clips"
)

AUDIO_FILE = (
    ROOT
    / "data"
    / "outputs"
    / "reference_audio.wav"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "outputs"
    / "renders"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "preview_10s.mp4"
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

PREVIEW_DURATION = 10.0

WIDTH = 1920
HEIGHT = 1080

FPS = 30


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def run(cmd):
    print(
        "\n$ "
        + " ".join(str(x) for x in cmd)
    )

    subprocess.run(
        cmd,
        check=True,
    )


def load_plan():

    with PLAN_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ------------------------------------------------------------
# FFmpeg filter for a clip
# ------------------------------------------------------------

def build_filter(
    style: str,
    duration: float,
):
    """
    Conservative first-pass AMV treatment.

    We intentionally avoid aggressive effects here.
    This render is primarily a pipeline validation pass.
    """

    style = style.lower()

    filters = []

    # --------------------------------------------------------
    # Normalize video format
    # --------------------------------------------------------

    filters.extend(
        [
            f"scale={WIDTH}:{HEIGHT}:"
            f"force_original_aspect_ratio=increase",

            f"crop={WIDTH}:{HEIGHT}",

            "setsar=1",

            "fps=30",

            "format=yuv420p",
        ]
    )

    # --------------------------------------------------------
    # Style treatment
    # --------------------------------------------------------

    if (
        "slow" in style
        or "cinematic" in style
        or style == "calm"
    ):

        # Gentle contrast/saturation lift.
        filters.append(
            "eq="
            "contrast=1.05:"
            "saturation=1.06:"
            "brightness=0.01"
        )

        # Subtle static punch.
        filters.append(
            "scale=1960:1102"
        )

        filters.append(
            "crop=1920:1080"
        )

        filters.append(
            "vignette=PI/5"
        )

    elif (
        "fast" in style
        or "action" in style
    ):

        filters.append(
            "eq="
            "contrast=1.10:"
            "saturation=1.08:"
            "brightness=0.005"
        )

        # Small action punch.
        filters.append(
            "scale=1980:1114"
        )

        filters.append(
            "crop=1920:1080"
        )

    elif "impact" in style:

        filters.append(
            "eq="
            "contrast=1.15:"
            "saturation=1.12:"
            "brightness=0.015"
        )

        # Stronger punch.
        filters.append(
            "scale=2020:1136"
        )

        filters.append(
            "crop=1920:1080"
        )

    else:

        filters.append(
            "eq="
            "contrast=1.05:"
            "saturation=1.05"
        )

    return ",".join(filters)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print()
    print("=" * 80)
    print("🎬 AMV DIRECTOR — 10s RENDER PREVIEW")
    print("=" * 80)

    plan = load_plan()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Flatten clips and calculate target timing.
    # --------------------------------------------------------

    timeline = []

    for interval in plan["intervals"]:

        cursor = float(
            interval["reference_start"]
        )

        for clip in interval["clips"]:

            duration = float(
                clip["duration"]
            )

            if cursor >= PREVIEW_DURATION:
                break

            usable_duration = min(
                duration,
                PREVIEW_DURATION - cursor,
            )

            if usable_duration <= 0:
                continue

            timeline.append(
                {
                    "target_start": cursor,
                    "target_end":
                        cursor + usable_duration,
                    "duration":
                        usable_duration,
                    "style":
                        interval.get(
                            "style",
                            "medium",
                        ),
                    "clip":
                        clip,
                }
            )

            cursor += usable_duration

        if cursor >= PREVIEW_DURATION:
            break

    print(
        f"Preview duration : "
        f"{PREVIEW_DURATION:.3f}s"
    )

    print(
        f"Clips in preview : "
        f"{len(timeline)}"
    )

    # --------------------------------------------------------
    # Temporary rendered clip directory.
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="amv_preview_"
    ) as tmp:

        tmp_dir = Path(tmp)

        rendered_clips = []

        # ----------------------------------------------------
        # Render individual clips.
        # ----------------------------------------------------

        for index, item in enumerate(
            timeline,
            start=1,
        ):

            clip = item["clip"]

            source = (
                CLIPS_DIR
                / clip["source_video"]
            )

            start = float(
                clip["source_start"]
            )

            duration = float(
                item["duration"]
            )

            style = item["style"]

            output = (
                tmp_dir
                / f"clip_{index:03d}.mp4"
            )

            video_filter = build_filter(
                style,
                duration,
            )

            print()
            print(
                f"[{index:02d}/{len(timeline)}] "
                f"{item['target_start']:.3f}"
                f"→"
                f"{item['target_end']:.3f}s | "
                f"{source.name} "
                f"S{clip['scene_id']:02d} | "
                f"src "
                f"{start:.3f}"
                f"→"
                f"{start + duration:.3f}s | "
                f"{style}"
            )

            cmd = [
                FFMPEG,

                "-hide_banner",
                "-loglevel",
                "error",

                "-ss",
                f"{start:.6f}",

                "-i",
                str(source),

                "-t",
                f"{duration:.6f}",

                "-vf",
                video_filter,

                "-an",

                "-c:v",
                "libx264",

                "-preset",
                "veryfast",

                "-crf",
                "18",

                "-pix_fmt",
                "yuv420p",

                "-r",
                str(FPS),

                "-movflags",
                "+faststart",

                "-y",
                str(output),
            ]

            run(cmd)

            rendered_clips.append(
                output
            )

        # ----------------------------------------------------
        # Concat list.
        # ----------------------------------------------------

        concat_file = (
            tmp_dir
            / "concat.txt"
        )

        with concat_file.open(
            "w",
            encoding="utf-8",
        ) as f:

            for clip in rendered_clips:

                # ffconcat requires escaped paths.
                path = (
                    str(clip)
                    .replace(
                        "'",
                        "'\\''"
                    )
                )

                f.write(
                    f"file '{path}'\n"
                )

        # ----------------------------------------------------
        # Join video clips.
        # ----------------------------------------------------

        silent_video = (
            tmp_dir
            / "silent_preview.mp4"
        )

        run(
            [
                FFMPEG,

                "-hide_banner",
                "-loglevel",
                "error",

                "-f",
                "concat",

                "-safe",
                "0",

                "-i",
                str(concat_file),

                "-an",

                "-c:v",
                "libx264",

                "-preset",
                "veryfast",

                "-crf",
                "18",

                "-pix_fmt",
                "yuv420p",

                "-movflags",
                "+faststart",

                "-y",
                str(silent_video),
            ]
        )

        # ----------------------------------------------------
        # Add reference music.
        # ----------------------------------------------------

        run(
            [
                FFMPEG,

                "-hide_banner",
                "-loglevel",
                "error",

                "-i",
                str(silent_video),

                "-i",
                str(AUDIO_FILE),

                "-t",
                f"{PREVIEW_DURATION:.6f}",

                "-map",
                "0:v:0",

                "-map",
                "1:a:0",

                "-c:v",
                "copy",

                "-c:a",
                "aac",

                "-b:a",
                "192k",

                "-ar",
                "44100",

                "-shortest",

                "-movflags",
                "+faststart",

                "-y",
                str(OUTPUT_FILE),
            ]
        )

    print()
    print("=" * 80)
    print("✅ PREVIEW RENDER COMPLETE")
    print("=" * 80)

    print()
    print("Saved:")
    print(OUTPUT_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
