from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import librosa


ROOT = Path(__file__).resolve().parents[2]

PLAN_FILE = (
    ROOT
    / "data"
    / "outputs"
    / "director_edit_plan_v11.json"
)

AUDIO_FILE = (
    ROOT
    / "data"
    / "outputs"
    / "reference_audio.wav"
)

CLIPS_DIR = (
    ROOT
    / "data"
    / "clips"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "outputs"
    / "renders"
)

# OUTPUT_FILE = (
#     OUTPUT_DIR
#     / "preview_v2_action_18_28.mp4"
# )
OUTPUT_FILE = (
    OUTPUT_DIR
    / "preview_v2_action_18_28.mp4"
)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# PREVIEW_DURATION = 10.0
PREVIEW_START = 18.0
PREVIEW_DURATION = 10.0

WIDTH = 1920
HEIGHT = 1080
FPS = 30

# ------------------------------------------------------------
# Music sync
# ------------------------------------------------------------

SYNC_TOLERANCE = 0.18

MIN_CLIP_DURATION = 0.10


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(cmd):
    print(
        "\n$ "
        + " ".join(str(x) for x in cmd)
    )

    subprocess.run(
        cmd,
        check=True,
    )


# ------------------------------------------------------------
# Detect strong musical onsets
# ------------------------------------------------------------

def detect_strong_onsets():

    print()
    print("🎵 ANALYZING MUSIC FOR EDIT-WORTHY ONSETS...")

    import numpy as np
    from scipy.signal import find_peaks

    y, sr = librosa.load(
        str(AUDIO_FILE),
        sr=None,
        mono=True,
    )

    hop_length = 512

    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
        aggregate="mean",
    )

    onset_times = librosa.times_like(
        onset_env,
        sr=sr,
        hop_length=hop_length,
    )

    if len(onset_env) == 0:
        return []

    # --------------------------------------------------------
    # We do NOT want every tiny onset.
    #
    # Keep only relatively prominent musical events and
    # enforce meaningful spacing between them.
    # --------------------------------------------------------

    percentile = float(
        np.percentile(
            onset_env,
            88,
        )
    )

    std = float(
        onset_env.std()
    )

    prominence = max(
        0.08,
        std * 0.35,
    )

    min_spacing_seconds = 0.18

    min_spacing_frames = max(
        1,
        int(
            min_spacing_seconds
            * sr
            / hop_length
        ),
    )

    peaks, properties = find_peaks(
        onset_env,
        height=percentile,
        prominence=prominence,
        distance=min_spacing_frames,
    )

    strong = [
        float(onset_times[i])
        for i in peaks
        if i < len(onset_times)
    ]

    print(
        f"Edit-worthy onsets detected : "
        f"{len(strong)}"
    )

    print("First 20:")

    for t in strong[:20]:
        print(
            f"  {t:.3f}s"
        )

    return strong


# ------------------------------------------------------------
# Load V11 and flatten
# ------------------------------------------------------------

def flatten_plan(plan):

    clips = []

    window_start = PREVIEW_START
    window_end = (
        PREVIEW_START
        + PREVIEW_DURATION
    )

    for interval in plan["intervals"]:

        cursor = float(
            interval["reference_start"]
        )

        for clip in interval["clips"]:

            duration = float(
                clip["duration"]
            )

            target_start = cursor
            target_end = (
                cursor + duration
            )

            # Entire clip is before preview window.
            if target_end <= window_start:
                cursor = target_end
                continue

            # Everything after preview window.
            if target_start >= window_end:
                return clips

            # ------------------------------------------------
            # Clip intersects preview window.
            # Convert timing to preview-relative coordinates.
            # ------------------------------------------------

            visible_start = max(
                target_start,
                window_start,
            )

            visible_end = min(
                target_end,
                window_end,
            )

            visible_duration = (
                visible_end
                - visible_start
            )

            if visible_duration <= 0:
                cursor = target_end
                continue

            clip_copy = dict(
                clip
            )

            clip_copy["target_start"] = (
                visible_start
                - window_start
            )

            clip_copy["target_end"] = (
                visible_end
                - window_start
            )

            clip_copy["duration"] = (
                visible_duration
            )

            clip_copy["style"] = (
                interval.get(
                    "style",
                    "medium",
                )
            )

            clip_copy["interval_id"] = (
                interval["interval_id"]
            )

            # For partial clips at the preview boundaries,
            # shift the source start forward proportionally.
            source_full_start = float(
                clip["source_start"]
            )

            source_full_end = float(
                clip["source_end"]
            )

            source_full_duration = (
                source_full_end
                - source_full_start
            )

            left_trim = (
                visible_start
                - target_start
            )

            right_trim = (
                target_end
                - visible_end
            )

            if (
                source_full_duration > 0
                and duration > 0
            ):

                source_offset = (
                    source_full_duration
                    * (
                        left_trim
                        / duration
                    )
                )

                source_offset = max(
                    0.0,
                    min(
                        source_offset,
                        source_full_duration,
                    ),
                )

                clip_copy["source_start"] = (
                    source_full_start
                    + source_offset
                )

                clip_copy["source_end"] = (
                    source_full_end
                    - (
                        source_full_duration
                        * (
                            right_trim
                            / duration
                        )
                    )
                )

            clips.append(
                clip_copy
            )

            cursor = target_end

    return clips


# ------------------------------------------------------------
# Music-sync timing correction
# ------------------------------------------------------------

def retime_boundaries(
    clips,
    strong_onsets,
):
    """
    Snap INTERNAL boundaries to strong musical onsets.

    Reference interval boundaries themselves remain fixed.
    We only adjust the artificial subdivision created by
    source-clip duration.

    The source video may therefore be sped/slowed modestly
    during rendering.
    """

    if len(clips) < 2:
        return clips

    corrected = [
        dict(c)
        for c in clips
    ]

    # Initial boundaries.
    boundaries = [
        float(
            c["target_start"]
        )
        for c in corrected[1:]
    ]

    # End of preview is fixed.
    for i, boundary in enumerate(
        boundaries,
        start=1,
    ):

        # Don't move across a reference interval boundary.
        previous = corrected[i - 1]
        current = corrected[i]

        if (
            previous["interval_id"]
            != current["interval_id"]
        ):
            continue

        candidates = [
            t
            for t in strong_onsets
            if abs(
                t - boundary
            ) <= SYNC_TOLERANCE
        ]

        if not candidates:
            continue

        nearest = min(
            candidates,
            key=lambda t:
                abs(t - boundary)
        )

        # Preserve minimum durations.
        prev_start = float(
            corrected[i - 1][
                "target_start"
            ]
        )

        current_end = float(
            corrected[i][
                "target_end"
            ]
        )

        if (
            nearest
            - prev_start
            < MIN_CLIP_DURATION
        ):
            continue

        if (
            current_end
            - nearest
            < MIN_CLIP_DURATION
        ):
            continue

        corrected[i]["target_start"] = (
            nearest
        )

        corrected[i - 1]["target_end"] = (
            nearest
        )

    # Recalculate durations.
    for clip in corrected:

        clip["duration"] = (
            float(
                clip["target_end"]
            )
            - float(
                clip["target_start"]
            )
        )

    print()
    print("🎯 MUSIC-SYNCED BOUNDARIES")

    for i, clip in enumerate(
        corrected
    ):

        print(
            f"  [{i+1:02d}] "
            f"{clip['target_start']:.3f}"
            f"→"
            f"{clip['target_end']:.3f}s "
            f"{clip['source_video']} "
            f"S{clip['scene_id']:02d}"
        )

    return corrected


# ------------------------------------------------------------
# Effect selection
# ------------------------------------------------------------

def effect_profile(
    style,
    duration,
):
    s = style.lower()

    if "impact" in s:
        return {
            "zoom": 0.085,
            "shake": 7,
            "flash": True,
            "blur": True,
            "contrast": 1.16,
            "saturation": 1.12,
        }

    if (
        "action" in s
        or "fast" in s
    ):
        return {
            "zoom": 0.055,
            "shake": 3.5,
            "flash": False,
            "blur": True,
            "contrast": 1.10,
            "saturation": 1.09,
        }

    if (
        "cinematic" in s
        or "calm" in s
        or "slow" in s
    ):
        return {
            "zoom": 0.035,
            "shake": 0,
            "flash": False,
            "blur": False,
            "contrast": 1.06,
            "saturation": 1.07,
        }

    return {
        "zoom": 0.04,
        "shake": 1.0,
        "flash": False,
        "blur": False,
        "contrast": 1.06,
        "saturation": 1.06,
    }


# ------------------------------------------------------------
# Build animated FFmpeg filter
# ------------------------------------------------------------

def build_filter(
    style,
    duration,
):
    profile = effect_profile(
        style,
        duration,
    )

    zoom = profile["zoom"]
    shake = profile["shake"]

    filters = []

    # --------------------------------------------------------
    # Base 16:9 conversion.
    # --------------------------------------------------------

    filters.extend(
        [
            (
                f"scale={WIDTH}:{HEIGHT}:"
                f"force_original_aspect_ratio=increase"
            ),
            f"crop={WIDTH}:{HEIGHT}",
            "setsar=1",
            "fps=30",
        ]
    )

    # --------------------------------------------------------
    # Animated zoom.
    #
    # Slight acceleration into the shot, then stabilizes.
    # --------------------------------------------------------

    zoom_expr = (
        f"(1+{zoom}*min(t/{max(duration,0.1):.6f},1))"
    )

    scaled_w = (
        f"trunc({WIDTH}*{zoom_expr}/2)*2"
    )

    scaled_h = (
        f"trunc({HEIGHT}*{zoom_expr}/2)*2"
    )

    filters.append(
        (
            f"scale="
            f"w='{scaled_w}':"
            f"h='{scaled_h}':"
            f"eval=frame"
        )
    )

    # --------------------------------------------------------
    # Animated shake.
    # --------------------------------------------------------

    if shake > 0:

        x_expr = (
            f"(iw-{WIDTH})/2"
            f"+"
            f"{shake}"
            f"*sin(2*PI*"
            f"11*t)"
        )

        y_expr = (
            f"(ih-{HEIGHT})/2"
            f"+"
            f"{shake}"
            f"*cos(2*PI*"
            f"13*t)"
        )

    else:

        x_expr = (
            f"(iw-{WIDTH})/2"
        )

        y_expr = (
            f"(ih-{HEIGHT})/2"
        )

    filters.append(
        (
            f"crop="
            f"{WIDTH}:{HEIGHT}:"
            f"x='{x_expr}':"
            f"y='{y_expr}'"
        )
    )

    # --------------------------------------------------------
    # Color treatment.
    # --------------------------------------------------------

    filters.append(
        (
            f"eq="
            f"contrast={profile['contrast']}:"
            f"saturation={profile['saturation']}:"
            f"brightness=0.005"
        )
    )

    # --------------------------------------------------------
    # Action motion blur.
    # --------------------------------------------------------

    if profile["blur"]:
        filters.append(
            "tmix="
            "frames=3:"
            "weights='1 2 1'"
        )

    # --------------------------------------------------------
    # Small sharpening after blur.
    # --------------------------------------------------------

    if profile["blur"]:
        filters.append(
            "unsharp="
            "5:5:0.55:"
            "5:5:0.0"
        )

    # --------------------------------------------------------
    # Flash at impact.
    # --------------------------------------------------------

    if profile["flash"]:

        flash_end = min(
            0.10,
            max(
                0.04,
                duration * 0.12,
            ),
        )

        filters.append(
            (
                "drawbox="
                "x=0:"
                "y=0:"
                "w=iw:"
                "h=ih:"
                "color=white@0.78:"
                "t=fill:"
                f"enable='between("
                f"t,0,{flash_end:.4f}"
                f")'"
            )
        )

    # --------------------------------------------------------
    # Vignette.
    # --------------------------------------------------------

    filters.append(
        "vignette=PI/5"
    )

    filters.append(
        "format=yuv420p"
    )

    return ",".join(
        filters
    )


# ------------------------------------------------------------
# Main renderer
# ------------------------------------------------------------

def main():

    print()
    print("=" * 80)
    print(
        "🎬 AMV DIRECTOR — RENDERER V2"
    )
    print("=" * 80)

    plan = load_json(
        PLAN_FILE
    )

    strong_onsets = (
        detect_strong_onsets()
    )

    clips = flatten_plan(
        plan
    )

    print()
    print(
        f"Original preview clips : "
        f"{len(clips)}"
    )

    clips = retime_boundaries(
        clips,
        strong_onsets,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Render individual clips.
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="amv_v2_"
    ) as tmp:

        tmp_dir = Path(tmp)

        rendered = []

        for index, clip in enumerate(
            clips,
            start=1,
        ):

            source = (
                CLIPS_DIR
                / clip[
                    "source_video"
                ]
            )

            source_start = float(
                clip[
                    "source_start"
                ]
            )

            original_duration = float(
                clip.get(
                    "duration",
                    0.0,
                )
            )

            # V11 source clip's actual available
            # duration before timing correction.
            source_duration = (
                float(
                    clip[
                        "source_end"
                    ]
                )
                -
                source_start
            )

            target_duration = (
                float(
                    clip[
                        "target_end"
                    ]
                )
                -
                float(
                    clip[
                        "target_start"
                    ]
                )
            )

            # ------------------------------------------------
            # Speed correction.
            #
            # Keep within practical ±20%.
            # For larger differences, use the first portion.
            # ------------------------------------------------

            speed = (
                source_duration
                /
                max(
                    target_duration,
                    0.001,
                )
            )

            speed = max(
                0.80,
                min(
                    1.20,
                    speed,
                ),
            )

            actual_source_duration = (
                target_duration
                * speed
            )

            actual_source_duration = min(
                actual_source_duration,
                source_duration,
            )

            output = (
                tmp_dir
                / f"clip_{index:03d}.mp4"
            )

            video_filter = build_filter(
                clip["style"],
                target_duration,
            )

            # setpts handles small speed changes.
            setpts_factor = (
                1.0
                / max(
                    speed,
                    0.001,
                )
            )

            full_filter = (
                f"setpts={setpts_factor:.6f}*PTS,"
                + video_filter
            )

            print()
            print(
                f"[{index:02d}/{len(clips)}] "
                f"{clip['target_start']:.3f}"
                f"→"
                f"{clip['target_end']:.3f}s | "
                f"{clip['source_video']} "
                f"S{clip['scene_id']:02d} | "
                f"source="
                f"{source_start:.3f}"
                f"→"
                f"{source_start + actual_source_duration:.3f}"
                f" | "
                f"speed={speed:.3f} | "
                f"{clip['style']}"
            )

            cmd = [
                FFMPEG,

                "-hide_banner",
                "-loglevel",
                "error",

                "-ss",
                f"{source_start:.6f}",

                "-i",
                str(source),

                "-t",
                f"{actual_source_duration:.6f}",

                "-vf",
                full_filter,

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
                "30",

                "-fps_mode",
                "cfr",

                "-video_track_timescale",
                "90000",

                "-y",
                str(output),
            ]

            run(cmd)

            rendered.append(
                output
            )

        # ----------------------------------------------------
        # Concat.
        # ----------------------------------------------------

        concat_file = (
            tmp_dir
            / "concat.txt"
        )

        with concat_file.open(
            "w",
            encoding="utf-8",
        ) as f:

            for path in rendered:

                escaped = str(
                    path
                ).replace(
                    "'",
                    "'\\''",
                )

                f.write(
                    f"file '{escaped}'\n"
                )

        joined = (
            tmp_dir
            / "joined.mp4"
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

                "-vf",
                (
                    "fps=30,"
                    f"tpad=stop_mode=clone:"
                    "stop_duration=0.5,"
                    f"trim=duration={PREVIEW_DURATION:.6f},"
                    "setpts=PTS-STARTPTS"
                ),

                "-c:v",
                "libx264",

                "-preset",
                "veryfast",

                "-crf",
                "18",

                "-pix_fmt",
                "yuv420p",

                "-r",
                "30",

                "-fps_mode",
                "cfr",

                "-video_track_timescale",
                "90000",

                "-y",
                str(joined),
            ]
        )

        # ----------------------------------------------------
        # Add exact reference audio.
        # ----------------------------------------------------

        run(
            [
                FFMPEG,

                "-hide_banner",
                "-loglevel",
                "error",

                "-i",
                str(joined),

                "-i",
                str(AUDIO_FILE),

                "-map",
                "0:v:0",

                "-map",
                "1:a:0",

                "-filter:a",
                (
                    # f"atrim=0:"
                    f"atrim={PREVIEW_START}:"
                    f"{PREVIEW_DURATION:.6f},"
                    "asetpts=PTS-STARTPTS"
                ),

                "-t",
                f"{PREVIEW_DURATION:.6f}",

                "-c:v",
                "copy",

                "-c:a",
                "aac",

                "-b:a",
                "192k",

                "-ar",
                "44100",

                "-ac",
                "2",

                "-video_track_timescale",
                "90000",

                "-movflags",
                "+faststart",

                "-y",
                str(OUTPUT_FILE),
            ]
        )

    print()
    print("=" * 80)
    print("✅ RENDERER V2 PREVIEW COMPLETE")
    print("=" * 80)

    print()
    print(
        "Output:"
    )
    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Preview duration target: "
        f"{PREVIEW_DURATION:.3f}s"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
