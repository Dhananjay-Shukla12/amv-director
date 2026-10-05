import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CREATIVE_PLAN = (
    BASE_DIR
    / "data"
    / "outputs"
    / "creative_director_plan.json"
)

EFFECT_PLAN = (
    BASE_DIR
    / "data"
    / "outputs"
    / "effect_director_plan.json"
)

DEFAULT_AUDIO = (
    BASE_DIR
    / "data"
    / "outputs"
    / "reference_audio.wav"
)

DEFAULT_OUTPUT = (
    BASE_DIR
    / "data"
    / "outputs"
    / "renders"
    / "amv_director_v3_preview.mp4"
)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

DEFAULT_CRF = 19
DEFAULT_PRESET = "veryfast"

FPS = 30

MIN_CLIP_DURATION = 0.08


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:
        if value is None:
            return default

        if isinstance(
            value,
            (list, tuple)
        ):
            if not value:
                return default

            value = value[0]

        return float(value)

    except Exception:
        return default


def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def run_command(
    command,
    label
):

    print()
    print(
        f"▶ {label}"
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if result.returncode != 0:

        print(result.stdout)

        raise RuntimeError(
            f"FFmpeg failed during: {label}"
        )

    return result.stdout


def ensure_even(value):

    value = int(value)

    if value % 2:
        value -= 1

    return max(
        2,
        value
    )


# ============================================================
# EFFECT EXTRACTION
# ============================================================

def effect_value(
    effects,
    effect_type,
    default=0.0
):

    for effect in effects:

        if effect.get("type") == effect_type:

            if "amount" in effect:
                return safe_float(
                    effect["amount"],
                    default
                )

    return default


def has_effect(
    effects,
    effect_type
):

    return any(
        effect.get("type")
        == effect_type
        for effect in effects
    )


# ============================================================
# VIDEO FILTER BUILDER
# ============================================================

def build_filter(
    effect_segment,
    duration,
    width,
    height
):

    choreography = effect_segment.get(
        "choreography",
        []
    )

    profile = effect_segment.get(
        "effect_profile",
        {}
    )

    finishing = effect_segment.get(
        "finishing",
        {}
    )

    role = effect_segment.get(
        "editorial_role",
        "build_zone"
    )

    intensity = safe_float(
        effect_segment.get(
            "intensity",
            0.5
        )
    )

    duration = max(
        duration,
        0.08
    )

    filters = []

    # --------------------------------------------------------
    # Normalize source
    # --------------------------------------------------------

    filters.append(
        (
            f"scale={width}:{height}:"
            "force_original_aspect_ratio=increase"
        )
    )

    filters.append(
        f"crop={width}:{height}"
    )

    # --------------------------------------------------------
    # ZOOM
    # --------------------------------------------------------

    zoom = safe_float(
        profile.get(
            "zoom",
            0.0
        )
    )

    zoom_requested = any(
        effect.get("type")
        in {
            "slow_zoom",
            "progressive_zoom",
            "push_zoom",
            "dynamic_zoom",
            "impact_pre_zoom",
        }
        for effect in choreography
    )

    if zoom_requested and zoom > 0:

        # Keep zoom visually controlled.
        max_scale = (
            1.0
            + 0.12 * min(
                zoom,
                1.0
            )
        )

        zoom_expr = (
            f"{1.0:.5f}+"
            f"{max_scale - 1.0:.5f}*"
            f"min(t/{duration:.6f},1)"
        )

        filters.append(
            (
                f"scale="
                f"w='ceil({width}*({zoom_expr})/2)*2':"
                f"h='ceil({height}*({zoom_expr})/2)*2':"
                "eval=frame"
            )
        )

        filters.append(
            (
                f"crop="
                f"{width}:{height}:"
                f"x='(in_w-out_w)/2':"
                f"y='(in_h-out_h)/2'"
            )
        )

    # --------------------------------------------------------
    # SHAKE
    # --------------------------------------------------------

    shake = safe_float(
        profile.get(
            "shake",
            0.0
        )
    )

    shake_requested = any(
        effect.get("type")
        in {
            "impact_shake",
            "directional_shake",
            "micro_shake",
        }
        for effect in choreography
    )

    if shake_requested and shake > 0:

        # Extra scale gives crop movement room.
        margin_scale = 1.08

        filters.append(
            (
                f"scale="
                f"{ensure_even(width * margin_scale)}:"
                f"{ensure_even(height * margin_scale)}"
            )
        )

        amplitude = (
            38.0
            * min(
                shake,
                1.0
            )
        )

        if has_effect(
            choreography,
            "impact_shake"
        ):

            impact_time = (
                duration * 0.72
            )

            sigma = max(
                0.035,
                min(
                    0.11,
                    duration * 0.20
                )
            )

            x_expr = (
                f"(in_w-out_w)/2+"
                f"{amplitude:.3f}*"
                f"sin(2*PI*23*t)*"
                f"exp(-pow((t-{impact_time:.4f})/"
                f"{sigma:.4f},2))"
            )

            y_expr = (
                f"(in_h-out_h)/2+"
                f"{amplitude * 0.65:.3f}*"
                f"cos(2*PI*19*t)*"
                f"exp(-pow((t-{impact_time:.4f})/"
                f"{sigma:.4f},2))"
            )

        else:

            x_expr = (
                f"(in_w-out_w)/2+"
                f"{amplitude:.3f}*"
                f"sin(2*PI*8*t)"
            )

            y_expr = (
                f"(in_h-out_h)/2+"
                f"{amplitude * 0.65:.3f}*"
                f"cos(2*PI*7*t)"
            )

        filters.append(
            (
                f"crop="
                f"{width}:{height}:"
                f"x='{x_expr}':"
                f"y='{y_expr}'"
            )
        )

    # --------------------------------------------------------
    # MOTION BLUR
    # --------------------------------------------------------

    blur = safe_float(
        profile.get(
            "blur",
            0.0
        )
    )

    if has_effect(
        choreography,
        "motion_blur"
    ) or has_effect(
        choreography,
        "post_impact_blur"
    ):

        sigma = min(
            2.2,
            0.8 + 2.0 * blur
        )

        filters.append(
            f"gblur=sigma={sigma:.3f}"
        )

    # --------------------------------------------------------
    # FLASH
    # --------------------------------------------------------

    flash = safe_float(
        profile.get(
            "flash",
            0.0
        )
    )

    if has_effect(
        choreography,
        "impact_flash"
    ) and flash > 0:

        impact_time = (
            duration * 0.72
        )

        flash_strength = min(
            0.80,
            0.65 * flash
        )

        sigma = max(
            0.025,
            min(
                0.07,
                duration * 0.10
            )
        )

        brightness_expr = (
            f"{flash_strength:.4f}*"
            f"exp(-pow((t-{impact_time:.4f})/"
            f"{sigma:.4f},2))"
        )

        filters.append(
            (
                f"eq="
                f"brightness='{brightness_expr}'"
            )
        )

    # --------------------------------------------------------
    # CONTRAST
    # --------------------------------------------------------

    contrast = safe_float(
        finishing.get(
            "contrast",
            profile.get(
                "contrast",
                0.0
            )
        )
    )

    contrast_value = (
        1.0
        + 0.45
        * min(
            contrast,
            1.0
        )
    )

    brightness_adjust = (
        -0.03
        if role == "impact_zone"
        else 0.0
    )

    filters.append(
        (
            f"eq="
            f"contrast={contrast_value:.4f}:"
            f"brightness={brightness_adjust:.4f}"
        )
    )

    # --------------------------------------------------------
    # VIGNETTE
    # --------------------------------------------------------

    vignette = safe_float(
        finishing.get(
            "vignette",
            profile.get(
                "vignette",
                0.0
            )
        )
    )

    if vignette > 0.02:

        angle = (
            0.55
            + 0.45
            * min(
                vignette,
                1.0
            )
        )

        filters.append(
            f"vignette={angle:.4f}"
        )

    # --------------------------------------------------------
    # Avoid excessive processing in very short clips.
    # --------------------------------------------------------

    if duration < 0.16:

        # Keep the result sharp enough for micro cuts.
        filters.append(
            "unsharp=5:5:0.25:5:5:0"
        )

    # --------------------------------------------------------
    # Final format
    # --------------------------------------------------------

    filters.append(
        "format=yuv420p"
    )

    return ",".join(
        filters
    )


# ============================================================
# FLATTEN CREATIVE TIMELINE
# ============================================================

def flatten_timeline(
    creative_plan
):

    timeline = []

    cursor = 0.0

    for segment in creative_plan.get(
        "segments",
        []
    ):

        segment_id = segment.get(
            "segment"
        )

        effects_id = segment_id

        for clip in segment.get(
            "clips",
            []
        ):

            duration = safe_float(
                clip.get(
                    "duration",
                    0
                )
            )

            if duration <= 0:
                continue

            source_start = safe_float(
                clip.get(
                    "source_start",
                    0
                )
            )

            source_end = safe_float(
                clip.get(
                    "source_end",
                    source_start + duration
                )
            )

            timeline.append(
                {
                    "segment_id":
                        segment_id,

                    "effect_segment_id":
                        effects_id,

                    "timeline_start":
                        cursor,

                    "timeline_end":
                        cursor + duration,

                    "duration":
                        duration,

                    "source_video":
                        clip[
                            "source_video"
                        ],

                    "source_path":
                        clip[
                            "source_path"
                        ],

                    "scene_id":
                        clip.get(
                            "scene_id"
                        ),

                    "source_start":
                        source_start,

                    "source_end":
                        source_end,
                }
            )

            cursor += duration

    return timeline, cursor


# ============================================================
# CLIP OVERLAP FOR PREVIEW
# ============================================================

def clip_for_window(
    clip,
    window_start,
    window_end
):

    clip_start = clip[
        "timeline_start"
    ]

    clip_end = clip[
        "timeline_end"
    ]

    overlap_start = max(
        clip_start,
        window_start
    )

    overlap_end = min(
        clip_end,
        window_end
    )

    if overlap_end <= overlap_start:

        return None

    offset = (
        overlap_start
        - clip_start
    )

    duration = (
        overlap_end
        - overlap_start
    )

    source_start = (
        clip["source_start"]
        + offset
    )

    return {
        **clip,
        "preview_timeline_start":
            overlap_start,

        "preview_timeline_end":
            overlap_end,

        "render_source_start":
            source_start,

        "render_duration":
            duration,
    }


# ============================================================
# BUILD EFFECT LOOKUP
# ============================================================

def effect_lookup(
    effect_plan
):

    return {
        segment.get(
            "segment"
        ): segment

        for segment in effect_plan.get(
            "segments",
            []
        )
    }


# ============================================================
# RENDER ONE SOURCE CLIP
# ============================================================

def render_clip(
    ffmpeg,
    clip,
    effect_segment,
    output_path,
    width,
    height,
    crf,
    preset
):

    duration = safe_float(
        clip["render_duration"]
    )

    source_start = safe_float(
        clip["render_source_start"]
    )

    source_path = Path(
        clip["source_path"]
    )

    if not source_path.exists():

        raise FileNotFoundError(
            f"Source video not found:\n"
            f"{source_path}"
        )

    filter_string = build_filter(
        effect_segment,
        duration,
        width,
        height
    )

    command = [
        ffmpeg,

        "-hide_banner",
        "-loglevel",
        "error",

        "-ss",
        f"{source_start:.6f}",

        "-t",
        f"{duration:.6f}",

        "-i",
        str(source_path),

        "-vf",
        filter_string,

        "-r",
        str(FPS),

        "-an",

        "-c:v",
        "libx264",

        "-preset",
        preset,

        "-crf",
        str(crf),

        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        "-y",
        str(output_path),
    ]

    run_command(
        command,
        (
            f"render "
            f"{clip['source_video']} "
            f"S{clip.get('scene_id')} "
            f"{duration:.3f}s"
        )
    )


# ============================================================
# CONCAT
# ============================================================

def concat_clips(
    ffmpeg,
    rendered_clips,
    concat_file,
    output_path
):

    with open(
        concat_file,
        "w",
        encoding="utf-8"
    ) as f:

        for clip in rendered_clips:

            # ffmpeg concat demuxer requires escaped paths.
            path = (
                str(
                    clip
                )
                .replace(
                    "'",
                    "'\\''"
                )
            )

            f.write(
                f"file '{path}'\n"
            )

    command = [
        ffmpeg,

        "-hide_banner",
        "-loglevel",
        "error",

        "-f",
        "concat",

        "-safe",
        "0",

        "-i",
        str(concat_file),

        "-c",
        "copy",

        "-movflags",
        "+faststart",

        "-y",
        str(output_path),
    ]

    run_command(
        command,
        "concatenate rendered clips"
    )


# ============================================================
# AUDIO MUX
# ============================================================

def mux_audio(
    ffmpeg,
    silent_video,
    audio_path,
    final_output,
    start,
    duration
):

    command = [
        ffmpeg,

        "-hide_banner",
        "-loglevel",
        "error",

        "-i",
        str(silent_video),

        "-ss",
        f"{start:.6f}",

        "-t",
        f"{duration:.6f}",

        "-i",
        str(audio_path),

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

        "-shortest",

        "-movflags",
        "+faststart",

        "-y",
        str(final_output),
    ]

    run_command(
        command,
        "mux audio"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AMV Director v3 renderer"
        )
    )

    parser.add_argument(
        "--audio",
        default=str(
            DEFAULT_AUDIO
        )
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=25.0
    )

    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH
    )

    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT
    )

    parser.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_CRF
    )

    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        )
    )

    args = parser.parse_args()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    audio_path = Path(
        args.audio
    ).expanduser().resolve()

    creative_path = CREATIVE_PLAN.resolve()
    effect_path = EFFECT_PLAN.resolve()

    output_path = Path(
        args.output
    ).expanduser().resolve()

    width = ensure_even(
        args.width
    )

    height = ensure_even(
        args.height
    )

    start = max(
        0.0,
        args.start
    )

    duration = max(
        0.10,
        args.duration
    )

    print("=" * 82)
    print("🎬 AMV DIRECTOR V3 RENDERER")
    print("=" * 82)

    print()
    print(
        f"FFmpeg               : "
        f"{ffmpeg}"
    )

    print(
        f"Resolution            : "
        f"{width}x{height}"
    )

    print(
        f"Timeline start        : "
        f"{start:.3f}s"
    )

    print(
        f"Timeline duration     : "
        f"{duration:.3f}s"
    )

    print(
        f"Audio                 : "
        f"{audio_path}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio not found:\n"
            f"{audio_path}"
        )

    creative_plan = load_json(
        creative_path
    )

    effect_plan = load_json(
        effect_path
    )

    effects = effect_lookup(
        effect_plan
    )

    timeline, total_duration = (
        flatten_timeline(
            creative_plan
        )
    )

    print()
    print(
        f"Creative timeline     : "
        f"{total_duration:.3f}s"
    )

    print(
        f"Render window         : "
        f"{start:.3f}→"
        f"{min(start + duration, total_duration):.3f}s"
    )

    actual_duration = min(
        duration,
        max(
            0.0,
            total_duration
            - start
        )
    )

    if actual_duration <= 0:

        raise RuntimeError(
            "Render window is outside "
            "the Creative Director timeline."
        )

    window_end = (
        start
        + actual_duration
    )

    # --------------------------------------------------------
    # Find clips
    # --------------------------------------------------------

    selected = []

    for clip in timeline:

        item = clip_for_window(
            clip,
            start,
            window_end
        )

        if item is not None:

            selected.append(
                item
            )

    if not selected:

        raise RuntimeError(
            "No source clips overlap "
            "the requested render window."
        )

    print(
        f"Clips to render       : "
        f"{len(selected)}"
    )

    # --------------------------------------------------------
    # Temporary directory
    # --------------------------------------------------------

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="amv_director_v3_"
        )
    )

    rendered_clips = []

    try:

        # ----------------------------------------------------
        # Render each clip
        # ----------------------------------------------------

        for index, clip in enumerate(
            selected,
            start=1
        ):

            segment_id = clip[
                "effect_segment_id"
            ]

            effect_segment = effects.get(
                segment_id
            )

            if effect_segment is None:

                # Safe fallback
                effect_segment = {
                    "editorial_role":
                        "build_zone",

                    "intensity":
                        0.35,

                    "effect_profile":
                        {},

                    "choreography":
                        [],

                    "finishing":
                        {},
                }

            output_clip = (
                temp_dir
                / f"clip_{index:04d}.mp4"
            )

            render_clip(
                ffmpeg,
                clip,
                effect_segment,
                output_clip,
                width,
                height,
                args.crf,
                args.preset
            )

            rendered_clips.append(
                output_clip
            )

        # ----------------------------------------------------
        # Concatenate
        # ----------------------------------------------------

        concat_file = (
            temp_dir
            / "concat.txt"
        )

        silent_video = (
            temp_dir
            / "silent.mp4"
        )

        concat_clips(
            ffmpeg,
            rendered_clips,
            concat_file,
            silent_video
        )

        # ----------------------------------------------------
        # Audio mux
        # ----------------------------------------------------

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        mux_audio(
            ffmpeg,
            silent_video,
            audio_path,
            output_path,
            start,
            actual_duration
        )

        print()
        print("=" * 82)
        print("✅ RENDER COMPLETE")
        print("=" * 82)

        print(
            f"Output                : "
            f"{output_path}"
        )

        print(
            f"Duration              : "
            f"{actual_duration:.3f}s"
        )

        print(
            f"Audio                 : "
            f"YES"
        )

        print(
            f"Video                 : "
            f"{width}x{height}"
        )

        print("=" * 82)

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


if __name__ == "__main__":
    main()