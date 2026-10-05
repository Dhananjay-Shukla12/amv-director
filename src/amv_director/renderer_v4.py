import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CREATIVE_PLAN_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "creative_director_plan.json"
)

EFFECT_PLAN_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "effect_director_plan_v2.json"
)

DEFAULT_AUDIO_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "reference_audio.wav"
)

DEFAULT_OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "renders"
    / "amv_director_v4_preview.mp4"
)


# ============================================================
# RENDER SETTINGS
# ============================================================

OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720

FPS = 30

CRF = 18
PRESET = "veryfast"

# Oversized working canvas gives room for camera movement.
WORK_WIDTH = 1920
WORK_HEIGHT = 1080

MIN_RENDER_DURATION = 0.06


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, (list, tuple)):
            if not value:
                return default
            value = value[0]

        return float(value)

    except Exception:
        return default


def clamp(value, low=0.0, high=1.0):
    return max(
        low,
        min(high, value)
    )


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


def run_ffmpeg(
    ffmpeg,
    command,
    description
):
    print(
        f"▶ {description}"
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
            f"FFmpeg failed: {description}"
        )

    return result.stdout


# ============================================================
# EFFECT LOOKUP
# ============================================================

def build_effect_lookup(effect_plan):

    lookup = {}

    for segment in effect_plan.get(
        "segments",
        []
    ):
        try:
            key = int(
                segment["segment"]
            )
            lookup[key] = segment
        except Exception:
            continue

    return lookup


# ============================================================
# TIMELINE
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

        try:
            segment_id = int(
                segment["segment"]
            )
        except Exception:
            continue

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

                    "timeline_start":
                        cursor,

                    "timeline_end":
                        cursor + duration,

                    "duration":
                        duration,

                    "source_video":
                        clip.get(
                            "source_video",
                            ""
                        ),

                    "source_path":
                        clip.get(
                            "source_path",
                            ""
                        ),

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
# PREVIEW WINDOW INTERSECTION
# ============================================================

def intersect_clip(
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

    render_duration = (
        overlap_end
        - overlap_start
    )

    source_start = (
        clip["source_start"]
        + offset
    )

    return {
        **clip,

        "preview_start":
            overlap_start,

        "preview_end":
            overlap_end,

        "render_source_start":
            source_start,

        "render_duration":
            render_duration,
    }


# ============================================================
# CAMERA EXPRESSIONS
# ============================================================

def build_camera_expressions(
    effect_segment,
    duration
):
    """
    Build dynamic camera expressions.

    Zoom:
        handled by dynamic SCALE.

    Pan:
        handled by moving the final crop window.

    Shake:
        handled by moving the final crop window with
        a short Gaussian-like burst envelope.

    The final crop dimensions always remain fixed at
    OUTPUT_WIDTH x OUTPUT_HEIGHT.
    """

    camera = effect_segment.get(
        "camera_profile",
        {}
    )

    actions = effect_segment.get(
        "actions",
        []
    )

    duration = max(
        0.001,
        duration
    )

    zoom_start = safe_float(
        camera.get(
            "zoom_start",
            1.0
        ),
        1.0
    )

    zoom_end = safe_float(
        camera.get(
            "zoom_end",
            1.02
        ),
        1.02
    )

    # --------------------------------------------------------
    # Base zoom curve
    # --------------------------------------------------------

    base_progress = (
        f"min("
        f"t/{duration:.6f},"
        f"1"
        f")"
    )

    base_zoom = (
        f"("
        f"{zoom_start:.5f}"
        f"+("
        f"{zoom_end:.5f}"
        f"-{zoom_start:.5f}"
        f")*("
        f"0.5-0.5*cos("
        f"PI*{base_progress}"
        f")"
        f")"
        f")"
    )

    zoom_terms = [
        base_zoom
    ]

    pan_x_terms = [
        "0"
    ]

    pan_y_terms = [
        "0"
    ]

    shake_x_terms = [
        "0"
    ]

    shake_y_terms = [
        "0"
    ]

    # ========================================================
    # TIMED ACTIONS
    # ========================================================

    for item in actions:

        effect = item.get(
            "effect",
            ""
        )

        start = safe_float(
            item.get(
                "start",
                0
            )
        )

        end = safe_float(
            item.get(
                "end",
                start
            )
        )

        if end <= start:
            continue

        # ----------------------------------------------------
        # CAMERA ZOOM / PRE ZOOM
        # ----------------------------------------------------

        if effect in {
            "camera_zoom",
            "pre_zoom"
        }:

            start_scale = safe_float(
                item.get(
                    "start_scale",
                    zoom_start
                ),
                zoom_start
            )

            end_scale = safe_float(
                item.get(
                    "end_scale",
                    zoom_end
                ),
                zoom_end
            )

            length = max(
                0.001,
                end - start
            )

            progress = (
                f"if("
                f"lt(t,{start:.5f}),"
                f"0,"
                f"if("
                f"gt(t,{end:.5f}),"
                f"1,"
                f"0.5-0.5*cos("
                f"PI*(t-{start:.5f})"
                f"/{length:.5f}"
                f")"
                f")"
                f")"
            )

            zoom_terms.append(
                (
                    f"("
                    f"{start_scale:.5f}"
                    f"+("
                    f"{end_scale:.5f}"
                    f"-{start_scale:.5f}"
                    f")*{progress}"
                    f")"
                )
            )

        # ----------------------------------------------------
        # IMPACT SNAP
        # ----------------------------------------------------

        elif effect == "impact_snap":

            strength = clamp(
                safe_float(
                    item.get(
                        "strength",
                        0.7
                    )
                )
            )

            center = (
                start + end
            ) / 2.0

            sigma = max(
                0.018,
                (end - start) * 0.45
            )

            zoom_terms.append(
                (
                    f"("
                    f"1+"
                    f"{0.08 * strength:.5f}"
                    f"*exp("
                    f"-pow("
                    f"(t-{center:.5f})"
                    f"/{sigma:.5f},"
                    f"2"
                    f")"
                    f")"
                    f")"
                )
            )

        # ----------------------------------------------------
        # CAMERA PAN
        # ----------------------------------------------------

        elif effect in {
            "camera_pan",
            "directional_motion"
        }:

            x = safe_float(
                item.get(
                    "x",
                    0
                )
            )

            y = safe_float(
                item.get(
                    "y",
                    0
                )
            )

            length = max(
                0.001,
                end - start
            )

            progress = (
                f"if("
                f"lt(t,{start:.5f}),"
                f"0,"
                f"if("
                f"gt(t,{end:.5f}),"
                f"1,"
                f"(t-{start:.5f})/{length:.5f}"
                f")"
                f")"
            )

            pan_x_terms.append(
                f"({x:.5f}*{progress})"
            )

            pan_y_terms.append(
                f"({y:.5f}*{progress})"
            )

        # ----------------------------------------------------
        # SHAKE BURST
        # ----------------------------------------------------

        elif effect == "shake_burst":

            amplitude = clamp(
                safe_float(
                    item.get(
                        "amplitude",
                        0.4
                    )
                )
            )

            frequency = safe_float(
                item.get(
                    "frequency",
                    18
                ),
                18
            )

            center = (
                start + end
            ) / 2.0

            sigma = max(
                0.02,
                (end - start) * 0.38
            )

            envelope = (
                f"exp("
                f"-pow("
                f"(t-{center:.5f})"
                f"/{sigma:.5f},"
                f"2"
                f")"
                f")"
            )

            shake_x_terms.append(
                (
                    f"("
                    f"{65.0 * amplitude:.3f}"
                    f"*sin("
                    f"2*PI*"
                    f"{frequency:.3f}"
                    f"*t"
                    f")*"
                    f"{envelope}"
                    f")"
                )
            )

            shake_y_terms.append(
                (
                    f"("
                    f"{42.0 * amplitude:.3f}"
                    f"*cos("
                    f"2*PI*"
                    f"{frequency * 0.83:.3f}"
                    f"*t"
                    f")*"
                    f"{envelope}"
                    f")"
                )
            )

    # ========================================================
    # COMBINE EXPRESSIONS
    # ========================================================

    if len(zoom_terms) == 1:

        zoom_expr = zoom_terms[0]

    else:

        zoom_expr = "*".join(
            zoom_terms
        )

    pan_x_expr = "+".join(
        pan_x_terms
    )

    pan_y_expr = "+".join(
        pan_y_terms
    )

    shake_x_expr = "+".join(
        shake_x_terms
    )

    shake_y_expr = "+".join(
        shake_y_terms
    )

    # ========================================================
    # DYNAMIC SCALE
    # ========================================================

    scaled_width = (
        f"{WORK_WIDTH}*({zoom_expr})"
    )

    scaled_height = (
        f"{WORK_HEIGHT}*({zoom_expr})"
    )

    # ========================================================
    # FIXED FINAL CROP + MOVEMENT
    # ========================================================

    crop_x = (
        f"(in_w-out_w)/2"
        f"+({pan_x_expr})*160"
        f"+({shake_x_expr})"
    )

    crop_y = (
        f"(in_h-out_h)/2"
        f"+({pan_y_expr})*90"
        f"+({shake_y_expr})"
    )

    return (
        scaled_width,
        scaled_height,
        crop_x,
        crop_y
    )


# ============================================================
# TIMED MOTION BLUR
# ============================================================

def build_blur_filters(
    actions
):

    filters = []

    for item in actions:

        if item.get(
            "effect"
        ) != "motion_blur":

            continue

        start = safe_float(
            item.get(
                "start",
                0
            )
        )

        end = safe_float(
            item.get(
                "end",
                start
            )
        )

        if end <= start:
            continue

        amount = clamp(
            safe_float(
                item.get(
                    "amount",
                    0.08
                )
            )
        )

        sigma = (
            0.30
            + 2.0
            * amount
        )

        filters.append(
            (
                f"gblur="
                f"sigma={sigma:.4f}:"
                f"steps=1:"
                f"enable='between("
                f"t,"
                f"{start:.5f},"
                f"{end:.5f}"
                f")'"
            )
        )

    return filters


# ============================================================
# TIMED FLASH
# ============================================================

def build_flash_expression(
    actions
):

    terms = []

    for item in actions:

        if item.get(
            "effect"
        ) != "flash":

            continue

        start = safe_float(
            item.get(
                "start",
                0
            )
        )

        end = safe_float(
            item.get(
                "end",
                start
            )
        )

        if end <= start:
            continue

        strength = clamp(
            safe_float(
                item.get(
                    "strength",
                    0.5
                )
            )
        )

        center = (
            start + end
        ) / 2.0

        sigma = max(
            0.01,
            (end - start) / 2.2
        )

        terms.append(
            (
                f"("
                f"{strength:.5f}"
                f"*exp("
                f"-pow("
                f"(t-{center:.5f})"
                f"/{sigma:.5f},"
                f"2"
                f")"
                f")"
                f")"
            )
        )

    if not terms:
        return None

    return "+".join(
        terms
    )


# ============================================================
# VIDEO FILTER
# ============================================================

def build_video_filter(
    effect_segment,
    duration
):

    actions = effect_segment.get(
        "actions",
        []
    )

    finishing = effect_segment.get(
        "finishing",
        {}
    )

    role = effect_segment.get(
        "editorial_role",
        "build_zone"
    )

    (
        scaled_width,
        scaled_height,
        crop_x,
        crop_y,
    ) = build_camera_expressions(
        effect_segment,
        duration
    )

    filters = []

    # ========================================================
    # 1. NORMALIZE TO WORKING CANVAS
    # ========================================================

    filters.append(
        (
            "scale="
            f"{WORK_WIDTH}:"
            f"{WORK_HEIGHT}:"
            "force_original_aspect_ratio=increase"
        )
    )

    filters.append(
        (
            "crop="
            f"{WORK_WIDTH}:"
            f"{WORK_HEIGHT}"
        )
    )

    # ========================================================
    # 2. DYNAMIC CAMERA SCALE
    # ========================================================

    filters.append(
        (
            "scale="
            f"w='{scaled_width}':"
            f"h='{scaled_height}':"
            "eval=frame"
        )
    )

    # ========================================================
    # 3. FIXED OUTPUT CROP
    # ========================================================

    filters.append(
        (
            "crop="
            f"{OUTPUT_WIDTH}:"
            f"{OUTPUT_HEIGHT}:"
            f"x='{crop_x}':"
            f"y='{crop_y}'"
        )
    )

    # ========================================================
    # 4. TIMED BLUR
    # ========================================================

    filters.extend(
        build_blur_filters(
            actions
        )
    )

    # ========================================================
    # 5. TIMED FLASH
    # ========================================================

    flash_expression = (
        build_flash_expression(
            actions
        )
    )

    if flash_expression:

        filters.append(
            (
                "eq="
                f"brightness="
                f"'{flash_expression}'"
            )
        )

    # ========================================================
    # 6. CONTRAST + SATURATION
    # ========================================================

    contrast = clamp(
        safe_float(
            finishing.get(
                "contrast",
                0.10
            )
        )
    )

    saturation = safe_float(
        finishing.get(
            "saturation",
            1.0
        ),
        1.0
    )

    saturation = max(
        0.80,
        min(
            1.35,
            saturation
        )
    )

    contrast_multiplier = (
        1.0
        + 0.35
        * contrast
    )

    filters.append(
        (
            "eq="
            f"contrast={contrast_multiplier:.4f}:"
            f"saturation={saturation:.4f}"
        )
    )

    # ========================================================
    # 7. SHARPEN
    # ========================================================

    if role == "cinematic_zone":

        filters.append(
            (
                "unsharp="
                "5:5:0.25:"
                "5:5:0"
            )
        )

    else:

        filters.append(
            (
                "unsharp="
                "5:5:0.35:"
                "5:5:0"
            )
        )

    # ========================================================
    # 8. OUTPUT FORMAT
    # ========================================================

    filters.append(
        "format=yuv420p"
    )

    return ",".join(
        filters
    )


# ============================================================
# RENDER ONE CLIP
# ============================================================

def render_clip(
    ffmpeg,
    clip,
    effect_segment,
    output_path
):

    source_path = Path(
        clip[
            "source_path"
        ]
    )

    if not source_path.exists():

        raise FileNotFoundError(
            f"Source video not found:\n"
            f"{source_path}"
        )

    duration = max(
        MIN_RENDER_DURATION,
        safe_float(
            clip[
                "render_duration"
            ]
        )
    )

    source_start = max(
        0.0,
        safe_float(
            clip[
                "render_source_start"
            ]
        )
    )

    filter_string = (
        build_video_filter(
            effect_segment,
            duration
        )
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

        "-an",

        "-r",
        str(FPS),

        "-c:v",
        "libx264",

        "-preset",
        PRESET,

        "-crf",
        str(CRF),

        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        "-y",
        str(output_path),
    ]

    run_ffmpeg(
        ffmpeg,
        command,
        (
            f"render "
            f"{clip['source_video']} "
            f"S{clip.get('scene_id')} "
            f"{duration:.3f}s"
        )
    )


# ============================================================
# CONCATENATE VIDEO
# ============================================================

def concat_video(
    ffmpeg,
    rendered_files,
    concat_file,
    silent_output
):

    with open(
        concat_file,
        "w",
        encoding="utf-8"
    ) as f:

        for path in rendered_files:

            safe_path = (
                str(path)
                .replace(
                    "'",
                    "'\\''"
                )
            )

            f.write(
                f"file '{safe_path}'\n"
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

        "-c:v",
        "libx264",

        "-preset",
        PRESET,

        "-crf",
        str(CRF),

        "-pix_fmt",
        "yuv420p",

        "-r",
        str(FPS),

        "-an",

        "-movflags",
        "+faststart",

        "-y",
        str(silent_output),
    ]

    run_ffmpeg(
        ffmpeg,
        command,
        "concatenate rendered video"
    )


# ============================================================
# MUX AUDIO
# ============================================================

def mux_audio(
    ffmpeg,
    silent_video,
    audio_path,
    final_output,
    audio_start,
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
        f"{audio_start:.6f}",

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

        "-ar",
        "44100",

        "-ac",
        "2",

        "-shortest",

        "-movflags",
        "+faststart",

        "-y",
        str(final_output),
    ]

    run_ffmpeg(
        ffmpeg,
        command,
        "mux music into final video"
    )


# ============================================================
# VERIFY OUTPUT
# ============================================================

def verify_output(
    ffmpeg,
    output_path
):

    command = [

        ffmpeg,

        "-hide_banner",

        "-i",
        str(output_path),

        "-f",
        "null",

        "-"
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    output = result.stdout

    has_video = (
        "Video:" in output
    )

    has_audio = (
        "Audio:" in output
    )

    return (
        has_video,
        has_audio
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AMV Director Renderer V4"
        )
    )

    parser.add_argument(
        "--start",
        type=float,
        default=10.0,
        help="Timeline start in seconds."
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=12.0,
        help="Preview duration."
    )

    parser.add_argument(
        "--audio",
        default=str(
            DEFAULT_AUDIO_PATH
        )
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT_PATH
        )
    )

    args = parser.parse_args()

    ffmpeg = (
        imageio_ffmpeg
        .get_ffmpeg_exe()
    )

    creative_plan = load_json(
        CREATIVE_PLAN_PATH
    )

    effect_plan = load_json(
        EFFECT_PLAN_PATH
    )

    audio_path = Path(
        args.audio
    ).expanduser().resolve()

    output_path = Path(
        args.output
    ).expanduser().resolve()

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio not found:\n"
            f"{audio_path}"
        )

    # ========================================================
    # FLATTEN TIMELINE
    # ========================================================

    timeline, total_duration = (
        flatten_timeline(
            creative_plan
        )
    )

    start = max(
        0.0,
        args.start
    )

    requested_duration = max(
        0.10,
        args.duration
    )

    end = min(
        total_duration,
        start + requested_duration
    )

    actual_duration = (
        end - start
    )

    if actual_duration <= 0:

        raise RuntimeError(
            "Requested render window "
            "is outside the Creative Director timeline."
        )

    # ========================================================
    # EFFECT LOOKUP
    # ========================================================

    effect_lookup = (
        build_effect_lookup(
            effect_plan
        )
    )

    # ========================================================
    # FIND SOURCE CLIPS
    # ========================================================

    selected = []

    for clip in timeline:

        item = intersect_clip(
            clip,
            start,
            end
        )

        if item is not None:

            selected.append(
                item
            )

    if not selected:

        raise RuntimeError(
            "No clips overlap render window."
        )

    # ========================================================
    # HEADER
    # ========================================================

    print(
        "=" * 82
    )

    print(
        "🎬 AMV DIRECTOR V4 RENDERER"
    )

    print(
        "=" * 82
    )

    print()

    print(
        f"Resolution       : "
        f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}"
    )

    print(
        f"Timeline          : "
        f"{start:.3f}→{end:.3f}s"
    )

    print(
        f"Duration          : "
        f"{actual_duration:.3f}s"
    )

    print(
        f"Source clips      : "
        f"{len(selected)}"
    )

    print(
        f"Audio             : "
        f"{audio_path}"
    )

    # ========================================================
    # TEMP DIRECTORY
    # ========================================================

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="amv_director_v4_"
        )
    )

    rendered_files = []

    try:

        # ====================================================
        # RENDER EACH CLIP
        # ====================================================

        for index, clip in enumerate(
            selected,
            start=1
        ):

            segment_id = int(
                clip[
                    "segment_id"
                ]
            )

            effect_segment = (
                effect_lookup.get(
                    segment_id
                )
            )

            if effect_segment is None:

                effect_segment = {

                    "editorial_role":
                        "build_zone",

                    "camera_profile": {

                        "zoom_start":
                            1.0,

                        "zoom_end":
                            1.02,
                    },

                    "actions":
                        [],

                    "finishing": {

                        "contrast":
                            0.10,

                        "saturation":
                            1.0,
                    },
                }

            output_clip = (
                temp_dir
                / f"clip_{index:04d}.mp4"
            )

            render_clip(
                ffmpeg,
                clip,
                effect_segment,
                output_clip
            )

            rendered_files.append(
                output_clip
            )

        # ====================================================
        # CONCAT
        # ====================================================

        concat_file = (
            temp_dir
            / "concat.txt"
        )

        silent_video = (
            temp_dir
            / "silent_video.mp4"
        )

        concat_video(
            ffmpeg,
            rendered_files,
            concat_file,
            silent_video
        )

        # ====================================================
        # AUDIO MUX
        # ====================================================

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

        # ====================================================
        # VERIFY
        # ====================================================

        has_video, has_audio = (
            verify_output(
                ffmpeg,
                output_path
            )
        )

        # ====================================================
        # RESULT
        # ====================================================

        print()

        print(
            "=" * 82
        )

        print(
            "✅ RENDER COMPLETE"
        )

        print(
            "=" * 82
        )

        print(
            f"Output        : "
            f"{output_path}"
        )

        print(
            f"Duration      : "
            f"{actual_duration:.3f}s"
        )

        print(
            f"Video stream  : "
            f"{'YES' if has_video else 'NO'}"
        )

        print(
            f"Audio stream  : "
            f"{'YES' if has_audio else 'NO'}"
        )

        print()
        print(
            "V4 executes timed effect actions "
            "inside each source clip."
        )

        print(
            "Camera movement is localized rather "
            "than using continuous vibration."
        )

        print(
            "=" * 82
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


if __name__ == "__main__":
    main()