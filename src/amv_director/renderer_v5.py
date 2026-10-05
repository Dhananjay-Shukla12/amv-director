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
    / "effect_director_plan_v3.json"
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
    / "amv_director_v5_preview.mp4"
)


# ============================================================
# RENDER SETTINGS
# ============================================================

OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720

WORK_WIDTH = 1920
WORK_HEIGHT = 1080

FPS = 30

CRF = 18
PRESET = "veryfast"

MIN_RENDER_DURATION = 0.06


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


def clamp(
    value,
    low=0.0,
    high=1.0
):

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

def build_effect_lookup(
    effect_plan
):

    lookup = {}

    for segment in effect_plan.get(
        "segments",
        []
    ):

        try:

            segment_id = int(
                segment["segment"]
            )

            lookup[
                segment_id
            ] = segment

        except Exception:

            continue

    return lookup


# ============================================================
# CREATIVE TIMELINE
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

        segment_start = safe_float(
            segment.get(
                "start",
                cursor
            )
        )

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

            timeline.append(
                {
                    "segment_id":
                        segment_id,

                    "segment_start":
                        segment_start,

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
                        safe_float(
                            clip.get(
                                "source_start",
                                0
                            )
                        ),

                    "source_end":
                        safe_float(
                            clip.get(
                                "source_end",
                                0
                            )
                        ),
                }
            )

            cursor += duration

    return (
        timeline,
        cursor
    )


# ============================================================
# PREVIEW INTERSECTION
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

    offset_inside_clip = (
        overlap_start
        - clip_start
    )

    render_duration = (
        overlap_end
        - overlap_start
    )

    source_start = (
        clip["source_start"]
        + offset_inside_clip
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

        "effect_offset":
            offset_inside_clip,
    }


# ============================================================
# EFFECT ACTION SHIFTING
# ============================================================

def localize_effect_actions(
    effect_segment,
    offset,
    duration
):
    """
    Convert full-segment action times into times relative
    to the actual rendered clip.

    Example:

        full segment action:
            0.20 → 0.30

        preview starts:
            0.15

        localized:
            0.05 → 0.15
    """

    localized = []

    for action in effect_segment.get(
        "actions",
        []
    ):

        start = safe_float(
            action.get(
                "start",
                0
            )
        )

        end = safe_float(
            action.get(
                "end",
                start
            )
        )

        local_start = (
            start
            - offset
        )

        local_end = (
            end
            - offset
        )

        if local_end <= 0:

            continue

        if local_start >= duration:

            continue

        item = dict(
            action
        )

        item[
            "start"
        ] = max(
            0.0,
            local_start
        )

        item[
            "end"
        ] = min(
            duration,
            local_end
        )

        localized.append(
            item
        )

    return localized


def localized_effect_segment(
    effect_segment,
    clip,
    duration
):

    output = dict(
        effect_segment
    )

    output[
        "actions"
    ] = localize_effect_actions(
        effect_segment,
        safe_float(
            clip.get(
                "effect_offset",
                0
            )
        ),
        duration
    )

    return output


# ============================================================
# ACTION LOOKUPS
# ============================================================

def actions_of_type(
    actions,
    effect_type
):

    return [
        x
        for x in actions
        if x.get(
            "effect"
        ) == effect_type
    ]


def has_action(
    actions,
    effect_type
):

    return any(
        x.get("effect")
        == effect_type
        for x in actions
    )


# ============================================================
# CAMERA ZOOM EXPRESSION
# ============================================================

def build_zoom_expression(
    effect_segment,
    duration
):
    """
    Build one controlled zoom expression.

    Important:
    We ADD small zoom contributions rather than multiplying
    many zoom filters together. This prevents runaway zoom.
    """

    camera = effect_segment.get(
        "camera_profile",
        {}
    )

    base_start = safe_float(
        camera.get(
            "zoom_start",
            1.0
        ),
        1.0
    )

    base_end = safe_float(
        camera.get(
            "zoom_end",
            1.0
        ),
        1.0
    )

    # Do not allow the base camera profile to become huge.
    base_start = max(
        1.0,
        min(
            1.10,
            base_start
        )
    )

    base_end = max(
        1.0,
        min(
            1.14,
            base_end
        )
    )

    progress = (
        f"min("
        f"t/{max(duration, 0.001):.6f},"
        f"1)"
    )

    base_curve = (
        f"("
        f"{base_start:.5f}"
        f"+("
        f"{base_end:.5f}"
        f"-{base_start:.5f}"
        f")*("
        f"0.5-0.5*cos("
        f"PI*{progress}"
        f")"
        f")"
        f")"
    )

    terms = [
        base_curve
    ]

    for action in effect_segment.get(
        "actions",
        []
    ):

        effect = action.get(
            "effect",
            ""
        )

        start = safe_float(
            action.get(
                "start",
                0
            )
        )

        end = safe_float(
            action.get(
                "end",
                start
            )
        )

        if end <= start:

            continue

        # ----------------------------------------------------
        # Explicit zoom curve
        # ----------------------------------------------------

        if effect == "zoom_curve":

            start_scale = max(
                1.0,
                min(
                    1.12,
                    safe_float(
                        action.get(
                            "start_scale",
                            1.0
                        ),
                        1.0
                    )
                )
            )

            end_scale = max(
                1.0,
                min(
                    1.16,
                    safe_float(
                        action.get(
                            "end_scale",
                            1.02
                        ),
                        1.02
                    )
                )
            )

            length = max(
                0.001,
                end - start
            )

            p = (
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

            terms.append(
                (
                    f"("
                    f"1+("
                    f"{end_scale:.5f}"
                    f"-{start_scale:.5f}"
                    f")*{p}"
                    f")"
                )
            )

        # ----------------------------------------------------
        # Micro push
        # ----------------------------------------------------

        elif effect in {
            "micro_push",
            "pre_push"
        }:

            amount = max(
                0.0,
                min(
                    0.10,
                    safe_float(
                        action.get(
                            "amount",
                            0.02
                        )
                    )
                )
            )

            length = max(
                0.001,
                end - start
            )

            p = (
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

            terms.append(
                (
                    f"(1+"
                    f"{amount:.5f}"
                    f"*{p})"
            )
            )
        # ----------------------------------------------------
        # Impact snap
        # ----------------------------------------------------

        elif effect == "impact_snap":

            strength = clamp(
                safe_float(
                    action.get(
                        "strength",
                        0.5
                    )
                )
            )

            center = (
                start + end
            ) / 2.0

            sigma = max(
                0.012,
                (end - start) * 0.35
            )

            terms.append(
                (
                    f"(1+"
                    f"{0.05 * strength:.5f}"
                    f"*exp("
                    f"-pow("
                    f"(t-{center:.5f})"
                    f"/{sigma:.5f},2"
                    f")"
                    f"))"
                )
            )

    # Multiply ONLY the base curve by localized small
    # event contributions.
    expression = "*".join(
        terms
    )

    return expression


# ============================================================
# CAMERA MOVEMENT
# ============================================================

def build_camera_position(
    effect_segment
):

    actions = effect_segment.get(
        "actions",
        []
    )

    x_terms = [
        "0"
    ]

    y_terms = [
        "0"
    ]

    for action in actions:

        effect = action.get(
            "effect",
            ""
        )

        start = safe_float(
            action.get(
                "start",
                0
            )
        )

        end = safe_float(
            action.get(
                "end",
                start
            )
        )

        if end <= start:

            continue

        # ----------------------------------------------------
        # Directional motion
        # ----------------------------------------------------

        if effect == "directional_motion":

            direction = action.get(
                "direction",
                "auto"
            )

            amount = max(
                0.0,
                min(
                    0.10,
                    safe_float(
                        action.get(
                            "amount",
                            0.03
                        )
                    )
                )
            )

            if direction == "left":

                dx = -amount
                dy = 0.0

            elif direction == "right":

                dx = amount
                dy = 0.0

            elif direction == "up":

                dx = 0.0
                dy = -amount

            elif direction == "down":

                dx = 0.0
                dy = amount

            else:

                # Auto movement deliberately kept tiny.
                dx = amount * 0.30
                dy = -amount * 0.15

            length = max(
                0.001,
                end - start
            )

            p = (
                f"if("
                f"lt(t,{start:.5f}),"
                f"0,"
                f"if("
                f"gt(t,{end:.5f}),"
                f"1,"
                f"(t-{start:.5f})/"
                f"{length:.5f}"
                f")"
                f")"
            )

            x_terms.append(
                f"({dx:.5f}*{p})"
            )

            y_terms.append(
                f"({dy:.5f}*{p})"
            )

        # ----------------------------------------------------
        # Camera drift
        # ----------------------------------------------------

        elif effect == "camera_drift":

            amount = max(
                0.0,
                min(
                    0.05,
                    safe_float(
                        action.get(
                            "amount",
                            0.01
                        )
                    )
                )
            )

            direction = action.get(
                "direction",
                "auto"
            )

            if direction == "left":

                dx = -amount

            elif direction == "right":

                dx = amount

            else:

                dx = amount * 0.5

            dy = (
                amount
                * 0.35
            )

            length = max(
                0.001,
                end - start
            )

            p = (
                f"if("
                f"lt(t,{start:.5f}),"
                f"0,"
                f"if("
                f"gt(t,{end:.5f}),"
                f"1,"
                f"(t-{start:.5f})/"
                f"{length:.5f}"
                f")"
                f")"
            )

            x_terms.append(
                f"({dx:.5f}*{p})"
            )

            y_terms.append(
                f"({dy:.5f}*{p})"
            )

        # ----------------------------------------------------
        # Shake burst
        # ----------------------------------------------------

        elif effect == "shake_burst":

            amplitude = max(
                0.0,
                min(
                    0.28,
                    safe_float(
                        action.get(
                            "amplitude",
                            0.08
                        )
                    )
                )
            )

            frequency = max(
                8.0,
                min(
                    28.0,
                    safe_float(
                        action.get(
                            "frequency",
                            16
                        )
                    )
                )
            )

            center = (
                start + end
            ) / 2.0

            sigma = max(
                0.015,
                (end - start) * 0.34
            )

            envelope = (
                f"exp("
                f"-pow("
                f"(t-{center:.5f})/"
                f"{sigma:.5f},2)"
                f")"
            )

            x_terms.append(
                (
                    f"("
                    f"{70.0 * amplitude:.3f}"
                    f"*sin("
                    f"2*PI*"
                    f"{frequency:.2f}"
                    f"*t"
                    f")*"
                    f"{envelope}"
                    f")"
                )
            )

            y_terms.append(
                (
                    f"("
                    f"{42.0 * amplitude:.3f}"
                    f"*cos("
                    f"2*PI*"
                    f"{frequency * 0.81:.2f}"
                    f"*t"
                    f")*"
                    f"{envelope}"
                    f")"
                )
            )

    return (
        "+".join(x_terms),
        "+".join(y_terms)
    )


# ============================================================
# BLUR
# ============================================================

def build_blur_filter(
    actions
):

    items = []

    for action in actions:

        if action.get(
            "effect"
        ) != "motion_blur":

            continue

        start = safe_float(
            action.get(
                "start",
                0
            )
        )

        end = safe_float(
            action.get(
                "end",
                start
            )
        )

        amount = clamp(
            safe_float(
                action.get(
                    "amount",
                    0.04
                )
            )
        )

        if end <= start:
            continue

        sigma = (
            0.25
            + amount * 1.6
        )

        items.append(
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

    return items


# ============================================================
# FLASH
# ============================================================

def build_flash_expression(
    actions
):

    terms = []

    for action in actions:

        if action.get(
            "effect"
        ) != "flash":

            continue

        start = safe_float(
            action.get(
                "start",
                0
            )
        )

        end = safe_float(
            action.get(
                "end",
                start
            )
        )

        strength = clamp(
            safe_float(
                action.get(
                    "strength",
                    0.4
                )
            )
        )

        if end <= start:
            continue

        center = (
            start + end
        ) / 2.0

        sigma = max(
            0.008,
            (end - start) / 2.0
        )

        terms.append(
            (
                f"("
                f"{strength:.5f}"
                f"*exp("
                f"-pow("
                f"(t-{center:.5f})/"
                f"{sigma:.5f},2)"
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

    actions = (
        effect_segment.get(
            "actions",
            []
        )
    )

    finishing = (
        effect_segment.get(
            "finishing",
            {}
        )
    )

    role = (
        effect_segment.get(
            "editorial_role",
            "build_zone"
        )
    )

    zoom_expr = build_zoom_expression(
        effect_segment,
        duration
    )

    pan_x, pan_y = (
        build_camera_position(
            effect_segment
        )
    )

    filters = []

    # ========================================================
    # Normalize source
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
    # Dynamic scale
    # ========================================================

    scaled_w = (
        f"{WORK_WIDTH}*({zoom_expr})"
    )

    scaled_h = (
        f"{WORK_HEIGHT}*({zoom_expr})"
    )

    filters.append(
        (
            "scale="
            f"w='{scaled_w}':"
            f"h='{scaled_h}':"
            "eval=frame"
        )
    )

    # ========================================================
    # Fixed output crop with movement
    # ========================================================

    crop_x = (
        f"(in_w-out_w)/2"
        f"+({pan_x})*120"
    )

    crop_y = (
        f"(in_h-out_h)/2"
        f"+({pan_y})*70"
    )

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
    # Blur
    # ========================================================

    filters.extend(
        build_blur_filter(
            actions
        )
    )

    # ========================================================
    # Flash
    # ========================================================

    flash_expr = (
        build_flash_expression(
            actions
        )
    )

    if flash_expr:

        filters.append(
            (
                "eq="
                f"brightness='{flash_expr}'"
            )
        )

    # ========================================================
    # Finishing
    # ========================================================

    contrast = max(
        1.0,
        min(
            1.18,
            safe_float(
                finishing.get(
                    "contrast",
                    1.03
                ),
                1.03
            )
        )
    )

    saturation = max(
        0.90,
        min(
            1.18,
            safe_float(
                finishing.get(
                    "saturation",
                    1.0
                ),
                1.0
            )
        )
    )

    filters.append(
        (
            "eq="
            f"contrast={contrast:.4f}:"
            f"saturation={saturation:.4f}"
        )
    )

    # Slight sharpening.
    if role == "cinematic_zone":

        filters.append(
            "unsharp=5:5:0.18:5:5:0"
        )

    else:

        filters.append(
            "unsharp=5:5:0.28:5:5:0"
        )

    filters.append(
        "format=yuv420p"
    )

    return ",".join(
        filters
    )


# ============================================================
# RENDER CLIP
# ============================================================

def render_clip(
    ffmpeg,
    clip,
    effect_segment,
    output_path
):

    source = Path(
        clip[
            "source_path"
        ]
    )

    if not source.exists():

        raise FileNotFoundError(
            f"Source video not found:\n"
            f"{source}"
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

    # --------------------------------------------------------
    # Localize actions.
    # --------------------------------------------------------

    local_segment = (
        localized_effect_segment(
            effect_segment,
            clip,
            duration
        )
    )

    filter_string = (
        build_video_filter(
            local_segment,
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
        str(source),

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
# CONCAT
# ============================================================

def concat_video(
    ffmpeg,
    rendered_files,
    concat_file,
    output_file
):

    with open(
        concat_file,
        "w",
        encoding="utf-8"
    ) as f:

        for file_path in rendered_files:

            safe_path = (
                str(file_path)
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

        "-an",

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

        "-movflags",
        "+faststart",

        "-y",
        str(output_file),
    ]

    run_ffmpeg(
        ffmpeg,
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
        "mux audio"
    )


# ============================================================
# VERIFY
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

    text = result.stdout

    return (
        "Video:" in text,
        "Audio:" in text
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AMV Director Renderer V5"
        )
    )

    parser.add_argument(
        "--start",
        type=float,
        default=10.0
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=12.0
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
            "Invalid render window."
        )

    effect_lookup = (
        build_effect_lookup(
            effect_plan
        )
    )

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

    print(
        "=" * 82
    )

    print(
        "🎬 AMV DIRECTOR V5 RENDERER"
    )

    print(
        "=" * 82
    )

    print()

    print(
        f"Resolution        : "
        f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}"
    )

    print(
        f"Timeline           : "
        f"{start:.3f}→{end:.3f}s"
    )

    print(
        f"Duration           : "
        f"{actual_duration:.3f}s"
    )

    print(
        f"Source clips       : "
        f"{len(selected)}"
    )

    # --------------------------------------------------------
    # Temporary directory
    # --------------------------------------------------------

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="amv_director_v5_"
        )
    )

    rendered = []

    try:

        # ====================================================
        # RENDER
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

                    "camera_profile":
                        {
                            "zoom_start":
                                1.0,

                            "zoom_end":
                                1.0,
                        },

                    "actions":
                        [],

                    "finishing":
                        {
                            "contrast":
                                1.03,

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

            rendered.append(
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
            / "silent.mp4"
        )

        concat_video(
            ffmpeg,
            rendered,
            concat_file,
            silent_video
        )

        # ====================================================
        # AUDIO
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

        print()

        print(
            "=" * 82
        )

        print(
            "✅ V5 RENDER COMPLETE"
        )

        print(
            "=" * 82
        )

        print(
            f"Output       : "
            f"{output_path}"
        )

        print(
            f"Duration     : "
            f"{actual_duration:.3f}s"
        )

        print(
            f"Video        : "
            f"{'YES' if has_video else 'NO'}"
        )

        print(
            f"Audio        : "
            f"{'YES' if has_audio else 'NO'}"
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