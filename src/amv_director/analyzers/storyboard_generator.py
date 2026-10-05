from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PLAN_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "director_edit_plan_v11.json"
)

CLIPS_DIR = (
    PROJECT_ROOT
    / "data"
    / "clips"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "storyboard"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "director_storyboard.jpg"
)


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

THUMB_W = 320
THUMB_H = 180

COLS = 4

LABEL_H = 70

MARGIN = 12


# ------------------------------------------------------------
# Load plan
# ------------------------------------------------------------

def load_plan():

    with PLAN_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ------------------------------------------------------------
# Locate video
# ------------------------------------------------------------

def find_video(video_name):

    path = CLIPS_DIR / video_name

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find source video:\n{path}"
        )

    return path


# ------------------------------------------------------------
# Extract frame
# ------------------------------------------------------------

def extract_frame(
    video_path: Path,
    timestamp: float,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open:\n{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 30.0

    target_frame = max(
        0,
        int(round(timestamp * fps))
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        target_frame,
    )

    ok, frame = cap.read()

    cap.release()

    if not ok:
        raise RuntimeError(
            f"Could not extract frame at "
            f"{timestamp:.3f}s from "
            f"{video_path.name}"
        )

    return frame


# ------------------------------------------------------------
# Thumbnail
# ------------------------------------------------------------

def make_thumbnail(frame):

    h, w = frame.shape[:2]

    target_ratio = (
        THUMB_W /
        THUMB_H
    )

    current_ratio = (
        w / h
    )

    # Center crop to 16:9.
    if current_ratio > target_ratio:

        new_w = int(
            h *
            target_ratio
        )

        x = (
            w -
            new_w
        ) // 2

        frame = frame[
            :,
            x:x + new_w
        ]

    else:

        new_h = int(
            w /
            target_ratio
        )

        y = (
            h -
            new_h
        ) // 2

        frame = frame[
            y:y + new_h,
            :
        ]

    frame = cv2.resize(
        frame,
        (
            THUMB_W,
            THUMB_H,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return frame


# ------------------------------------------------------------

# Text label
# ------------------------------------------------------------

def make_label(
    clip,
    target_start,
    target_end,
    style,
):
    label = np.zeros(
        (
            LABEL_H,
            THUMB_W,
            3,
        ),
        dtype=np.uint8,
    )

    video = clip["source_video"]
    scene_id = clip["scene_id"]

    source_start = float(
        clip["source_start"]
    )

    source_end = float(
        clip["source_end"]
    )

    quality = float(
        clip.get(
            "quality_score",
            0.0,
        )
    )

    mode = str(
        clip.get(
            "selection_mode",
            "unknown",
        )
    )

    lines = [
        (
            f"{target_start:.2f}"
            f" → "
            f"{target_end:.2f}s"
        ),
        (
            f"{style.upper()} | "
            f"{video} S{scene_id}"
        ),
        (
            f"src {source_start:.2f}"
            f" → "
            f"{source_end:.2f}s | "
            f"q={quality:.2f} | "
            f"{mode}"
        ),
    ]

    y = 20

    for line in lines:

        cv2.putText(
            label,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 22

    return label


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print()
    print("🎬 DIRECTOR STORYBOARD GENERATOR")
    print()

    data = load_plan()

    intervals = data.get(
        "intervals",
        [],
    )

    clips = []

    # Flatten V9:
    # intervals -> clips
    #
    # Also reconstruct exact target timing for every clip.
    for interval in intervals:

        target_cursor = float(
            interval["reference_start"]
        )

        style = str(
            interval.get(
                "style",
                "medium",
            )
        )

        for clip in interval.get(
            "clips",
            [],
        ):

            duration = float(
                clip["duration"]
            )

            target_start = target_cursor
            target_end = (
                target_cursor
                + duration
            )

            clips.append(
                {
                    "interval": interval,
                    "clip": clip,
                    "style": style,
                    "target_start":
                        target_start,
                    "target_end":
                        target_end,
                }
            )

            target_cursor = target_end

    clips.sort(
        key=lambda item:
            item["target_start"]
    )

    print(
        f"Planned clips: "
        f"{len(clips)}"
    )

    if not clips:
        raise RuntimeError(
            "No clips found in V9 director plan."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Cache opened video frames.
    cache = {}

    tiles = []

    for index, item in enumerate(
        clips,
        start=1,
    ):

        interval = item["interval"]
        clip = item["clip"]

        video_name = clip[
            "source_video"
        ]

        video_path = find_video(
            video_name
        )

        source_start = float(
            clip["source_start"]
        )

        source_end = float(
            clip["source_end"]
        )

        duration = max(
            0.0,
            source_end
            - source_start,
        )

        # Use the temporal center of the selected source window.
        source_time = (
            source_start
            + duration / 2.0
        )

        cache_key = (
            video_name,
            round(
                source_time,
                3,
            ),
        )

        if cache_key in cache:

            frame = cache[
                cache_key
            ]

        else:

            frame = extract_frame(
                video_path,
                source_time,
            )

            cache[
                cache_key
            ] = frame

        thumb = make_thumbnail(
            frame
        )

        label = make_label(
            clip,
            item["target_start"],
            item["target_end"],
            item["style"],
        )

        tile = np.vstack(
            [
                thumb,
                label,
            ]
        )

        # Small clip number.
        cv2.putText(
            tile,
            f"#{index}",
            (
                THUMB_W - 48,
                24,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        tiles.append(
            tile
        )

        print(
            f"[{index:02d}/{len(clips)}] "
            f"{item['target_start']:.3f}s → "
            f"{item['target_end']:.3f}s | "
            f"{video_name} "
            f"S{clip['scene_id']:02d} | "
            f"src "
            f"{source_start:.3f} → "
            f"{source_end:.3f}s | "
            f"q={clip.get('quality_score', 0):.3f}"
        )

    rows = math.ceil(
        len(tiles) /
        COLS
    )

    canvas_w = (
        COLS *
        THUMB_W
        +
        (
            COLS + 1
        )
        *
        MARGIN
    )

    canvas_h = (
        rows *
        (
            THUMB_H +
            LABEL_H
        )
        +
        (
            rows + 1
        )
        *
        MARGIN
    )

    canvas = np.zeros(
        (
            canvas_h,
            canvas_w,
            3,
        ),
        dtype=np.uint8,
    )

    for index, tile in enumerate(
        tiles
    ):

        row = (
            index //
            COLS
        )

        col = (
            index %
            COLS
        )

        x = (
            MARGIN
            +
            col *
            (
                THUMB_W +
                MARGIN
            )
        )

        y = (
            MARGIN
            +
            row *
            (
                THUMB_H +
                LABEL_H +
                MARGIN
            )
        )

        h, w = tile.shape[:2]

        canvas[
            y:y + h,
            x:x + w
        ] = tile

    cv2.imwrite(
        str(OUTPUT_FILE),
        canvas,
    )

    print()
    print("=" * 70)
    print("✅ STORYBOARD GENERATED")
    print("=" * 70)
    print(
        f"Clips    : {len(clips)}"
    )
    print(
        f"Duration : "
        f"{data['reference_duration']:.3f}s"
    )
    print(
        f"Quality  : "
        f"{data['average_quality']:.3f}"
    )
    print()
    print(
        "Saved:"
    )
    print(
        OUTPUT_FILE
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
