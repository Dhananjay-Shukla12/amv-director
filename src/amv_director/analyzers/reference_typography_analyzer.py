from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import cv2


def get_video_info(video_path: Path) -> tuple[float, float, float]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps else 0.0

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cap.release()

    return duration, fps, width / height if height else 0.0


def extract_sample_frames(
    video_path: Path,
    output_dir: Path,
    interval: float,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    duration, fps, aspect = get_video_info(video_path)

    print(f"Video duration: {duration:.3f}s")
    print(f"FPS: {fps:.3f}")
    print(f"Aspect ratio: {aspect:.3f}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    events = []

    timestamp = 0.0
    index = 1

    while timestamp < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()

        if not ok:
            break

        output_path = output_dir / f"frame_{index:04d}_{timestamp:07.3f}.jpg"
        cv2.imwrite(str(output_path), frame)

        events.append(
            {
                "index": index,
                "timestamp": round(timestamp, 3),
                "path": str(output_path),
            }
        )

        timestamp += interval
        index += 1

    cap.release()

    return events


def create_contact_sheet(
    frames: list[dict],
    output_path: Path,
    columns: int = 4,
    thumb_width: int = 480,
) -> None:
    if not frames:
        return

    images = []

    for item in frames:
        image = cv2.imread(item["path"])

        if image is None:
            continue

        height, width = image.shape[:2]

        scale = thumb_width / width
        thumb_height = max(1, int(height * scale))

        image = cv2.resize(
            image,
            (thumb_width, thumb_height),
            interpolation=cv2.INTER_AREA,
        )

        cv2.putText(
            image,
            f'{item["timestamp"]:.2f}s',
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        images.append(image)

    if not images:
        return

    rows = math.ceil(len(images) / columns)

    tile_h, tile_w = images[0].shape[:2]

    sheet = cv2.createImage = None

    canvas = 255 * (
        __import__("numpy").ones(
            (
                rows * tile_h,
                columns * tile_w,
                3,
            ),
            dtype="uint8",
        )
    )

    for i, image in enumerate(images):
        row = i // columns
        col = i % columns

        y = row * tile_h
        x = col * tile_w

        canvas[y : y + tile_h, x : x + tile_w] = image

    cv2.imwrite(str(output_path), canvas)

    print(f"✅ Contact sheet: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        default="data/references/reference_full.mp4",
    )

    parser.add_argument(
        "--output",
        default="data/outputs/reference_typography",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    frames = extract_sample_frames(
        video_path,
        output_dir,
        args.interval,
    )

    contact_sheet = output_dir / "contact_sheet.jpg"

    create_contact_sheet(
        frames,
        contact_sheet,
    )

    print()
    print("===== TYPOGRAPHY AUDIT V1 =====")
    print(f"Frames sampled: {len(frames)}")
    print(f"Interval:       {args.interval}s")
    print(f"Output:         {output_dir}")
    print("===============================")


if __name__ == "__main__":
    main()