from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


REGIONS = [
    (4.0, 6.0),
    (7.5, 9.0),
    (11.0, 13.5),
    (15.0, 21.0),
    (39.0, 43.0),
    (55.0, 58.0),
]


def get_video_info(video_path: Path) -> tuple[float, float, float, float]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    duration = frame_count / fps if fps else 0.0

    cap.release()

    return duration, fps, width, height


def extract_region_frames(
    video_path: Path,
    output_dir: Path,
    start: float,
    end: float,
    interval: float,
    region_index: int,
) -> list[dict]:
    region_dir = output_dir / f"region_{region_index:02d}"
    region_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []

    timestamp = start
    index = 1

    while timestamp <= end + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

        ok, frame = cap.read()

        if not ok:
            print(f"⚠️ Could not decode frame at {timestamp:.3f}s")
            timestamp += interval
            continue

        output_path = (
            region_dir
            / f"frame_{index:04d}_{timestamp:07.3f}.jpg"
        )

        cv2.imwrite(str(output_path), frame)

        frames.append(
            {
                "timestamp": round(timestamp, 3),
                "path": str(output_path),
            }
        )

        index += 1
        timestamp += interval

    cap.release()

    return frames


def create_contact_sheet(
    frames: list[dict],
    output_path: Path,
    columns: int = 5,
    thumb_width: int = 320,
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

        # Timestamp label
        cv2.rectangle(
            image,
            (0, 0),
            (145, 38),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            image,
            f'{item["timestamp"]:.1f}s',
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        images.append(image)

    if not images:
        return

    rows = math.ceil(len(images) / columns)

    tile_h = images[0].shape[0]
    tile_w = images[0].shape[1]

    canvas = np.zeros(
        (
            rows * tile_h,
            columns * tile_w,
            3,
        ),
        dtype=np.uint8,
    )

    for i, image in enumerate(images):
        row = i // columns
        col = i % columns

        y = row * tile_h
        x = col * tile_w

        canvas[
            y:y + tile_h,
            x:x + tile_w
        ] = image

    cv2.imwrite(str(output_path), canvas)

    print(f"✅ Contact sheet: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-resolution reference typography audit."
    )

    parser.add_argument(
        "--video",
        default="data/references/reference_full.mp4",
    )

    parser.add_argument(
        "--output",
        default="data/outputs/reference_typography_v2",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    duration, fps, width, height = get_video_info(video_path)

    print(f"Video duration: {duration:.3f}s")
    print(f"FPS: {fps:.3f}")
    print(f"Resolution: {int(width)}x{int(height)}")
    print(f"Sampling interval: {args.interval}s")
    print()

    all_regions = []

    for region_index, (start, end) in enumerate(REGIONS, start=1):
        print(
            f"Region {region_index}: "
            f"{start:.1f}s → {end:.1f}s"
        )

        frames = extract_region_frames(
            video_path=video_path,
            output_dir=output_dir,
            start=start,
            end=end,
            interval=args.interval,
            region_index=region_index,
        )

        contact_sheet = (
            output_dir
            / f"region_{region_index:02d}_contact_sheet.jpg"
        )

        create_contact_sheet(
            frames=frames,
            output_path=contact_sheet,
        )

        all_regions.append(
            {
                "region": region_index,
                "start": start,
                "end": end,
                "frame_count": len(frames),
                "contact_sheet": str(contact_sheet),
            }
        )

        print(f"Frames: {len(frames)}")
        print()

    print("================================")
    print("REFERENCE TYPOGRAPHY AUDIT V2")
    print("================================")
    print(f"Regions: {len(all_regions)}")
    print(f"Output:  {output_dir}")
    print()

    total_frames = sum(
        region["frame_count"]
        for region in all_regions
    )

    print(f"Total frames: {total_frames}")
    print()
    print("Generated contact sheets:")

    for region in all_regions:
        print(
            f'  Region {region["region"]}: '
            f'{region["contact_sheet"]}'
        )


if __name__ == "__main__":
    main()