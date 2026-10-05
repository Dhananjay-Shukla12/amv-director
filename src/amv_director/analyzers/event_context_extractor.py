from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


VIDEO_PATH = "data/references/reference_full.mp4"
EVENTS_PATH = "data/outputs/visual_events.json"

OUTPUT_DIR = Path(
    "data/outputs/event_context"
)

CONTACT_SHEET = Path(
    "data/outputs/event_context_sheet.jpg"
)


def add_label(frame, text: str):
    """Add a readable label to a frame."""

    output = frame.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 38),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def resize_keep_aspect(
    frame,
    width: int,
):
    """Resize while preserving aspect ratio."""

    height = int(
        frame.shape[0]
        * width
        / frame.shape[1]
    )

    return cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def build_contact_sheet(
    images: list,
    columns: int = 2,
):
    """Build a simple contact sheet."""

    if not images:
        return None

    rows = []

    for i in range(
        0,
        len(images),
        columns,
    ):
        row = images[
            i:i + columns
        ]

        while len(row) < columns:
            row.append(
                row[-1].copy()
            )

        rows.append(
            cv2.hconcat(row)
        )

    return cv2.vconcat(rows)


def extract_context(
    video_path: str,
    events_path: str,
    output_dir: Path,
    max_events: int = 30,
    context_seconds: float = 0.35,
):
    """
    Extract before/event/after frames for the strongest
    visual events.

    Important:
    Frames are decoded sequentially instead of repeatedly
    seeking with VideoCapture.set(). This reduces decoder
    warnings and gives us more reliable frame extraction.
    """

    with open(
        events_path,
        "r",
        encoding="utf-8",
    ) as file:
        data: dict[str, Any] = json.load(file)

    events = data["strongest_events"][
        :max_events
    ]

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        cap.release()
        raise RuntimeError(
            "Could not determine video FPS."
        )

    duration = total_frames / fps

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Extracting context for "
        f"{len(events)} events..."
    )

    # ---------------------------------------------------------
    # Build all required frame targets first.
    #
    # We decode the video ONCE and capture the needed frames
    # as we pass them.
    # ---------------------------------------------------------

    targets: dict[int, list[tuple[int, str, float]]] = {}

    for event_index, event in enumerate(
        events,
        start=1,
    ):
        event_time = float(
            event["time"]
        )

        requested_times = {
            "before": max(
                0.0,
                event_time
                - context_seconds,
            ),
            "event": event_time,
            "after": min(
                duration,
                event_time
                + context_seconds,
            ),
        }

        for label, time in requested_times.items():

            frame_number = int(
                round(time * fps)
            )

            frame_number = max(
                0,
                min(
                    total_frames - 1,
                    frame_number,
                ),
            )

            targets.setdefault(
                frame_number,
                [],
            ).append(
                (
                    event_index,
                    label,
                    time,
                )
            )

    sorted_targets = sorted(
        targets.keys()
    )

    target_index = 0

    manifest: list[dict[str, Any]] = [
        {
            "event_number": index,
            "time": float(event["time"]),
            "strength": event["strength"],
            "event_type": event["event_type"],
            "files": {},
        }
        for index, event in enumerate(
            events,
            start=1,
        )
    ]

    contact_images = []

    current_frame_number = 0

    # ---------------------------------------------------------
    # Sequential video decoding.
    # ---------------------------------------------------------

    while target_index < len(
        sorted_targets
    ):

        success, frame = cap.read()

        if not success:
            break

        if (
            current_frame_number
            >= sorted_targets[target_index]
        ):

            target_frame = (
                sorted_targets[target_index]
            )

            # There can theoretically be multiple labels
            # attached to one exact frame.
            for (
                event_index,
                label,
                requested_time,
            ) in targets[
                target_frame
            ]:

                event_dir = (
                    output_dir
                    / f"event_{event_index:03d}"
                )

                event_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                labeled = add_label(
                    frame,
                    (
                        f"Event {event_index} | "
                        f"{label.upper()} | "
                        f"{requested_time:.3f}s"
                    ),
                )

                file_path = (
                    event_dir
                    / f"{label}.jpg"
                )

                success_write = cv2.imwrite(
                    str(file_path),
                    labeled,
                )

                if not success_write:
                    raise RuntimeError(
                        f"Could not save frame: "
                        f"{file_path}"
                    )

                manifest[
                    event_index - 1
                ]["files"][label] = str(
                    file_path
                )

            target_index += 1

        current_frame_number += 1

    cap.release()

    # ---------------------------------------------------------
    # Build before/event/after strips.
    # ---------------------------------------------------------

    for item in manifest:

        files = item["files"]

        if not all(
            key in files
            for key in (
                "before",
                "event",
                "after",
            )
        ):
            continue

        before = cv2.imread(
            files["before"]
        )

        event_frame = cv2.imread(
            files["event"]
        )

        after = cv2.imread(
            files["after"]
        )

        if (
            before is None
            or event_frame is None
            or after is None
        ):
            continue

        before = resize_keep_aspect(
            before,
            240,
        )

        event_frame = resize_keep_aspect(
            event_frame,
            240,
        )

        after = resize_keep_aspect(
            after,
            240,
        )

        strip = cv2.hconcat(
            [
                before,
                event_frame,
                after,
            ]
        )

        event_number = item[
            "event_number"
        ]

        strip_path = (
            output_dir
            / f"event_{event_number:03d}"
            / "context.jpg"
        )

        cv2.imwrite(
            str(strip_path),
            strip,
        )

        contact_images.append(
            resize_keep_aspect(
                strip,
                360,
            )
        )

    # ---------------------------------------------------------
    # Master contact sheet.
    # ---------------------------------------------------------

    sheet = build_contact_sheet(
        contact_images,
        columns=2,
    )

    if sheet is not None:

        CONTACT_SHEET.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(CONTACT_SHEET),
            sheet,
        )

    # ---------------------------------------------------------
    # Save manifest.
    # ---------------------------------------------------------

    manifest_path = (
        output_dir
        / "manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print(
        "\n✅ Event context extraction complete."
    )

    print(
        f"Event folders: {output_dir}"
    )

    print(
        f"Contact sheet: {CONTACT_SHEET}"
    )

    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    extract_context(
        video_path=VIDEO_PATH,
        events_path=EVENTS_PATH,
        output_dir=OUTPUT_DIR,
    )