from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)

EVENTS_PATH = Path(
    "data/outputs/visual_events.json"
)

CONTEXT_DIR = Path(
    "data/outputs/event_context"
)

OUTPUT_PATH = Path(
    "data/outputs/semantic_events.json"
)

# ------------------------------------------------------------
# IMPORTANT:
# We test ONLY event 8 first.
# ------------------------------------------------------------

TEST_EVENT_NUMBER = 8


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

class SemanticEvent(BaseModel):

    event: Literal[
        "ordinary_motion",
        "character_reveal",
        "action_impact",
        "camera_movement",
        "flash_or_brightness_change",
        "transition",
        "dramatic_pause",
        "scene_change",
        "other",
    ]

    character_visible: bool

    character_framing: Literal[
        "close_up",
        "medium",
        "full_body",
        "wide",
        "unclear",
    ]

    action_intensity: float = Field(
        ge=0.0,
        le=1.0,
    )

    source_camera_movement: Literal[
        "none",
        "push_in",
        "pull_out",
        "pan_left",
        "pan_right",
        "shake_like",
        "rapid_motion",
        "unclear",
    ]

    motion_direction: Literal[
        "left",
        "right",
        "up",
        "down",
        "toward_camera",
        "away_from_camera",
        "mixed",
        "unclear",
    ]

    edit_motion_effect: Literal[
        "none",
        "zoom",
        "zoom_punch",
        "motion_blur",
        "shake",
        "whip_like",
        "speed_ramp_like",
        "mixed",
        "unclear",
    ]

    brightness_change: bool

    flash_effect: bool

    visual_impact: float = Field(
        ge=0.0,
        le=1.0,
    )

    dramatic_importance: float = Field(
        ge=0.0,
        le=1.0,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    description: str

    evidence: str


# ============================================================
# HELPERS
# ============================================================

def load_event(
    event_number: int,
) -> dict:

    with EVENTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    events = data[
        "strongest_events"
    ]

    if (
        event_number < 1
        or event_number > len(events)
    ):
        raise ValueError(
            f"Event {event_number} does not exist."
        )

    return events[
        event_number - 1
    ]


def image_to_base64(
    image_path: Path,
) -> str:

    with image_path.open(
        "rb",
    ) as file:

        return base64.b64encode(
            file.read()
        ).decode("utf-8")


def build_image_part(
    image_path: Path,
) -> dict:

    return {
        "type": "image",
        "data": image_to_base64(
            image_path
        ),
        "mime_type": "image/jpeg",
    }


# ============================================================
# ANALYZE ONE EVENT
# ============================================================

def analyze_event(
    client: genai.Client,
    event_number: int,
    event: dict,
) -> dict:

    event_dir = (
        CONTEXT_DIR
        / f"event_{event_number:03d}"
    )

    context_image = (
        event_dir / "context.jpg"
    )

    if not context_image.exists():
        raise FileNotFoundError(
            f"Missing context image: "
            f"{context_image}"
        )

    prompt = """
You are analyzing ONE moment from an anime video edit.

The supplied image contains three chronological panels:

LEFT   = BEFORE
CENTER = EVENT
RIGHT  = AFTER

Identify the main visual change around the CENTER panel.

Focus ONLY on information useful for automatic video editing.

Distinguish:
- movement that exists in the original anime footage
- editing effects added by the editor

Do not guess character names.

Do not call something an impact just because it is blurry.

Do not call something a camera movement unless the sequence
provides evidence of camera movement.

Use "unclear" when evidence is insufficient.

Keep numerical scores conservative.

Return ONLY JSON matching the requested schema.
""".strip()

    cv_context = {
        "event_number": event_number,
        "time": event["time"],
        "cv_event_type": event[
            "event_type"
        ],
        "cv_strength": event[
            "strength"
        ],
        "cv_motion": event[
            "motion_score"
        ],
        "cv_brightness": event[
            "brightness_score"
        ],
        "cv_edges": event[
            "edge_score"
        ],
    }

    prompt += (
        "\n\nComputer vision data:\n"
        + json.dumps(
            cv_context,
            indent=2,
        )
    )

    response = client.interactions.create(
        model=MODEL_NAME,

        input=[
            {
                "type": "text",
                "text": prompt,
            },
            build_image_part(
                context_image
            ),
        ],

        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": (
                SemanticEvent
                .model_json_schema()
            ),
        },

        generation_config={
            "thinking_level": "low",
        },
    )

    parsed = SemanticEvent.model_validate_json(
        response.output_text
    )

    return {
        "event_number": event_number,
        "time": event["time"],

        "cv_event": {
            "event_type": event[
                "event_type"
            ],
            "strength": event[
                "strength"
            ],
            "motion_score": event[
                "motion_score"
            ],
            "brightness_score": event[
                "brightness_score"
            ],
            "edge_score": event[
                "edge_score"
            ],
        },

        "semantic_event": (
            parsed.model_dump()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found."
        )

    client = genai.Client(
        api_key=api_key,
        http_options={
            "timeout": 60_000,
        },
    )

    event = load_event(
        TEST_EVENT_NUMBER
    )

    print(
        "===================================="
    )
    print(
        " AMV DIRECTOR"
    )
    print(
        " SEMANTIC TEST V0.4"
    )
    print(
        "===================================="
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Event: {TEST_EVENT_NUMBER}"
    )

    print(
        f"Time: {event['time']:.3f}s"
    )

    print(
        "\nSending ONE context image..."
    )

    try:

        result = analyze_event(
            client=client,
            event_number=TEST_EVENT_NUMBER,
            event=event,
        )

        semantic = result[
            "semantic_event"
        ]

        print("\n=== RESULT ===")

        print(
            "Event:",
            semantic["event"],
        )

        print(
            "Framing:",
            semantic[
                "character_framing"
            ],
        )

        print(
            "Action:",
            semantic[
                "action_intensity"
            ],
        )

        print(
            "Source camera:",
            semantic[
                "source_camera_movement"
            ],
        )

        print(
            "Edit motion:",
            semantic[
                "edit_motion_effect"
            ],
        )

        print(
            "Flash:",
            semantic[
                "flash_effect"
            ],
        )

        print(
            "Impact:",
            semantic[
                "visual_impact"
            ],
        )

        print(
            "Importance:",
            semantic[
                "dramatic_importance"
            ],
        )

        print(
            "Confidence:",
            semantic[
                "confidence"
            ],
        )

        print(
            "Description:",
            semantic[
                "description"
            ],
        )

        print(
            "Evidence:",
            semantic[
                "evidence"
            ],
        )

        # ----------------------------------------------------
        # Save this successful test.
        #
        # Do NOT overwrite the existing 1-7 results.
        # ----------------------------------------------------

        existing = []

        if OUTPUT_PATH.exists():

            try:

                with OUTPUT_PATH.open(
                    "r",
                    encoding="utf-8",
                ) as file:

                    old_data = json.load(
                        file
                    )

                existing = old_data.get(
                    "events",
                    [],
                )

            except (
                json.JSONDecodeError,
                OSError,
            ):
                existing = []

        # Remove an old copy of this event if present.
        existing = [
            item
            for item in existing
            if item.get("event_number")
            != TEST_EVENT_NUMBER
        ]

        existing.append(
            result
        )

        existing.sort(
            key=lambda item:
            item["event_number"]
        )

        output = {
            "model": MODEL_NAME,
            "events_requested": 30,
            "events_completed": len(
                existing
            ),
            "events": existing,
        }

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "\n✅ Event 8 saved."
        )

        print(
            f"JSON: {OUTPUT_PATH}"
        )

    except Exception as error:

        print(
            "\n❌ Event analysis failed."
        )

        print(
            type(error).__name__,
            ":",
            error,
        )


if __name__ == "__main__":
    main()