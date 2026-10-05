from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)

IMAGE_PATH = Path(
    "data/outputs/event_context/event_008/context.jpg"
)


def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found."
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Missing image: {IMAGE_PATH}"
        )

    with IMAGE_PATH.open("rb") as f:
        image_bytes = f.read()

    print("Image bytes:", len(image_bytes))
    print("Model:", MODEL)
    print("Sending request...")

    client = genai.Client(
        api_key=api_key,
        http_options={
            "timeout": 60000
        }
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_text(
                text=(
                    "Look at this 3-panel anime editing "
                    "context image. In one short sentence, "
                    "describe what visually changes from "
                    "left to center to right."
                )
            ),
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg",
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config={
                "thinking_level": "low"
            },
            max_output_tokens=100,
        ),
    )

    print("\n=== RESPONSE ===")
    print(response.text)


if __name__ == "__main__":
    main()