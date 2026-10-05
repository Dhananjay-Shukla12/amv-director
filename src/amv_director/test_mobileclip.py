from pathlib import Path

import cv2
import numpy as np
import torch
import open_clip
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

REFERENCE_VIDEO = (
    BASE_DIR
    / "data"
    / "references"
    / "reference_video.mp4"
)

SOURCE_VIDEO = (
    BASE_DIR
    / "data"
    / "clips"
    / "anime_clip1.mp4"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED = "dfndr2b"

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frame(
    video_path,
    time_seconds
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open:\n{video_path}"
        )

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        time_seconds * 1000.0
    )

    ok, frame = cap.read()

    cap.release()

    if not ok:
        raise RuntimeError(
            f"Could not extract frame at "
            f"{time_seconds:.2f}s from "
            f"{video_path}"
        )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(
        frame
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a,
    b
):

    a = a / (
        a.norm(
            dim=-1,
            keepdim=True
        )
        + 1e-8
    )

    b = b / (
        b.norm(
            dim=-1,
            keepdim=True
        )
        + 1e-8
    )

    return float(
        (a @ b.T).item()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("🧠 MOBILECLIP2 TEST")
    print("=" * 72)

    print()
    print(
        f"Model  : {MODEL_NAME}"
    )

    print(
        f"Device : {DEVICE}"
    )

    print()
    print(
        "Loading model..."
    )

    # --------------------------------------------------------
    # Load MobileCLIP2-S0 through current OpenCLIP support.
    # --------------------------------------------------------

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
            device=DEVICE,
        )
    )

    tokenizer = (
        open_clip.get_tokenizer(
            MODEL_NAME
        )
    )

    model.eval()

    print(
        "✅ Model loaded"
    )

    # --------------------------------------------------------
    # Reference frame
    # --------------------------------------------------------

    print()
    print(
        "Extracting reference frame..."
    )

    reference_image = extract_frame(
        REFERENCE_VIDEO,
        13.0
    )

    # --------------------------------------------------------
    # Source frame
    # --------------------------------------------------------

    print(
        "Extracting source frame..."
    )

    source_image = extract_frame(
        SOURCE_VIDEO,
        7.2
    )

    reference_input = (
        preprocess(
            reference_image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    source_input = (
        preprocess(
            source_image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Image embeddings
    # --------------------------------------------------------

    print()
    print(
        "Generating image embeddings..."
    )

    with torch.no_grad():

        reference_features = (
            model.encode_image(
                reference_input
            )
        )

        source_features = (
            model.encode_image(
                source_input
            )
        )

        reference_features = (
            reference_features
            / reference_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        source_features = (
            source_features
            / source_features.norm(
                dim=-1,
                keepdim=True
            )
        )

    similarity = float(
        (
            reference_features
            @
            source_features.T
        ).item()
    )

    print(
        f"✅ Image embedding generated"
    )

    print()
    print(
        f"Reference ↔ Source cosine similarity: "
        f"{similarity:.4f}"
    )

    # --------------------------------------------------------
    # Zero-shot visual descriptions
    # --------------------------------------------------------

    labels = [

        "a close-up of a character",

        "a wide shot of a character",

        "a character fighting",

        "a character attacking",

        "a character standing",

        "a character looking at another character",

        "two characters facing each other",

        "a dramatic anime scene",

        "a fast action scene",

        "a calm cinematic scene",

        "a dark anime scene",

        "a bright anime scene",

        "an explosion or impact",

        "a close-up of eyes",

        "a full body character shot",

        "a landscape or environment shot",
    ]

    print()
    print(
        "Running visual descriptions..."
    )

    text = tokenizer(
        labels
    ).to(DEVICE)

    with torch.no_grad():

        text_features = (
            model.encode_text(
                text
            )
        )

        text_features = (
            text_features
            / text_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        logits = (
            100.0
            * reference_features
            @ text_features.T
        )

        probabilities = (
            logits.softmax(
                dim=-1
            )[0]
        )

    values = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    ranked = sorted(
        zip(
            labels,
            values
        ),
        key=lambda x:
            x[1],
        reverse=True
    )

    print()
    print(
        "REFERENCE FRAME TOP DESCRIPTIONS"
    )

    for label, probability in ranked[:8]:

        print(
            f"  {probability:6.3f}  "
            f"{label}"
        )

    print()
    print("=" * 72)
    print(
        "✅ MOBILECLIP TEST COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()