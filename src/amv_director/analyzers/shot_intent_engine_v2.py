from __future__ import annotations

import json
import math
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import open_clip
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

INPUT_JSON = (
    BASE_DIR
    / "data/outputs/clip_analysis/all_clips_analysis.json"
)

OUTPUT_JSON = (
    BASE_DIR
    / "data/outputs/clip_analysis/semantic_shot_analysis_v2.json"
)


# ============================================================
# DEVICE / MODEL
# ============================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED = "dfndr2b"


# ============================================================
# PROMPTS
# ============================================================

PROMPT_GROUPS = {

    "shot_type": {
        "close_up": (
            "a tightly framed anime close-up shot "
            "focused on a character"
        ),
        "medium_shot": (
            "an anime medium shot showing a character "
            "from roughly the waist or chest"
        ),
        "wide_shot": (
            "an anime wide shot showing the full character "
            "and significant surroundings"
        ),
        "extreme_wide": (
            "an anime extreme wide establishing shot "
            "where the environment occupies most of the frame"
        ),
    },

    "subject": {
        "face": "an anime character's face",
        "eyes": "a close-up of anime character eyes",
        "full_body": "a full body anime character",
        "two_characters": "two anime characters in the same shot",
        "multiple_characters": "multiple anime characters in the same shot",
        "environment": "an anime environment or landscape",
        "object": "an anime object or visual prop",
    },

    "action": {
        "combat": "anime characters fighting",
        "attack": "an anime character attacking",
        "defense": "an anime character defending or blocking",
        "movement": "an anime character moving quickly",
        "standing": "an anime character standing still",
        "walking": "an anime character walking",
        "interaction": "anime characters interacting",
    },

    "energy": {
        "calm": "a calm cinematic anime scene",
        "dramatic": "a dramatic anime scene",
        "action": "a high energy anime action scene",
        "impact": "a powerful anime impact moment",
        "intense": "an extremely intense anime moment",
    },

    "composition": {
        "centered_character": (
            "an anime character centered in the frame"
        ),
        "dynamic_composition": (
            "a dynamically composed anime action shot"
        ),
        "face_focus": (
            "an anime frame strongly focused on a face"
        ),
        "environment_focus": (
            "an anime frame dominated by the environment"
        ),
    },
}


# ============================================================
# NUMERIC HELPERS
# ============================================================

def safe_float(value, default=0.0) -> float:
    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:
        return default


def clamp(value: float, low=0.0, high=1.0) -> float:
    return float(max(low, min(high, value)))


def softmax(values: List[float], temperature=0.07) -> List[float]:

    arr = np.asarray(values, dtype=np.float32)

    if len(arr) == 0:
        return []

    temperature = max(temperature, 0.001)

    arr = arr / temperature
    arr -= np.max(arr)

    exp_values = np.exp(arr)

    denom = float(np.sum(exp_values))

    if denom <= 0:
        return [0.0] * len(values)

    return (exp_values / denom).tolist()


# ============================================================
# PATH
# ============================================================

def resolve_video_path(path_value: str) -> Path:

    path = Path(path_value)

    if path.is_absolute():
        return path

    return BASE_DIR / path


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frame(
    video_path: Path,
    timestamp: float,
) -> Image.Image | None:

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    try:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            max(0.0, timestamp) * 1000.0,
        )

        ok, frame = cap.read()

        if not ok or frame is None:
            return None

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        return Image.fromarray(frame)

    finally:

        cap.release()


def sample_timestamps(
    start: float,
    end: float,
) -> List[float]:

    duration = max(
        0.01,
        end - start,
    )

    if duration <= 0.5:
        return [
            start + duration * 0.50
        ]

    if duration <= 1.2:
        return [
            start + duration * 0.35,
            start + duration * 0.65,
        ]

    return [
        start + duration * 0.25,
        start + duration * 0.50,
        start + duration * 0.75,
    ]


# ============================================================
# IMAGE METRICS
# ============================================================

def calculate_frame_metrics(
    image: Image.Image,
) -> Dict[str, float]:

    frame = np.asarray(image)

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2GRAY,
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    edges = cv2.Canny(
        gray,
        80,
        160,
    )

    edge_density = float(
        np.mean(edges > 0)
    )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "edge_density": edge_density,
    }


# ============================================================
# MODEL
# ============================================================

def load_model():

    print("=" * 72)
    print("🧠 SHOT INTENT ENGINE V2")
    print("=" * 72)

    print()
    print(f"Model  : {MODEL_NAME}")
    print(f"Device : {DEVICE}")

    print()
    print("Loading MobileCLIP2...")

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
            device=DEVICE,
        )
    )

    tokenizer = open_clip.get_tokenizer(
        MODEL_NAME
    )

    model.eval()

    print("✅ Model loaded")

    return (
        model,
        preprocess,
        tokenizer,
    )


# ============================================================
# TEXT EMBEDDINGS
# ============================================================

def build_text_embeddings(
    model,
    tokenizer,
):

    all_embeddings = {}

    with torch.no_grad():

        for group_name, prompts in PROMPT_GROUPS.items():

            names = list(prompts.keys())

            texts = [
                prompts[name]
                for name in names
            ]

            tokens = tokenizer(texts)

            tokens = tokens.to(DEVICE)

            features = model.encode_text(
                tokens
            )

            features = (
                features
                / features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            all_embeddings[group_name] = {
                name: features[i].detach()
                for i, name in enumerate(names)
            }

    return all_embeddings


# ============================================================
# IMAGE EMBEDDING
# ============================================================

def encode_image(
    model,
    preprocess,
    image: Image.Image,
):

    tensor = preprocess(
        image
    ).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        features = model.encode_image(
            tensor
        )

        features = (
            features
            / features.norm(
                dim=-1,
                keepdim=True,
            )
        )

    return features[0].detach()


# ============================================================
# CLASSIFY GROUP
# ============================================================

def classify_group(
    image_feature,
    group_embeddings,
):

    names = list(
        group_embeddings.keys()
    )

    raw_scores = []

    with torch.no_grad():

        for name in names:

            score = torch.sum(
                image_feature
                * group_embeddings[name]
            ).item()

            raw_scores.append(
                float(score)
            )

    probabilities = softmax(
        raw_scores,
        temperature=0.07,
    )

    order = np.argsort(
        probabilities
    )[::-1]

    best_index = int(order[0])

    second_index = (
        int(order[1])
        if len(order) > 1
        else best_index
    )

    best_score = probabilities[
        best_index
    ]

    second_score = probabilities[
        second_index
    ]

    margin = (
        best_score
        - second_score
    )

    # --------------------------------------------------------
    # Confidence is NOT probability.
    #
    # It measures how clearly the best label beats
    # the second-best label.
    # --------------------------------------------------------

    confidence = clamp(
        margin / 0.18
    )

    best_name = names[
        best_index
    ]

    # Weak evidence should remain uncertain.
    if confidence < 0.12:
        best_name = "uncertain"

    return {
        "scores": {
            names[i]: float(
                probabilities[i]
            )
            for i in range(len(names))
        },
        "raw_scores": {
            names[i]: float(
                raw_scores[i]
            )
            for i in range(len(names))
        },
        "best": best_name,
        "confidence": round(
            confidence,
            4,
        ),
        "margin": round(
            margin,
            4,
        ),
    }


# ============================================================
# TEMPORAL AGGREGATION
# ============================================================

def aggregate_group(
    frame_results: List[Dict],
    group_name: str,
):

    group_results = [
        item["groups"][group_name]
        for item in frame_results
    ]

    labels = [
        item["best"]
        for item in group_results
    ]

    usable_labels = [
        label
        for label in labels
        if label != "uncertain"
    ]

    if usable_labels:

        dominant = Counter(
            usable_labels
        ).most_common(1)[0][0]

        consistency = (
            Counter(usable_labels)[dominant]
            / len(usable_labels)
        )

    else:

        dominant = "uncertain"
        consistency = 0.0

    confidence = float(
        np.mean(
            [
                item["confidence"]
                for item in group_results
            ]
        )
    )

    # Temporal disagreement reduces trust.
    confidence *= (
        0.5 + 0.5 * consistency
    )

    score_dicts = [
        item["scores"]
        for item in group_results
    ]

    names = list(
        score_dicts[0].keys()
    )

    averaged_scores = {}

    for name in names:

        averaged_scores[name] = float(
            np.mean(
                [
                    scores[name]
                    for scores in score_dicts
                ]
            )
        )

    return {
        "dominant": dominant,
        "confidence": round(
            confidence,
            4,
        ),
        "consistency": round(
            consistency,
            4,
        ),
        "scores": averaged_scores,
    }


# ============================================================
# QUALITY
# ============================================================

def quality_profile(
    frame_metrics: List[Dict[str, float]],
    original_metrics: Dict[str, float],
):

    brightness = np.mean([
        x["brightness"]
        for x in frame_metrics
    ])

    contrast = np.mean([
        x["contrast"]
        for x in frame_metrics
    ])

    sharpness = np.mean([
        x["sharpness"]
        for x in frame_metrics
    ])

    edge_density = np.mean([
        x["edge_density"]
        for x in frame_metrics
    ])

    original_motion = safe_float(
        original_metrics.get("motion")
    )

    # --------------------------------------------------------
    # Near-black detector
    # --------------------------------------------------------

    near_black = (
        brightness < 5
        and contrast < 8
        and sharpness < 20
        and edge_density < 0.005
    )

    # --------------------------------------------------------
    # Detail score
    # --------------------------------------------------------

    sharpness_score = clamp(
        math.log1p(max(sharpness, 0))
        / math.log1p(1500)
    )

    edge_score = clamp(
        edge_density / 0.08
    )

    contrast_score = clamp(
        contrast / 100.0
    )

    detail_score = (
        sharpness_score * 0.45
        + edge_score * 0.30
        + contrast_score * 0.25
    )

    # --------------------------------------------------------
    # Motion score
    # --------------------------------------------------------

    motion_score = clamp(
        original_motion / 0.25
    )

    # --------------------------------------------------------
    # Visual energy
    #
    # Deliberately DOES NOT rely on CLIP probabilities.
    # --------------------------------------------------------

    visual_energy = clamp(
        detail_score * 0.45
        + motion_score * 0.40
        + contrast_score * 0.15
    )

    if near_black:

        quality_flag = "reject"

    elif detail_score < 0.18:

        quality_flag = "weak"

    else:

        quality_flag = "usable"

    return {
        "brightness": round(
            float(brightness),
            3,
        ),
        "contrast": round(
            float(contrast),
            3,
        ),
        "sharpness": round(
            float(sharpness),
            3,
        ),
        "edge_density": round(
            float(edge_density),
            5,
        ),
        "motion_source": round(
            original_motion,
            4,
        ),
        "detail_score": round(
            float(detail_score),
            4,
        ),
        "motion_score": round(
            float(motion_score),
            4,
        ),
        "visual_energy": round(
            float(visual_energy),
            4,
        ),
        "quality_flag": quality_flag,
    }


# ============================================================
# DIRECTOR PROFILE
# ============================================================

def build_director_profile(
    aggregated,
    quality,
):

    shot = aggregated["shot_type"]
    subject = aggregated["subject"]
    action = aggregated["action"]
    energy = aggregated["energy"]
    composition = aggregated["composition"]

    # --------------------------------------------------------
    # Character strength
    # --------------------------------------------------------

    character_strength = max(
        subject["scores"].get("face", 0.0),
        subject["scores"].get("eyes", 0.0),
        subject["scores"].get("full_body", 0.0),
        subject["scores"].get("two_characters", 0.0),
        subject["scores"].get("multiple_characters", 0.0),
    )

    environment_strength = (
        subject["scores"].get(
            "environment",
            0.0,
        )
    )

    object_strength = (
        subject["scores"].get(
            "object",
            0.0,
        )
    )

    # --------------------------------------------------------
    # Action strength
    # --------------------------------------------------------

    combat_strength = max(
        action["scores"].get(
            "combat",
            0.0,
        ),
        action["scores"].get(
            "attack",
            0.0,
        ),
        action["scores"].get(
            "defense",
            0.0,
        ),
    )

    movement_strength = max(
        action["scores"].get(
            "movement",
            0.0,
        ),
        action["scores"].get(
            "attack",
            0.0,
        ),
    )

    # --------------------------------------------------------
    # Semantic confidence
    # --------------------------------------------------------

    confidence_values = [
        shot["confidence"],
        subject["confidence"],
        action["confidence"],
        energy["confidence"],
        composition["confidence"],
    ]

    semantic_confidence = float(
        np.mean(confidence_values)
    )

    # --------------------------------------------------------
    # Scene role
    # --------------------------------------------------------

    if quality["quality_flag"] == "reject":

        scene_role = "reject"

    elif semantic_confidence < 0.15:

        scene_role = "unknown"

    elif combat_strength > 0.25:

        scene_role = "combat"

    elif energy["scores"].get(
        "impact",
        0.0,
    ) > 0.23:

        scene_role = "impact"

    elif energy["scores"].get(
        "dramatic",
        0.0,
    ) > 0.23:

        scene_role = "dramatic"

    elif environment_strength > character_strength:

        scene_role = "environment"

    elif object_strength > character_strength:

        scene_role = "object"

    else:

        scene_role = "character"

    # --------------------------------------------------------
    # Preferred editing roles
    # --------------------------------------------------------

    usable_for = []

    visual_energy = quality[
        "visual_energy"
    ]

    face_emphasis = (
        max(
            subject["scores"].get(
                "face",
                0.0,
            ),
            subject["scores"].get(
                "eyes",
                0.0,
            ),
            composition["scores"].get(
                "face_focus",
                0.0,
            ),
        )
    )

    if quality["quality_flag"] != "reject":

        if visual_energy < 0.35:
            usable_for.append(
                "cinematic"
            )

        if face_emphasis > 0.27:
            usable_for.append(
                "face_emphasis"
            )

        if combat_strength > 0.22:
            usable_for.append(
                "action"
            )

        if energy["scores"].get(
            "impact",
            0.0,
        ) > 0.21:
            usable_for.append(
                "impact"
            )

        if movement_strength > 0.21:
            usable_for.append(
                "motion"
            )

    return {
        "scene_role": scene_role,

        "semantic_confidence": round(
            semantic_confidence,
            4,
        ),

        "preferred_shot": shot["dominant"],

        "shot_confidence": shot[
            "confidence"
        ],

        "character_strength": round(
            character_strength,
            4,
        ),

        "environment_strength": round(
            environment_strength,
            4,
        ),

        "combat_strength": round(
            combat_strength,
            4,
        ),

        "movement_strength": round(
            movement_strength,
            4,
        ),

        "face_emphasis": round(
            face_emphasis,
            4,
        ),

        "visual_energy": quality[
            "visual_energy"
        ],

        "usable_for": usable_for,
    }


# ============================================================
# SCENE ANALYSIS
# ============================================================

def analyze_scene(
    model,
    preprocess,
    text_embeddings,
    video_path: Path,
    scene: Dict,
):

    start = safe_float(
        scene.get("start")
    )

    end = safe_float(
        scene.get("end")
    )

    duration = max(
        0.01,
        end - start,
    )

    timestamps = sample_timestamps(
        start,
        end,
    )

    frame_results = []

    for timestamp in timestamps:

        image = extract_frame(
            video_path,
            timestamp,
        )

        if image is None:
            continue

        metrics = calculate_frame_metrics(
            image
        )

        feature = encode_image(
            model,
            preprocess,
            image,
        )

        groups = {}

        for group_name, embeddings in (
            text_embeddings.items()
        ):

            groups[group_name] = (
                classify_group(
                    feature,
                    embeddings,
                )
            )

        frame_results.append(
            {
                "timestamp": round(
                    timestamp,
                    3,
                ),
                "metrics": metrics,
                "groups": groups,
            }
        )

    if not frame_results:

        return {
            "analysis_status": "failed",
            "start": start,
            "end": end,
            "duration": duration,
        }

    # --------------------------------------------------------
    # Aggregate semantic groups
    # --------------------------------------------------------

    aggregated = {}

    for group_name in PROMPT_GROUPS:

        aggregated[group_name] = (
            aggregate_group(
                frame_results,
                group_name,
            )
        )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    original_metrics = (
        scene.get(
            "brightness",
            0.0,
        ),
        scene.get(
            "contrast",
            0.0,
        ),
        scene.get(
            "sharpness",
            0.0,
        ),
    )

    original_dict = {
        "brightness": safe_float(
            scene.get(
                "brightness"
            )
        ),
        "contrast": safe_float(
            scene.get(
                "contrast"
            )
        ),
        "sharpness": safe_float(
            scene.get(
                "sharpness"
            )
        ),
        "motion": safe_float(
            scene.get(
                "motion"
            )
        ),
    }

    frame_metrics = [
        item["metrics"]
        for item in frame_results
    ]

    quality = quality_profile(
        frame_metrics,
        original_dict,
    )

    director_profile = (
        build_director_profile(
            aggregated,
            quality,
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    return {
        "scene_id": scene.get(
            "scene_id"
        ),
        "start": round(
            start,
            3,
        ),
        "end": round(
            end,
            3,
        ),
        "duration": round(
            duration,
            3,
        ),
        "analysis_status": "success",

        "frames_analyzed": len(
            frame_results
        ),

        "sample_timestamps": [
            item["timestamp"]
            for item in frame_results
        ],

        "original_metrics": original_dict,

        "quality": quality,

        "semantic": {
            "shot_type": aggregated[
                "shot_type"
            ],
            "subject": aggregated[
                "subject"
            ],
            "action": aggregated[
                "action"
            ],
            "energy": aggregated[
                "energy"
            ],
            "composition": aggregated[
                "composition"
            ],
        },

        "director_profile": director_profile,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_JSON.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_JSON}"
        )

    with open(
        INPUT_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        videos = json.load(f)

    if not isinstance(videos, list):

        raise RuntimeError(
            "Expected all_clips_analysis.json "
            "to contain a list."
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, preprocess, tokenizer = (
        load_model()
    )

    # --------------------------------------------------------
    # Text embeddings
    # --------------------------------------------------------

    print()
    print("Building text embeddings...")

    text_embeddings = (
        build_text_embeddings(
            model,
            tokenizer,
        )
    )

    print("✅ Text embeddings ready")

    # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    total_scenes = sum(
        len(
            video.get(
                "scenes",
                [],
            )
        )
        for video in videos
    )

    print()
    print(
        f"Scenes to analyze: {total_scenes}"
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    results = []

    processed = 0

    for video in videos:

        video_name = video.get(
            "video",
            "unknown_video",
        )

        video_path_value = video.get(
            "path"
        )

        if not video_path_value:
            continue

        video_path = resolve_video_path(
            video_path_value
        )

        print()
        print("-" * 72)
        print(
            f"🎬 {video_name}"
        )
        print("-" * 72)

        for scene in video.get(
            "scenes",
            [],
        ):

            processed += 1

            scene_id = scene.get(
                "scene_id"
            )

            print(
                f"[{processed:02d}/{total_scenes}] "
                f"Scene {scene_id} ... ",
                end="",
                flush=True,
            )

            try:

                result = analyze_scene(
                    model,
                    preprocess,
                    text_embeddings,
                    video_path,
                    scene,
                )

                result["video"] = (
                    video_name
                )

                result["video_path"] = (
                    str(video_path)
                )

                results.append(result)

                if (
                    result["analysis_status"]
                    == "success"
                ):

                    profile = (
                        result[
                            "director_profile"
                        ]
                    )

                    print(
                        f"✅ "
                        f"{profile['scene_role']} | "
                        f"{profile['preferred_shot']} | "
                        f"energy={profile['visual_energy']:.2f} | "
                        f"conf={profile['semantic_confidence']:.2f} | "
                        f"quality={result['quality']['quality_flag']}"
                    )

                else:

                    print("⚠️ failed")

            except Exception as exc:

                print(
                    f"❌ {type(exc).__name__}: {exc}"
                )

                results.append(
                    {
                        "scene_id": scene_id,
                        "video": video_name,
                        "video_path": str(
                            video_path
                        ),
                        "analysis_status": "error",
                        "error": str(exc),
                    }
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    successful = [
        x
        for x in results
        if x.get(
            "analysis_status"
        ) == "success"
    ]

    quality_counts = Counter(
        x["quality"]["quality_flag"]
        for x in successful
    )

    role_counts = Counter(
        x["director_profile"]["scene_role"]
        for x in successful
    )

    shot_counts = Counter(
        x["director_profile"]["preferred_shot"]
        for x in successful
    )

    average_energy = (
        float(
            np.mean(
                [
                    x["director_profile"][
                        "visual_energy"
                    ]
                    for x in successful
                ]
            )
        )
        if successful
        else 0.0
    )

    average_confidence = (
        float(
            np.mean(
                [
                    x["director_profile"][
                        "semantic_confidence"
                    ]
                    for x in successful
                ]
            )
        )
        if successful
        else 0.0
    )

    output = {

        "metadata": {
            "engine_version": "v2",
            "model": MODEL_NAME,
            "pretrained": PRETRAINED,
            "device": DEVICE,
            "total_scenes": total_scenes,
            "successful_scenes": len(
                successful
            ),
            "failed_scenes": total_scenes
            - len(successful),
        },

        "summary": {

            "quality": dict(
                quality_counts
            ),

            "scene_roles": dict(
                role_counts
            ),

            "shot_types": dict(
                shot_counts
            ),

            "average_visual_energy": round(
                average_energy,
                4,
            ),

            "average_semantic_confidence": round(
                average_confidence,
                4,
            ),
        },

        "scenes": results,
    }

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 72)
    print("✅ SHOT INTENT ENGINE V2 COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Scenes      : {total_scenes}"
    )

    print(
        f"Successful  : {len(successful)}"
    )

    print(
        f"Rejected    : "
        f"{quality_counts.get('reject', 0)}"
    )

    print(
        f"Weak        : "
        f"{quality_counts.get('weak', 0)}"
    )

    print()
    print(
        "Average visual energy     : "
        f"{average_energy:.3f}"
    )

    print(
        "Average semantic confidence: "
        f"{average_confidence:.3f}"
    )

    print()
    print("Scene roles:")

    for key, value in role_counts.most_common():

        print(
            f"  {key:12s}: {value}"
        )

    print()
    print("Shot types:")

    for key, value in shot_counts.most_common():

        print(
            f"  {key:12s}: {value}"
        )

    print()
    print("Output:")

    print(
        OUTPUT_JSON
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()