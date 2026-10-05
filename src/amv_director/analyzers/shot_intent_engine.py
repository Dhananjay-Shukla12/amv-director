from __future__ import annotations

import json
import math
from pathlib import Path
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

INPUT_JSON = BASE_DIR / "data/outputs/clip_analysis/all_clips_analysis.json"
OUTPUT_JSON = (
    BASE_DIR
    / "data/outputs/clip_analysis/semantic_shot_analysis.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED = "dfndr2b"


# ============================================================
# PROMPT GROUPS
# ============================================================

SHOT_TYPE_PROMPTS = {
    "close_up": "a close-up anime shot of a character's face",
    "medium_shot": "a medium shot of an anime character",
    "wide_shot": "a wide shot showing an anime character and surroundings",
    "extreme_wide": "an extreme wide anime shot showing the environment",
}

SUBJECT_PROMPTS = {
    "face": "an anime character face",
    "eyes": "a close-up of anime character eyes",
    "full_body": "a full body anime character shot",
    "two_characters": "two anime characters together",
    "multiple_characters": "multiple anime characters together",
    "environment": "an anime environment or landscape",
    "object": "an anime shot focused on an object",
}

ACTION_PROMPTS = {
    "combat": "anime characters fighting in combat",
    "attack": "an anime character attacking",
    "defense": "an anime character defending",
    "movement": "an anime character moving quickly",
    "standing": "an anime character standing still",
    "walking": "an anime character walking",
    "interaction": "anime characters interacting with each other",
}

ENERGY_PROMPTS = {
    "calm": "a calm cinematic anime scene",
    "dramatic": "a highly dramatic anime scene",
    "action": "a high energy anime action scene",
    "impact": "a powerful anime impact or attack moment",
    "intense": "an extremely intense anime scene",
}

COMPOSITION_PROMPTS = {
    "centered_character": "an anime character centered in the frame",
    "dynamic_composition": "a dynamically composed anime action shot",
    "face_focus": "an anime shot strongly focused on the face",
    "environment_focus": "an anime shot where the environment dominates the frame",
}

ALL_GROUPS = {
    "shot_type": SHOT_TYPE_PROMPTS,
    "subject": SUBJECT_PROMPTS,
    "action": ACTION_PROMPTS,
    "energy": ENERGY_PROMPTS,
    "composition": COMPOSITION_PROMPTS,
}


# ============================================================
# HELPERS
# ============================================================

def softmax(values: List[float], temperature: float = 0.07) -> List[float]:
    arr = np.asarray(values, dtype=np.float32)

    if temperature <= 0:
        temperature = 0.07

    arr = arr / temperature

    arr = arr - np.max(arr)

    exp_values = np.exp(arr)

    total = float(np.sum(exp_values))

    if total <= 0:
        return [0.0 for _ in values]

    return (exp_values / total).tolist()


def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}

    keys = dicts[0].keys()

    result = {}

    for key in keys:
        result[key] = float(
            np.mean([float(d.get(key, 0.0)) for d in dicts])
        )

    return result


def normalize_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return BASE_DIR / path


def safe_float(value, default=0.0) -> float:
    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value
    except Exception:
        return default


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frame(
    video_path: Path,
    timestamp: float,
) -> Image.Image | None:

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"⚠️ Could not open video: {video_path}")
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)

        ok, frame = cap.read()

        if not ok or frame is None:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame)

        return image

    finally:
        cap.release()


def sample_timestamps(
    start: float,
    end: float,
) -> List[float]:

    duration = max(0.01, end - start)

    if duration <= 0.5:
        return [
            start + duration * 0.5
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
# MODEL SETUP
# ============================================================

def load_model():

    print("=" * 72)
    print("🧠 SHOT INTENT ENGINE")
    print("=" * 72)

    print()
    print(f"Model  : {MODEL_NAME}")
    print(f"Device : {DEVICE}")

    print()
    print("Loading MobileCLIP2...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=DEVICE,
    )

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model.eval()

    print("✅ Model loaded")

    return model, preprocess, tokenizer


# ============================================================
# TEXT EMBEDDINGS
# ============================================================

def build_text_embeddings(
    model,
    tokenizer,
) -> Dict[str, Dict[str, torch.Tensor]]:

    embeddings = {}

    with torch.no_grad():

        for group_name, prompts in ALL_GROUPS.items():

            names = list(prompts.keys())
            texts = list(prompts.values())

            tokens = tokenizer(texts)

            if DEVICE == "mps":
                tokens = tokens.to("mps")

            text_features = model.encode_text(tokens)

            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True,
            )

            embeddings[group_name] = {
                name: text_features[i].detach()
                for i, name in enumerate(names)
            }

    return embeddings


# ============================================================
# IMAGE EMBEDDING
# ============================================================

def image_embedding(
    model,
    preprocess,
    image: Image.Image,
) -> torch.Tensor:

    tensor = preprocess(image).unsqueeze(0)

    if DEVICE == "mps":
        tensor = tensor.to("mps")

    with torch.no_grad():

        features = model.encode_image(tensor)

        features = features / features.norm(
            dim=-1,
            keepdim=True,
        )

    return features[0].detach()


# ============================================================
# ZERO-SHOT GROUP CLASSIFICATION
# ============================================================

def classify_group(
    image_feature: torch.Tensor,
    group_embeddings: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, float], str]:

    names = list(group_embeddings.keys())

    similarities = []

    for name in names:

        text_feature = group_embeddings[name]

        score = torch.sum(
            image_feature * text_feature
        ).item()

        similarities.append(float(score))

    probabilities = softmax(
        similarities,
        temperature=0.07,
    )

    scores = {
        name: float(probabilities[i])
        for i, name in enumerate(names)
    }

    best_name = names[int(np.argmax(probabilities))]

    return scores, best_name


# ============================================================
# DERIVED DIRECTOR FEATURES
# ============================================================

def derive_features(
    results: Dict,
) -> Dict[str, float | str]:

    shot_scores = results["shot_type_scores"]
    subject_scores = results["subject_scores"]
    action_scores = results["action_scores"]
    energy_scores = results["energy_scores"]
    composition_scores = results["composition_scores"]

    # --------------------------------------------------------
    # Aggregate conceptual scores
    # --------------------------------------------------------

    combat_level = max(
        action_scores.get("combat", 0.0),
        action_scores.get("attack", 0.0),
        action_scores.get("defense", 0.0),
    )

    movement_level = max(
        action_scores.get("movement", 0.0),
        action_scores.get("attack", 0.0),
    )

    character_focus = max(
        subject_scores.get("face", 0.0),
        subject_scores.get("eyes", 0.0),
        subject_scores.get("full_body", 0.0),
    )

    environment_focus = subject_scores.get(
        "environment",
        0.0,
    )

    dramatic_level = max(
        energy_scores.get("dramatic", 0.0),
        energy_scores.get("intense", 0.0),
    )

    action_level = max(
        energy_scores.get("action", 0.0),
        energy_scores.get("impact", 0.0),
    )

    impact_level = energy_scores.get(
        "impact",
        0.0,
    )

    calm_level = energy_scores.get(
        "calm",
        0.0,
    )

    # --------------------------------------------------------
    # Dominant labels
    # --------------------------------------------------------

    dominant_shot_type = max(
        shot_scores,
        key=shot_scores.get,
    )

    dominant_subject = max(
        subject_scores,
        key=subject_scores.get,
    )

    dominant_action = max(
        action_scores,
        key=action_scores.get,
    )

    dominant_energy = max(
        energy_scores,
        key=energy_scores.get,
    )

    dominant_composition = max(
        composition_scores,
        key=composition_scores.get,
    )

    # --------------------------------------------------------
    # Director-friendly semantic interpretation
    # --------------------------------------------------------

    if combat_level > 0.45:
        scene_role = "combat"

    elif impact_level > 0.40:
        scene_role = "impact"

    elif dramatic_level > 0.40:
        scene_role = "dramatic"

    elif calm_level > 0.40:
        scene_role = "calm"

    elif environment_focus > character_focus:
        scene_role = "environment"

    else:
        scene_role = "character"

    # --------------------------------------------------------
    # Numeric visual energy
    # --------------------------------------------------------

    visual_energy = float(
        min(
            1.0,
            max(
                0.0,
                (
                    action_level * 0.35
                    + dramatic_level * 0.25
                    + combat_level * 0.25
                    + impact_level * 0.15
                ),
            ),
        )
    )

    # --------------------------------------------------------
    # Face emphasis
    # --------------------------------------------------------

    face_emphasis = max(
        subject_scores.get("face", 0.0),
        subject_scores.get("eyes", 0.0),
        composition_scores.get("face_focus", 0.0),
    )

    # --------------------------------------------------------
    # Character vs environment
    # --------------------------------------------------------

    character_vs_environment = float(
        max(
            0.0,
            min(
                1.0,
                character_focus
                /
                (
                    character_focus
                    + environment_focus
                    + 1e-8
                ),
            ),
        )
    )

    return {
        "dominant_shot_type": dominant_shot_type,
        "dominant_subject": dominant_subject,
        "dominant_action": dominant_action,
        "dominant_energy": dominant_energy,
        "dominant_composition": dominant_composition,
        "scene_role": scene_role,
        "visual_energy": round(visual_energy, 4),
        "combat_level": round(combat_level, 4),
        "movement_level": round(movement_level, 4),
        "dramatic_level": round(dramatic_level, 4),
        "impact_level": round(impact_level, 4),
        "calm_level": round(calm_level, 4),
        "character_focus": round(character_focus, 4),
        "environment_focus": round(environment_focus, 4),
        "face_emphasis": round(face_emphasis, 4),
        "character_vs_environment": round(
            character_vs_environment,
            4,
        ),
    }


# ============================================================
# ANALYZE ONE SCENE
# ============================================================

def analyze_scene(
    model,
    preprocess,
    text_embeddings,
    video_path: Path,
    scene: Dict,
) -> Dict:

    scene_id = scene.get("scene_id")

    start = safe_float(scene.get("start"))
    end = safe_float(scene.get("end"))

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

        feature = image_embedding(
            model,
            preprocess,
            image,
        )

        frame_result = {
            "timestamp": round(timestamp, 3),
            "groups": {},
        }

        for group_name, group_embeddings in text_embeddings.items():

            scores, dominant = classify_group(
                feature,
                group_embeddings,
            )

            frame_result["groups"][group_name] = {
                "scores": scores,
                "dominant": dominant,
            }

        frame_results.append(
            frame_result
        )

    # --------------------------------------------------------
    # Handle failure
    # --------------------------------------------------------

    if not frame_results:

        return {
            "scene_id": scene_id,
            "start": start,
            "end": end,
            "duration": duration,
            "analysis_status": "failed",
            "frames_analyzed": 0,
        }

    # --------------------------------------------------------
    # Aggregate frame scores
    # --------------------------------------------------------

    aggregated = {}

    dominant_labels = {}

    for group_name in ALL_GROUPS.keys():

        per_frame_scores = [
            frame["groups"][group_name]["scores"]
            for frame in frame_results
        ]

        aggregated_scores = mean_dict(
            per_frame_scores
        )

        aggregated[group_name] = aggregated_scores

        # Most frequent dominant label.
        labels = [
            frame["groups"][group_name]["dominant"]
            for frame in frame_results
        ]

        dominant_labels[group_name] = max(
            set(labels),
            key=labels.count,
        )

    semantic_results = {
        "shot_type_scores": aggregated["shot_type"],
        "subject_scores": aggregated["subject"],
        "action_scores": aggregated["action"],
        "energy_scores": aggregated["energy"],
        "composition_scores": aggregated["composition"],
    }

    derived = derive_features(
        semantic_results
    )

    # --------------------------------------------------------
    # Preserve original analysis metrics
    # --------------------------------------------------------

    original_metrics = {
        "brightness": safe_float(
            scene.get("brightness")
        ),
        "contrast": safe_float(
            scene.get("contrast")
        ),
        "sharpness": safe_float(
            scene.get("sharpness")
        ),
        "edge_density": safe_float(
            scene.get("edge_density")
        ),
        "motion": safe_float(
            scene.get("motion")
        ),
    }

    # --------------------------------------------------------
    # Final record
    # --------------------------------------------------------

    result = {
        "scene_id": scene_id,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration, 3),
        "analysis_status": "success",
        "frames_analyzed": len(frame_results),
        "sample_timestamps": [
            frame["timestamp"]
            for frame in frame_results
        ],
        "original_metrics": original_metrics,
        "semantic": {
            "shot_type": dominant_labels["shot_type"],
            "subject": dominant_labels["subject"],
            "action": dominant_labels["action"],
            "energy": dominant_labels["energy"],
            "composition": dominant_labels["composition"],
            **derived,
        },
        "scores": {
            "shot_type": aggregated["shot_type"],
            "subject": aggregated["subject"],
            "action": aggregated["action"],
            "energy": aggregated["energy"],
            "composition": aggregated["composition"],
        },
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_JSON.exists():

        raise FileNotFoundError(
            f"Input analysis not found:\n{INPUT_JSON}"
        )

    print()
    print(f"Input : {INPUT_JSON}")
    print(f"Output: {OUTPUT_JSON}")
    print()

    # --------------------------------------------------------
    # Load source scene database
    # --------------------------------------------------------

    with open(
        INPUT_JSON,
        "r",
        encoding="utf-8",
    ) as f:

        videos = json.load(f)

    if not isinstance(videos, list):

        raise RuntimeError(
            "Expected all_clips_analysis.json to contain a list."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, preprocess, tokenizer = load_model()

    # --------------------------------------------------------
    # Prepare text embeddings ONCE
    # --------------------------------------------------------

    print()
    print("Building text embeddings...")

    text_embeddings = build_text_embeddings(
        model,
        tokenizer,
    )

    print("✅ Text embeddings ready")

    # --------------------------------------------------------
    # Analyze scenes
    # --------------------------------------------------------

    all_results = []

    total_scenes = sum(
        len(video.get("scenes", []))
        for video in videos
    )

    processed = 0

    print()
    print(f"Scenes to analyze: {total_scenes}")
    print()

    for video_index, video in enumerate(videos, start=1):

        video_name = video.get(
            "video",
            f"video_{video_index}",
        )

        raw_path = video.get("path")

        if not raw_path:

            print(
                f"⚠️ Skipping {video_name}: no path"
            )

            continue

        video_path = normalize_path(
            raw_path
        )

        if not video_path.exists():

            print()
            print(
                f"⚠️ Video not found:\n{video_path}"
            )

            continue

        scenes = video.get(
            "scenes",
            [],
        )

        print()
        print("-" * 72)
        print(f"🎬 {video_name}")
        print(f"Scenes: {len(scenes)}")
        print("-" * 72)

        for scene in scenes:

            processed += 1

            scene_id = scene.get(
                "scene_id",
                processed,
            )

            print(
                f"[{processed:02d}/{total_scenes}] "
                f"Scene {scene_id} ...",
                end=" ",
                flush=True,
            )

            try:

                result = analyze_scene(
                    model=model,
                    preprocess=preprocess,
                    text_embeddings=text_embeddings,
                    video_path=video_path,
                    scene=scene,
                )

                result["video"] = video_name
                result["video_path"] = str(video_path)

                all_results.append(
                    result
                )

                if result["analysis_status"] == "success":

                    semantic = result["semantic"]

                    print(
                        f"✅ "
                        f"{semantic['scene_role']} | "
                        f"{semantic['dominant_shot_type']} | "
                        f"{semantic['dominant_action']}"
                    )

                else:

                    print("⚠️ failed")

            except Exception as exc:

                print(
                    f"❌ {type(exc).__name__}: {exc}"
                )

                all_results.append(
                    {
                        "video": video_name,
                        "video_path": str(video_path),
                        "scene_id": scene_id,
                        "analysis_status": "error",
                        "error": str(exc),
                    }
                )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    successful = [
        item
        for item in all_results
        if item.get("analysis_status") == "success"
    ]

    failed = len(all_results) - len(successful)

    role_counts = {}

    shot_counts = {}

    for item in successful:

        role = item["semantic"]["scene_role"]

        shot_type = item["semantic"]["dominant_shot_type"]

        role_counts[role] = role_counts.get(
            role,
            0,
        ) + 1

        shot_counts[shot_type] = shot_counts.get(
            shot_type,
            0,
        ) + 1

    output = {
        "metadata": {
            "model": MODEL_NAME,
            "pretrained": PRETRAINED,
            "device": DEVICE,
            "total_scenes": total_scenes,
            "successful_scenes": len(successful),
            "failed_scenes": failed,
            "frames_per_scene": "1-3",
        },
        "summary": {
            "scene_roles": role_counts,
            "shot_types": shot_counts,
        },
        "scenes": all_results,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("✅ SEMANTIC SHOT ANALYSIS COMPLETE")
    print("=" * 72)

    print()
    print(f"Scenes analyzed : {total_scenes}")
    print(f"Successful      : {len(successful)}")
    print(f"Failed          : {failed}")

    print()
    print("Scene roles:")

    for key, value in sorted(
        role_counts.items(),
        key=lambda x: -x[1],
    ):

        print(
            f"  {key:12s}: {value}"
        )

    print()
    print("Shot types:")

    for key, value in sorted(
        shot_counts.items(),
        key=lambda x: -x[1],
    ):

        print(
            f"  {key:12s}: {value}"
        )

    print()
    print("Output:")
    print(OUTPUT_JSON)

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()