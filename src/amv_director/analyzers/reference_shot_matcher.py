from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import open_clip
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

REFERENCE_SHOT_MAP = (
    BASE_DIR / "data/outputs/reference_shot_map.json"
)

MUSIC_DIRECTOR_PLAN = (
    BASE_DIR / "data/outputs/music_director_plan.json"
)

SOURCE_SEMANTICS = (
    BASE_DIR
    / "data/outputs/clip_analysis/semantic_shot_analysis_v2.json"
)

OUTPUT_JSON = (
    BASE_DIR
    / "data/outputs/reference_conditioned_shot_plan.json"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED = "dfndr2b"

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# MATCHING WEIGHTS
# ============================================================

WEIGHTS = {
    "visual_similarity": 0.28,
    "music_role_fit": 0.20,
    "reference_framing_fit": 0.13,
    "visual_energy_fit": 0.13,
    "quality": 0.12,
    "semantic_confidence": 0.04,
    "duration_fit": 0.06,
    "continuity": 0.04,
}


# ============================================================
# HELPERS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:
        return default


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:

    return float(
        max(
            low,
            min(high, value),
        )
    )


def cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:

    value = torch.sum(a * b).item()

    return float(
        max(-1.0, min(1.0, value))
    )


def normalized_cosine(
    similarity: float,
) -> float:

    # Convert [-1, 1] into [0, 1].
    return clamp(
        (similarity + 1.0) / 2.0
    )


def load_json(path: Path) -> Any:

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# FLEXIBLE RECORD FINDERS
# ============================================================

def find_list_by_key(
    obj: Any,
    possible_keys: List[str],
) -> Optional[List[Dict]]:

    if isinstance(obj, dict):

        for key in possible_keys:

            value = obj.get(key)

            if (
                isinstance(value, list)
                and value
                and all(
                    isinstance(x, dict)
                    for x in value
                )
            ):

                return value

        for value in obj.values():

            result = find_list_by_key(
                value,
                possible_keys,
            )

            if result is not None:
                return result

    elif isinstance(obj, list):

        for value in obj:

            result = find_list_by_key(
                value,
                possible_keys,
            )

            if result is not None:
                return result

    return None


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================

def extract_frame(
    video_path: Path,
    timestamp: float,
) -> Optional[Image.Image]:

    cap = cv2.VideoCapture(
        str(video_path)
    )

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


# ============================================================
# MODEL
# ============================================================

def load_model():

    print("=" * 72)
    print("🎯 REFERENCE-CONDITIONED SHOT MATCHER")
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

    model.eval()

    print("✅ Model loaded")

    return model, preprocess


# ============================================================
# IMAGE EMBEDDING
# ============================================================

def encode_image(
    model,
    preprocess,
    image: Image.Image,
) -> torch.Tensor:

    tensor = preprocess(
        image
    ).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.inference_mode():

        feature = model.encode_image(
            tensor
        )

        feature = (
            feature
            / feature.norm(
                dim=-1,
                keepdim=True,
            )
        )

    return feature[0].detach()


# ============================================================
# SCENE CENTER
# ============================================================

def scene_center(
    start: float,
    end: float,
) -> float:

    return (
        start
        + (end - start) * 0.5
    )


# ============================================================
# REFERENCE FRAMING
# ============================================================

def framing_from_reference_duration(
    duration: float,
) -> Dict[str, float]:

    # This is intentionally a soft prior.
    #
    # Very short reference shots usually benefit from
    # tighter framing.
    #
    # Long reference shots can breathe with wider framing.

    if duration < 0.35:

        return {
            "close_up": 1.00,
            "medium_shot": 0.70,
            "wide_shot": 0.20,
            "extreme_wide": 0.10,
        }

    if duration < 0.70:

        return {
            "close_up": 0.90,
            "medium_shot": 1.00,
            "wide_shot": 0.45,
            "extreme_wide": 0.15,
        }

    if duration < 1.40:

        return {
            "close_up": 0.75,
            "medium_shot": 1.00,
            "wide_shot": 0.75,
            "extreme_wide": 0.35,
        }

    if duration < 2.50:

        return {
            "close_up": 0.55,
            "medium_shot": 0.95,
            "wide_shot": 1.00,
            "extreme_wide": 0.65,
        }

    return {
        "close_up": 0.45,
        "medium_shot": 0.80,
        "wide_shot": 1.00,
        "extreme_wide": 0.90,
    }


# ============================================================
# MUSIC ROLE FIT
# ============================================================

def music_role_fit(
    source_profile: Dict,
    role: str,
) -> float:

    role = role.lower()

    visual_energy = safe_float(
        source_profile.get(
            "visual_energy"
        )
    )

    combat = safe_float(
        source_profile.get(
            "combat_strength"
        )
    )

    movement = safe_float(
        source_profile.get(
            "movement_strength"
        )
    )

    face = safe_float(
        source_profile.get(
            "face_emphasis"
        )
    )

    # --------------------------------------------------------
    # Cinematic
    # --------------------------------------------------------

    if role == "cinematic_zone":

        score = (
            (1.0 - visual_energy) * 0.45
            + face * 0.20
            + (1.0 - combat) * 0.20
            + (1.0 - movement) * 0.15
        )

        return clamp(score)

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    if role == "build_zone":

        score = (
            visual_energy * 0.30
            + movement * 0.35
            + face * 0.15
            + combat * 0.20
        )

        return clamp(score)

    # --------------------------------------------------------
    # Impact
    # --------------------------------------------------------

    if role == "impact_zone":

        score = (
            visual_energy * 0.35
            + combat * 0.35
            + movement * 0.20
            + face * 0.10
        )

        return clamp(score)

    # --------------------------------------------------------
    # Accent
    # --------------------------------------------------------

    if role == "accent_zone":

        score = (
            face * 0.35
            + visual_energy * 0.30
            + movement * 0.20
            + combat * 0.15
        )

        return clamp(score)

    # --------------------------------------------------------
    # Action
    # --------------------------------------------------------

    if role == "action_zone":

        score = (
            combat * 0.45
            + movement * 0.35
            + visual_energy * 0.20
        )

        return clamp(score)

    return visual_energy


# ============================================================
# ENERGY FIT
# ============================================================

def target_energy_for_role(
    role: str,
) -> float:

    mapping = {
        "cinematic_zone": 0.25,
        "build_zone": 0.50,
        "impact_zone": 0.85,
        "accent_zone": 0.65,
        "action_zone": 0.80,
    }

    return mapping.get(
        role.lower(),
        0.50,
    )


def energy_fit(
    source_energy: float,
    target_energy: float,
) -> float:

    difference = abs(
        source_energy
        - target_energy
    )

    return clamp(
        1.0 - difference
    )


# ============================================================
# DURATION FIT
# ============================================================

def duration_fit(
    source_duration: float,
    target_duration: float,
) -> float:

    if target_duration <= 0:
        return 0.0

    ratio = (
        source_duration
        / target_duration
    )

    # Ideal when scene contains enough footage
    # for the requested window.

    if ratio >= 1.0:
        overflow = min(
            ratio - 1.0,
            2.0,
        )

        return clamp(
            1.0
            - overflow * 0.12
        )

    # Source shorter than desired segment.
    shortage = (
        1.0 - ratio
    )

    return clamp(
        1.0
        - shortage * 1.2
    )


# ============================================================
# SOURCE WINDOW
# ============================================================

def calculate_source_window(
    scene_start: float,
    scene_end: float,
    target_duration: float,
) -> Tuple[float, float]:

    scene_duration = (
        scene_end
        - scene_start
    )

    if scene_duration <= 0:
        return scene_start, scene_end

    target_duration = max(
        0.05,
        target_duration,
    )

    if target_duration >= scene_duration:

        return (
            scene_start,
            scene_end,
        )

    # Center window by default.
    #
    # Later this will become action-aware using
    # temporal motion peaks.

    center = (
        scene_start
        + scene_duration * 0.5
    )

    half = (
        target_duration * 0.5
    )

    window_start = (
        center - half
    )

    window_end = (
        center + half
    )

    if window_start < scene_start:

        shift = (
            scene_start
            - window_start
        )

        window_start += shift
        window_end += shift

    if window_end > scene_end:

        shift = (
            window_end
            - scene_end
        )

        window_start -= shift
        window_end -= shift

    window_start = max(
        scene_start,
        window_start,
    )

    window_end = min(
        scene_end,
        window_end,
    )

    return (
        window_start,
        window_end,
    )


# ============================================================
# FIND REFERENCE SHOT
# ============================================================

def reference_shot_for_time(
    reference_shots: List[Dict],
    timestamp: float,
) -> Optional[Dict]:

    candidates = []

    for shot in reference_shots:

        start = safe_float(
            shot.get("start")
        )

        end = safe_float(
            shot.get("end")
        )

        if start <= timestamp <= end:

            candidates.append(
                shot
            )

    if candidates:

        # Prefer the shortest containing shot.
        return min(
            candidates,
            key=lambda x: safe_float(
                x.get("duration"),
                safe_float(
                    x.get("end")
                )
                - safe_float(
                    x.get("start")
                ),
            ),
        )

    # Fallback: nearest midpoint.

    if not reference_shots:
        return None

    return min(
        reference_shots,
        key=lambda x: abs(
            scene_center(
                safe_float(
                    x.get("start")
                ),
                safe_float(
                    x.get("end")
                ),
            )
            - timestamp
        ),
    )


# ============================================================
# MUSIC ROLE
# ============================================================

# def get_role(
#     segment: Dict,
# ) -> str:

#     for key in (
#         "role",
#         "segment_role",
#         "zone",
#         "music_role",
#     ):

#         value = segment.get(key)

#         if isinstance(value, str):
#             return value

#     return "free"

def get_role(segment: Dict) -> str:
    for key in (
        "editorial_role",
        "role",
        "segment_role",
        "zone",
        "music_role",
    ):
        value = segment.get(key)

        if isinstance(value, str):
            return value

    return "free"
# ============================================================
# MATCH SCORE
# ============================================================

def candidate_score(
    source: Dict,
    reference_similarity: float,
    target_role: str,
    target_duration: float,
    reference_duration: float,
    previous_scene: Optional[Tuple[str, Any]],
) -> Tuple[float, Dict]:

    profile = source[
        "director_profile"
    ]

    quality = source[
        "quality"
    ]

    preferred_shot = profile.get(
        "preferred_shot",
        "uncertain",
    )

    shot_prior = framing_from_reference_duration(
        reference_duration
    )

    framing_score = shot_prior.get(
        preferred_shot,
        0.35,
    )

    if preferred_shot == "uncertain":
        framing_score *= 0.75

    role_score = music_role_fit(
        profile,
        target_role,
    )

    target_energy = (
        target_energy_for_role(
            target_role
        )
    )

    source_energy = safe_float(
        profile.get(
            "visual_energy"
        )
    )

    energy_score = energy_fit(
        source_energy,
        target_energy,
    )

    quality_flag = quality.get(
        "quality_flag",
        "usable",
    )

    quality_score = {
        "usable": 1.0,
        "weak": 0.55,
        "reject": 0.0,
    }.get(
        quality_flag,
        0.5,
    )

    semantic_confidence = clamp(
        safe_float(
            profile.get(
                "semantic_confidence"
            )
        )
    )

    source_duration = safe_float(
        source.get(
            "duration"
        )
    )

    dur_score = duration_fit(
        source_duration,
        target_duration,
    )

    continuity_score = 1.0

    video_name = source.get(
        "video"
    )

    scene_id = source.get(
        "scene_id"
    )

    current_id = (
        video_name,
        scene_id,
    )

    if (
        previous_scene is not None
        and current_id == previous_scene
    ):

        continuity_score = 0.0

    components = {

        "visual_similarity": (
            reference_similarity
        ),

        "music_role_fit": (
            role_score
        ),

        "reference_framing_fit": (
            framing_score
        ),

        "visual_energy_fit": (
            energy_score
        ),

        "quality": (
            quality_score
        ),

        "semantic_confidence": (
            semantic_confidence
        ),

        "duration_fit": (
            dur_score
        ),

        "continuity": (
            continuity_score
        ),
    }

    score = sum(
        WEIGHTS[name]
        * components[name]
        for name in WEIGHTS
    )

    return clamp(score), components


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    reference_data = load_json(
        REFERENCE_SHOT_MAP
    )

    music_data = load_json(
        MUSIC_DIRECTOR_PLAN
    )

    source_data = load_json(
        SOURCE_SEMANTICS
    )

    # --------------------------------------------------------
    # Locate records
    # --------------------------------------------------------

    reference_shots = (
        find_list_by_key(
            reference_data,
            [
                "shots",
                "editorial_shots",
                "shot_map",
            ],
        )
    )

    music_segments = (
        find_list_by_key(
            music_data,
            [
                "segments",
                "editorial_segments",
                "music_segments",
            ],
        )
    )

    source_scenes = source_data.get(
        "scenes",
        []
    )

    if not reference_shots:

        raise RuntimeError(
            "Could not find reference shots "
            "in reference_shot_map.json"
        )

    if not music_segments:

        raise RuntimeError(
            "Could not find music segments "
            "in music_director_plan.json"
        )

    if not source_scenes:

        raise RuntimeError(
            "No source scenes found in "
            "semantic_shot_analysis_v2.json"
        )

    print(
        f"Reference shots : "
        f"{len(reference_shots)}"
    )

    print(
        f"Music segments  : "
        f"{len(music_segments)}"
    )

    print(
        f"Source scenes   : "
        f"{len(source_scenes)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, preprocess = load_model()

    # --------------------------------------------------------
    # Reference embeddings
    # --------------------------------------------------------

    print()
    print(
        "Encoding reference shots..."
    )

    reference_embeddings = {}

    reference_video = (
        BASE_DIR
        / "data/references/reference_video.mp4"
    )

    for index, shot in enumerate(
        reference_shots,
        start=1,
    ):

        start = safe_float(
            shot.get("start")
        )

        end = safe_float(
            shot.get("end")
        )

        midpoint = (
            start
            + (end - start) * 0.5
        )

        image = extract_frame(
            reference_video,
            midpoint,
        )

        if image is None:

            print(
                f"  ⚠️ Reference shot "
                f"{index}: frame failed"
            )

            continue

        embedding = encode_image(
            model,
            preprocess,
            image,
        )

        reference_embeddings[
            index - 1
        ] = embedding

    print(
        f"✅ Encoded "
        f"{len(reference_embeddings)} "
        f"reference shots"
    )

    # --------------------------------------------------------
    # Source embeddings
    # --------------------------------------------------------

    print()
    print(
        "Encoding source scenes..."
    )

    source_embeddings = {}

    cache_key_counter = 0

    for index, scene in enumerate(
        source_scenes,
        start=1,
    ):

        if scene.get(
            "analysis_status"
        ) != "success":

            continue

        if scene.get(
            "quality",
            {}
        ).get(
            "quality_flag"
        ) == "reject":

            continue

        video_path_value = (
            scene.get(
                "video_path"
            )
        )

        if not video_path_value:
            continue

        video_path = Path(
            video_path_value
        )

        if not video_path.exists():
            continue

        start = safe_float(
            scene.get("start")
        )

        end = safe_float(
            scene.get("end")
        )

        timestamps = [
            start
            + (end - start) * 0.35,

            start
            + (end - start) * 0.65,
        ]

        embeddings = []

        for timestamp in timestamps:

            image = extract_frame(
                video_path,
                timestamp,
            )

            if image is None:
                continue

            embeddings.append(
                encode_image(
                    model,
                    preprocess,
                    image,
                )
            )

        if not embeddings:
            continue

        stacked = torch.stack(
            embeddings
        )

        mean_embedding = (
            stacked.mean(dim=0)
        )

        mean_embedding = (
            mean_embedding
            / mean_embedding.norm()
        )

        source_embeddings[index - 1] = (
            mean_embedding.detach()
        )

        cache_key_counter += 1

        print(
            f"  [{cache_key_counter:02d}] "
            f"{scene.get('video')} "
            f"scene {scene.get('scene_id')}"
        )

    print(
        f"✅ Encoded "
        f"{len(source_embeddings)} "
        f"source scenes"
    )

    # --------------------------------------------------------
    # Match
    # --------------------------------------------------------

    print()
    print(
        "=" * 72
    )
    print(
        "🎬 BUILDING CREATIVE SHOT PLAN"
    )
    print(
        "=" * 72
    )

    matches = []

    previous_scene = None

    used_windows = set()

    for target_index, segment in enumerate(
        music_segments,
        start=1,
    ):

        target_start = safe_float(
            segment.get("start")
        )

        target_end = safe_float(
            segment.get("end")
        )

        if target_end <= target_start:

            continue

        target_duration = (
            target_end
            - target_start
        )

        target_midpoint = (
            target_start
            + target_duration * 0.5
        )

        role = get_role(
            segment
        )

        # ----------------------------------------------------
        # Reference shot at same timeline position
        # ----------------------------------------------------

        ref_shot = reference_shot_for_time(
            reference_shots,
            target_midpoint,
        )

        if ref_shot is None:

            continue

        ref_index = reference_shots.index(
            ref_shot
        )

        reference_embedding = (
            reference_embeddings.get(
                ref_index
            )
        )

        reference_duration = safe_float(
            ref_shot.get(
                "duration",
                safe_float(
                    ref_shot.get("end")
                )
                - safe_float(
                    ref_shot.get("start")
                ),
            )
        )

        best = None

        for source_index, source in enumerate(
            source_scenes
        ):

            if source.get(
                "analysis_status"
            ) != "success":

                continue

            if source.get(
                "quality",
                {}
            ).get(
                "quality_flag"
            ) == "reject":

                continue

            embedding = (
                source_embeddings.get(
                    source_index
                )
            )

            if embedding is None:
                continue

            similarity = 0.5

            if reference_embedding is not None:

                cosine = cosine_similarity(
                    reference_embedding,
                    embedding,
                )

                similarity = (
                    normalized_cosine(
                        cosine
                    )
                )

            score, breakdown = (
                candidate_score(
                    source=source,
                    reference_similarity=similarity,
                    target_role=role,
                    target_duration=target_duration,
                    reference_duration=reference_duration,
                    previous_scene=previous_scene,
                )
            )

            video_name = source.get(
                "video"
            )

            scene_id = source.get(
                "scene_id"
            )

            window_start, window_end = (
                calculate_source_window(
                    safe_float(
                        source.get("start")
                    ),
                    safe_float(
                        source.get("end")
                    ),
                    target_duration,
                )
            )

            window_key = (
                video_name,
                scene_id,
                round(window_start, 3),
                round(window_end, 3),
            )

            # Exact repeated window is strongly discouraged.
            if window_key in used_windows:

                score -= 0.12

            candidate = {
                "score": score,
                "source": source,
                "source_index": source_index,
                "window_start": window_start,
                "window_end": window_end,
                "similarity": similarity,
                "breakdown": breakdown,
            }

            if (
                best is None
                or candidate["score"]
                > best["score"]
            ):

                best = candidate

        if best is None:

            continue

        source = best["source"]

        source_id = (
            source.get("video"),
            source.get("scene_id"),
        )

        used_windows.add(
            (
                source.get("video"),
                source.get("scene_id"),
                round(
                    best["window_start"],
                    3,
                ),
                round(
                    best["window_end"],
                    3,
                ),
            )
        )

        previous_scene = source_id

        result = {

            "target_index": target_index,

            "target": {
                "start": round(
                    target_start,
                    3,
                ),
                "end": round(
                    target_end,
                    3,
                ),
                "duration": round(
                    target_duration,
                    3,
                ),
                "music_role": role,
            },

            "reference": {
                "shot_index": ref_index + 1,
                "start": round(
                    safe_float(
                        ref_shot.get(
                            "start"
                        )
                    ),
                    3,
                ),
                "end": round(
                    safe_float(
                        ref_shot.get(
                            "end"
                        )
                    ),
                    3,
                ),
                "duration": round(
                    reference_duration,
                    3,
                ),
            },

            "source": {
                "video": source.get(
                    "video"
                ),
                "scene_id": source.get(
                    "scene_id"
                ),
                "scene_start": round(
                    safe_float(
                        source.get(
                            "start"
                        )
                    ),
                    3,
                ),
                "scene_end": round(
                    safe_float(
                        source.get(
                            "end"
                        )
                    ),
                    3,
                ),
                "window_start": round(
                    best["window_start"],
                    3,
                ),
                "window_end": round(
                    best["window_end"],
                    3,
                ),
            },

            "match_score": round(
                best["score"],
                4,
            ),

            "visual_similarity": round(
                best["similarity"],
                4,
            ),

            "score_breakdown": {
                key: round(
                    float(value),
                    4,
                )
                for key, value
                in best[
                    "breakdown"
                ].items()
            },

            "source_profile": {
                "scene_role": source[
                    "director_profile"
                ].get(
                    "scene_role"
                ),
                "preferred_shot": source[
                    "director_profile"
                ].get(
                    "preferred_shot"
                ),
                "visual_energy": source[
                    "director_profile"
                ].get(
                    "visual_energy"
                ),
                "semantic_confidence": source[
                    "director_profile"
                ].get(
                    "semantic_confidence"
                ),
                "quality": source[
                    "quality"
                ].get(
                    "quality_flag"
                ),
            },
        }

        matches.append(
            result
        )

        print(
            f"[{len(matches):02d}] "
            f"{role:15s} → "
            f"{source.get('video')} "
            f"S{source.get('scene_id')} | "
            f"score={best['score']:.3f} | "
            f"sim={best['similarity']:.3f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {

        "metadata": {

            "engine": (
                "reference_conditioned_shot_matcher_v1"
            ),

            "model": MODEL_NAME,

            "device": DEVICE,

            "reference_shots": len(
                reference_shots
            ),

            "music_segments": len(
                music_segments
            ),

            "source_scenes": len(
                source_scenes
            ),

            "matched_segments": len(
                matches
            ),

            "weights": WEIGHTS,
        },

        "matches": matches,
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

    print()
    print(
        "=" * 72
    )

    print(
        "✅ REFERENCE-CONDITIONED SHOT PLAN COMPLETE"
    )

    print()
    print(
        f"Matched segments : "
        f"{len(matches)}"
    )

    print(
        f"Output:\n{OUTPUT_JSON}"
    )

    print()
    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()