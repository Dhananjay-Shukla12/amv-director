from __future__ import annotations

import gc
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

AUDIO_SCAN_PATH = (
    BASE_DIR
    / "data/outputs/reference_clap_sfx_scan.json"
)

MUSIC_PLAN_PATH = (
    BASE_DIR
    / "data/outputs/music_director_plan.json"
)

REFERENCE_SHOT_MAP_PATH = (
    BASE_DIR
    / "data/outputs/reference_shot_map.json"
)

REFERENCE_VIDEO_PATH = (
    BASE_DIR
    / "data/references/reference_video.mp4"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data/outputs/reference_audio_visual_events_v3.json"
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
# SETTINGS
# ============================================================

FRAME_OFFSETS = [
    -0.25,
    0.0,
    0.25,
]

BATCH_SIZE = 8

SIGNIFICANT_THRESHOLD = 0.48
STRONG_THRESHOLD = 0.72

EVENT_MERGE_DISTANCE = 0.40


# ============================================================
# VISUAL CONCEPTS
#
# Each concept has positive and negative descriptions.
#
# This prevents concepts from competing in one global
# softmax and gives us an independent score for each concept.
# ============================================================

VISUAL_CONCEPTS = {

    "gun_weapon": {
        "positive": [
            "an anime character holding a gun",
            "an anime character holding a firearm",
            "a pistol or rifle in a character's hand",
            "an anime character aiming a gun",
        ],
        "negative": [
            "an anime scene with no gun or firearm",
            "an anime character with empty hands",
            "an anime scene without a weapon",
            "an anime character not holding a gun",
        ],
    },

    "gun_fire": {
        "positive": [
            "an anime character firing a gun",
            "a gun being fired in an anime scene",
            "a character shooting a firearm",
            "an anime muzzle flash from a gun",
        ],
        "negative": [
            "an anime character not firing a gun",
            "an anime scene without a gun being fired",
            "a character with no shooting action",
            "a calm anime character not attacking",
        ],
    },

    "sword_weapon": {
        "positive": [
            "an anime character holding a sword",
            "an anime character wielding a blade",
            "a sword visible in an anime scene",
            "two anime characters fighting with swords",
        ],
        "negative": [
            "an anime scene with no sword",
            "an anime character with no blade",
            "an anime character without a weapon",
            "a scene without sword combat",
        ],
    },

    "combat": {
        "positive": [
            "anime characters fighting",
            "an anime character attacking another character",
            "an anime combat scene",
            "a physical fight between anime characters",
        ],
        "negative": [
            "a calm anime character standing still",
            "a peaceful anime scene",
            "an anime environment with no combat",
            "an anime character not fighting",
        ],
    },

    "physical_hit": {
        "positive": [
            "an anime character punching another character",
            "an anime character being hit",
            "a powerful physical impact in an anime fight",
            "an anime character striking an opponent",
        ],
        "negative": [
            "an anime character standing peacefully",
            "a calm anime scene",
            "an anime environment without a fight",
            "an anime character not being hit",
        ],
    },

    "explosion": {
        "positive": [
            "a large explosion in an anime scene",
            "an anime fireball explosion",
            "a powerful blast in an anime scene",
            "an anime explosion with flames",
        ],
        "negative": [
            "an anime scene with no explosion",
            "a calm anime scene without a blast",
            "an anime character standing without an explosion",
            "an anime environment with no fireball",
        ],
    },

    "fast_motion": {
        "positive": [
            "an anime character moving extremely fast",
            "a rapidly moving anime character",
            "an anime character lunging forward",
            "a dynamic fast action pose",
        ],
        "negative": [
            "a completely still anime character",
            "a calm anime character standing still",
            "a static anime environment",
            "a slow peaceful anime scene",
        ],
    },

    "walking_running": {
        "positive": [
            "an anime character running",
            "an anime character walking",
            "an anime character moving on foot",
            "an anime character sprinting",
        ],
        "negative": [
            "an anime character standing still",
            "an anime close-up with no walking",
            "an anime character sitting still",
            "a static anime scene",
        ],
    },

    "scream": {
        "positive": [
            "an anime character screaming",
            "an anime character shouting loudly",
            "an anime character crying out",
            "an anime character with an open mouth scream",
        ],
        "negative": [
            "a silent anime character",
            "an anime character with a neutral expression",
            "a calm anime character",
            "an anime scene without a scream",
        ],
    },

    "glass_break": {
        "positive": [
            "glass shattering in an anime scene",
            "broken glass flying through the air",
            "an anime window breaking",
            "glass fragments exploding outward",
        ],
        "negative": [
            "an anime scene with no glass",
            "an anime scene without broken glass",
            "a character standing with no glass",
            "a normal intact environment",
        ],
    },

    "impact_visual": {
        "positive": [
            "a dramatic impact frame in an anime fight",
            "a powerful collision between characters",
            "an anime character being knocked back",
            "a strong hit in an anime scene",
        ],
        "negative": [
            "a calm anime scene",
            "a character standing without impact",
            "a peaceful environment",
            "an anime character not involved in a collision",
        ],
    },

    "face_closeup": {
        "positive": [
            "a close-up of an anime character's face",
            "an anime close-up focused on the eyes",
            "a dramatic anime facial close-up",
            "an intense anime eye close-up",
        ],
        "negative": [
            "an extreme wide anime environment shot",
            "an anime scene where the face is not prominent",
            "a distant full-scene composition",
            "an environment-dominated anime frame",
        ],
    },

    "environment": {
        "positive": [
            "an anime environment or landscape",
            "an anime establishing shot",
            "an anime scene dominated by surroundings",
            "a wide environmental anime shot",
        ],
        "negative": [
            "a tight anime character close-up",
            "an anime face close-up",
            "a frame dominated by a character's face",
            "a very tight character portrait",
        ],
    },
}


# ============================================================
# HELPERS
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

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
            min(
                high,
                value,
            ),
        )
    )


def load_json(
    path: Path,
) -> Any:

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
# REFERENCE LOOKUPS
# ============================================================

def get_reference_shots(
    data: Any,
) -> List[Dict]:

    if isinstance(data, dict):

        for key in (
            "shots",
            "editorial_shots",
            "shot_map",
        ):

            value = data.get(key)

            if isinstance(value, list):

                return value

    if isinstance(data, list):

        return data

    return []


def find_reference_shot(
    shots: List[Dict],
    timestamp: float,
) -> Optional[Dict]:

    matches = []

    for shot in shots:

        start = safe_float(
            shot.get("start")
        )

        end = safe_float(
            shot.get("end")
        )

        if start <= timestamp <= end:

            matches.append(
                shot
            )

    if matches:

        return min(
            matches,
            key=lambda x:
                safe_float(
                    x.get("duration"),
                    safe_float(
                        x.get("end")
                    )
                    - safe_float(
                        x.get("start")
                    ),
                ),
        )

    return None


def find_music_segment(
    segments: List[Dict],
    timestamp: float,
) -> Optional[Dict]:

    for segment in segments:

        start = safe_float(
            segment.get("start")
        )

        end = safe_float(
            segment.get("end")
        )

        if start <= timestamp <= end:

            return segment

    return None


# ============================================================
# VIDEO
# ============================================================

class VideoReader:

    def __init__(
        self,
        path: Path,
    ):

        self.cap = cv2.VideoCapture(
            str(path)
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Could not open:\n{path}"
            )

        self.fps = safe_float(
            self.cap.get(
                cv2.CAP_PROP_FPS
            ),
            30.0,
        )

        self.frame_count = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        self.duration = (
            self.frame_count
            / self.fps
            if self.fps > 0
            else 0.0
        )

    def get_frame(
        self,
        timestamp: float,
    ) -> Optional[Image.Image]:

        timestamp = max(
            0.0,
            min(
                timestamp,
                max(
                    0.0,
                    self.duration - 0.001,
                ),
            ),
        )

        self.cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000.0,
        )

        ok, frame = self.cap.read()

        if not ok or frame is None:

            return None

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        return Image.fromarray(
            frame
        )

    def close(self):

        self.cap.release()


# ============================================================
# MOTION
# ============================================================

def calculate_motion(
    images: List[Image.Image],
) -> float:

    if len(images) < 2:

        return 0.0

    differences = []

    for first, second in zip(
        images,
        images[1:],
    ):

        a = np.asarray(first)

        b = np.asarray(second)

        a = cv2.cvtColor(
            a,
            cv2.COLOR_RGB2GRAY,
        )

        b = cv2.cvtColor(
            b,
            cv2.COLOR_RGB2GRAY,
        )

        a = cv2.resize(
            a,
            (256, 144),
        )

        b = cv2.resize(
            b,
            (256, 144),
        )

        diff = np.mean(
            np.abs(
                a.astype(
                    np.float32
                )
                -
                b.astype(
                    np.float32
                )
            )
        )

        differences.append(
            clamp(
                diff / 35.0
            )
        )

    return float(
        np.mean(
            differences
        )
    )


# ============================================================
# MOBILECLIP
# ============================================================

def load_model():

    print()
    print(
        "Loading MobileCLIP2..."
    )

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

    print(
        "✅ MobileCLIP2 loaded"
    )

    return (
        model,
        preprocess,
        tokenizer,
    )


# ============================================================
# BUILD POSITIVE / NEGATIVE EMBEDDINGS
# ============================================================

def build_concept_embeddings(
    model,
    tokenizer,
):

    concept_embeddings = {}

    print()
    print(
        "Building calibrated visual concepts..."
    )

    with torch.inference_mode():

        for concept, definitions in (
            VISUAL_CONCEPTS.items()
        ):

            positive_tokens = tokenizer(
                definitions["positive"]
            )

            negative_tokens = tokenizer(
                definitions["negative"]
            )

            positive_tokens = (
                positive_tokens.to(
                    DEVICE
                )
            )

            negative_tokens = (
                negative_tokens.to(
                    DEVICE
                )
            )

            positive_features = (
                model.encode_text(
                    positive_tokens
                )
            )

            negative_features = (
                model.encode_text(
                    negative_tokens
                )
            )

            positive_features = (
                positive_features
                /
                positive_features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            negative_features = (
                negative_features
                /
                negative_features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            positive_mean = (
                positive_features.mean(
                    dim=0
                )
            )

            positive_mean = (
                positive_mean
                /
                positive_mean.norm()
            )

            negative_mean = (
                negative_features.mean(
                    dim=0
                )
            )

            negative_mean = (
                negative_mean
                /
                negative_mean.norm()
            )

            concept_embeddings[
                concept
            ] = {
                "positive":
                    positive_mean.detach(),

                "negative":
                    negative_mean.detach(),
            }

    print(
        f"✅ {len(concept_embeddings)} "
        f"concepts ready"
    )

    return concept_embeddings


# ============================================================
# IMAGE EMBEDDINGS
# ============================================================

def encode_images(
    model,
    preprocess,
    images: List[Image.Image],
) -> List[torch.Tensor]:

    if not images:

        return []

    tensors = [
        preprocess(image)
        for image in images
    ]

    embeddings = []

    for start in range(
        0,
        len(tensors),
        BATCH_SIZE,
    ):

        batch = torch.stack(
            tensors[
                start:
                start + BATCH_SIZE
            ]
        )

        batch = batch.to(
            DEVICE
        )

        with torch.inference_mode():

            features = (
                model.encode_image(
                    batch
                )
            )

            features = (
                features
                /
                features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

        embeddings.extend(
            [
                feature.detach()
                for feature in features
            ]
        )

        del batch
        del features

    return embeddings


# ============================================================
# CALIBRATED CONCEPT SCORE
# ============================================================

def score_concept(
    image_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    negative_embedding: torch.Tensor,
) -> float:

    positive_similarity = torch.sum(
        image_embedding
        * positive_embedding
    )

    negative_similarity = torch.sum(
        image_embedding
        * negative_embedding
    )

    logits = torch.stack(
        [
            positive_similarity,
            negative_similarity,
        ]
    )

    probability = torch.softmax(
        logits * 10.0,
        dim=0,
    )[0]

    return float(
        probability.item()
    )


def visual_scores(
    image_embeddings: List[torch.Tensor],
    concept_embeddings: Dict,
) -> Dict[str, float]:

    if not image_embeddings:

        return {
            concept: 0.0
            for concept
            in concept_embeddings
        }

    results = {}

    for concept, embeddings in (
        concept_embeddings.items()
    ):

        frame_scores = []

        for image_embedding in (
            image_embeddings
        ):

            score = score_concept(
                image_embedding,
                embeddings["positive"],
                embeddings["negative"],
            )

            frame_scores.append(
                score
            )

        # For an event, the strongest nearby frame is
        # often more useful than the average.
        #
        # Example: a muzzle flash may only exist in
        # the center frame.
        strongest = max(
            frame_scores
        )

        average = float(
            np.mean(
                frame_scores
            )
        )

        results[concept] = clamp(
            strongest * 0.75
            + average * 0.25
        )

    return results


# ============================================================
# AUDIO
# ============================================================

def get_audio_scores(
    analysis: Dict,
) -> Dict[str, float]:

    return {
        key: safe_float(value)
        for key, value
        in analysis.get(
            "group_scores",
            {},
        ).items()
    }


# ============================================================
# JOINT EVENT RULES
# ============================================================

def joint_events(
    audio: Dict[str, float],
    visual: Dict[str, float],
    motion: float,
) -> Dict[str, float]:

    gun_audio = max(
        audio.get(
            "gunshot",
            0.0,
        ),
        audio.get(
            "pistol_firing",
            0.0,
        ),
    )

    impact_audio = audio.get(
        "impact",
        0.0,
    )

    sword_audio = max(
        audio.get(
            "sword",
            0.0,
        ),
        audio.get(
            "weapon_clash",
            0.0,
        ),
    )

    explosion_audio = audio.get(
        "explosion",
        0.0,
    )

    whoosh_audio = audio.get(
        "whoosh",
        0.0,
    )

    footsteps_audio = audio.get(
        "footsteps",
        0.0,
    )

    scream_audio = audio.get(
        "scream",
        0.0,
    )

    glass_audio = audio.get(
        "glass",
        0.0,
    )

    gun_visual = visual.get(
        "gun_weapon",
        0.0,
    )

    gun_fire_visual = visual.get(
        "gun_fire",
        0.0,
    )

    sword_visual = visual.get(
        "sword_weapon",
        0.0,
    )

    combat_visual = visual.get(
        "combat",
        0.0,
    )

    hit_visual = visual.get(
        "physical_hit",
        0.0,
    )

    impact_visual = visual.get(
        "impact_visual",
        0.0,
    )

    explosion_visual = visual.get(
        "explosion",
        0.0,
    )

    fast_visual = visual.get(
        "fast_motion",
        0.0,
    )

    walking_visual = visual.get(
        "walking_running",
        0.0,
    )

    scream_visual = visual.get(
        "scream",
        0.0,
    )

    glass_visual = visual.get(
        "glass_break",
        0.0,
    )

    # ========================================================
    # WEAPON FIRE
    #
    # Two possible routes:
    #
    # 1. Actual gunshot audio + weapon visual.
    # 2. Impact-like audio + strong firearm/fire visual.
    #
    # We keep the threshold conservative so "gun visible"
    # alone does not become WEAPON_FIRE.
    # ========================================================

    weapon_fire_audio_visual = (
        gun_audio * 0.45
        + gun_visual * 0.25
        + gun_fire_visual * 0.30
    )

    weapon_fire_inferred = (
        impact_audio * 0.25
        + gun_visual * 0.35
        + gun_fire_visual * 0.40
    )

    weapon_fire = max(
        weapon_fire_audio_visual,
        weapon_fire_inferred,
    )

    # ========================================================
    # SWORD
    # ========================================================

    sword_impact = (
        sword_audio * 0.45
        + sword_visual * 0.35
        + min(
            sword_audio,
            sword_visual,
        ) * 0.20
    )

    # ========================================================
    # PHYSICAL IMPACT
    # ========================================================

    physical_impact = (
        impact_audio * 0.45
        + hit_visual * 0.30
        + impact_visual * 0.15
        + combat_visual * 0.10
    )

    # ========================================================
    # EXPLOSION
    # ========================================================

    explosion = (
        explosion_audio * 0.55
        + explosion_visual * 0.45
    )

    # ========================================================
    # WHOOSH
    # ========================================================

    whoosh = (
        whoosh_audio * 0.50
        + fast_visual * 0.30
        + motion * 0.20
    )

    # ========================================================
    # FOOTSTEPS
    # ========================================================

    footsteps = (
        footsteps_audio * 0.65
        + walking_visual * 0.35
    )

    # ========================================================
    # SCREAM
    # ========================================================

    scream = (
        scream_audio * 0.60
        + scream_visual * 0.40
    )

    # ========================================================
    # GLASS
    # ========================================================

    glass = (
        glass_audio * 0.55
        + glass_visual * 0.45
    )

    return {

        "WEAPON_FIRE":
            clamp(
                weapon_fire
            ),

        "SWORD_IMPACT":
            clamp(
                sword_impact
            ),

        "PHYSICAL_IMPACT":
            clamp(
                physical_impact
            ),

        "EXPLOSION":
            clamp(
                explosion
            ),

        "WHOOSH_ACTION":
            clamp(
                whoosh
            ),

        "FOOTSTEP_ACTION":
            clamp(
                footsteps
            ),

        "SCREAM":
            clamp(
                scream
            ),

        "GLASS_BREAK":
            clamp(
                glass
            ),
    }


# ============================================================
# EVENT SELECTION
# ============================================================

def select_event(
    joint: Dict[str, float],
    audio: Dict[str, float],
    visual: Dict[str, float],
) -> tuple[str, float, float]:

    ranked = sorted(
        joint.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    if not ranked:

        return (
            "NONE",
            0.0,
            0.0,
        )

    best_name = ranked[0][0]
    best_score = ranked[0][1]

    second_score = (
        ranked[1][1]
        if len(ranked) > 1
        else 0.0
    )

    margin = (
        best_score
        - second_score
    )

    # --------------------------------------------------------
    # Weapon-fire gate
    #
    # A gun being visible should not by itself trigger
    # a gunshot event.
    # --------------------------------------------------------

    if best_name == "WEAPON_FIRE":

        gun_visual = visual.get(
            "gun_weapon",
            0.0,
        )

        gun_fire_visual = visual.get(
            "gun_fire",
            0.0,
        )

        gun_audio = max(
            audio.get(
                "gunshot",
                0.0,
            ),
            audio.get(
                "pistol_firing",
                0.0,
            ),
        )

        impact_audio = audio.get(
            "impact",
            0.0,
        )

        # Strong visual firing evidence.
        visual_fire_route = (
            gun_visual >= 0.62
            and gun_fire_visual >= 0.60
        )

        # Acoustic route.
        audio_fire_route = (
            gun_visual >= 0.58
            and gun_audio >= 0.08
        )

        # Inferred route when a gun is clearly present and
        # there is a substantial impact transient.
        inferred_route = (
            gun_visual >= 0.68
            and impact_audio >= 0.28
            and gun_fire_visual >= 0.48
        )

        if not (
            visual_fire_route
            or audio_fire_route
            or inferred_route
        ):

            # Do not hallucinate gunfire.
            if (
                visual.get(
                    "physical_hit",
                    0.0,
                )
                > visual.get(
                    "gun_weapon",
                    0.0,
                )
            ):

                return (
                    "PHYSICAL_IMPACT",
                    joint.get(
                        "PHYSICAL_IMPACT",
                        0.0,
                    ),
                    margin,
                )

            return (
                "NONE",
                best_score,
                margin,
            )

    if best_score < SIGNIFICANT_THRESHOLD:

        return (
            "NONE",
            best_score,
            margin,
        )

    return (
        best_name,
        best_score,
        margin,
    )


def confidence_band(
    score: float,
    margin: float,
) -> str:

    if (
        score >= STRONG_THRESHOLD
        and margin >= 0.08
    ):

        return "strong"

    if (
        score >= SIGNIFICANT_THRESHOLD
        and margin >= 0.04
    ):

        return "medium"

    if score >= 0.32:

        return "weak"

    return "none"


# ============================================================
# MERGE
# ============================================================

def merge_events(
    events: List[Dict],
) -> List[Dict]:

    candidates = [
        item
        for item in events
        if item[
            "event_name"
        ]
        != "NONE"
        and item[
            "confidence"
        ]
        >= SIGNIFICANT_THRESHOLD
    ]

    candidates.sort(
        key=lambda x: (
            x["event_name"],
            x["time"],
        )
    )

    merged = []

    for candidate in candidates:

        if not merged:

            merged.append(
                candidate
            )

            continue

        previous = merged[-1]

        same_type = (
            candidate[
                "event_name"
            ]
            ==
            previous[
                "event_name"
            ]
        )

        close = (
            candidate[
                "time"
            ]
            -
            previous[
                "time"
            ]
            <
            EVENT_MERGE_DISTANCE
        )

        if same_type and close:

            if (
                candidate[
                    "confidence"
                ]
                >
                previous[
                    "confidence"
                ]
            ):

                merged[-1] = candidate

        else:

            merged.append(
                candidate
            )

    merged.sort(
        key=lambda x: x["time"]
    )

    return merged


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "🎬 AUDIO-VISUAL EVENT ALIGNER V3"
    )
    print("=" * 72)

    audio_data = load_json(
        AUDIO_SCAN_PATH
    )

    music_data = load_json(
        MUSIC_PLAN_PATH
    )

    reference_data = load_json(
        REFERENCE_SHOT_MAP_PATH
    )

    audio_windows = audio_data.get(
        "window_analyses",
        [],
    )

    music_segments = music_data.get(
        "segments",
        [],
    )

    reference_shots = (
        get_reference_shots(
            reference_data
        )
    )

    print()
    print(
        f"Audio windows   : "
        f"{len(audio_windows)}"
    )

    print(
        f"Music segments  : "
        f"{len(music_segments)}"
    )

    print(
        f"Reference shots : "
        f"{len(reference_shots)}"
    )

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    reader = VideoReader(
        REFERENCE_VIDEO_PATH
    )

    print()
    print(
        f"Reference video : "
        f"{reader.duration:.3f}s"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, preprocess, tokenizer = (
        load_model()
    )

    concept_embeddings = (
        build_concept_embeddings(
            model,
            tokenizer,
        )
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    results = []

    print()
    print(
        "Analyzing calibrated audiovisual events..."
    )

    for index, analysis in enumerate(
        audio_windows,
        start=1,
    ):

        timestamp = safe_float(
            analysis.get(
                "timestamp"
            )
        )

        print(
            f"[{index:03d}/{len(audio_windows):03d}] "
            f"{timestamp:7.3f}s ... ",
            end="",
            flush=True,
        )

        # ----------------------------------------------------
        # Temporal visual frames
        # ----------------------------------------------------

        images = []

        for offset in FRAME_OFFSETS:

            image = reader.get_frame(
                timestamp + offset
            )

            if image is not None:

                images.append(
                    image
                )

        motion = calculate_motion(
            images
        )

        # ----------------------------------------------------
        # Visual embeddings
        # ----------------------------------------------------

        image_embeddings = encode_images(
            model,
            preprocess,
            images,
        )

        visual = visual_scores(
            image_embeddings,
            concept_embeddings,
        )

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        audio = get_audio_scores(
            analysis
        )

        # ----------------------------------------------------
        # Joint events
        # ----------------------------------------------------

        joint = joint_events(
            audio,
            visual,
            motion,
        )

        event_name, confidence, margin = (
            select_event(
                joint,
                audio,
                visual,
            )
        )

        band = confidence_band(
            confidence,
            margin,
        )

        reference_shot = (
            find_reference_shot(
                reference_shots,
                timestamp,
            )
        )

        music_segment = (
            find_music_segment(
                music_segments,
                timestamp,
            )
        )

        top_audio = (
            max(
                audio.items(),
                key=lambda x: x[1],
            )
            if audio
            else (
                "none",
                0.0,
            )
        )

        top_visual = (
            max(
                visual.items(),
                key=lambda x: x[1],
            )
            if visual
            else (
                "none",
                0.0,
            )
        )

        result = {

            "time":
                round(
                    timestamp,
                    4,
                ),

            "event_name":
                event_name,

            "confidence":
                round(
                    confidence,
                    4,
                ),

            "confidence_margin":
                round(
                    margin,
                    4,
                ),

            "confidence_band":
                band,

            "audio": {

                "top":
                    top_audio[0],

                "top_score":
                    round(
                        top_audio[1],
                        4,
                    ),

                "scores": {
                    key: round(
                        value,
                        4,
                    )
                    for key, value
                    in audio.items()
                },
            },

            "visual": {

                "top":
                    top_visual[0],

                "top_score":
                    round(
                        top_visual[1],
                        4,
                    ),

                "scores": {
                    key: round(
                        value,
                        4,
                    )
                    for key, value
                    in sorted(
                        visual.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                },
            },

            "joint_scores": {
                key: round(
                    value,
                    4,
                )
                for key, value
                in sorted(
                    joint.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            },

            "motion":
                round(
                    motion,
                    4,
                ),

            "music": {

                "editorial_role":
                    (
                        music_segment.get(
                            "editorial_role"
                        )
                        if music_segment
                        else None
                    ),

                "sync_type":
                    (
                        music_segment.get(
                            "sync_type"
                        )
                        if music_segment
                        else None
                    ),

                "sync_priority":
                    (
                        music_segment.get(
                            "sync_priority"
                        )
                        if music_segment
                        else None
                    ),
            },

            "reference": {

                "shot_index":
                    (
                        reference_shots.index(
                            reference_shot
                        ) + 1
                        if reference_shot
                        else None
                    ),

                "start":
                    (
                        round(
                            safe_float(
                                reference_shot.get(
                                    "start"
                                )
                            ),
                            4,
                        )
                        if reference_shot
                        else None
                    ),

                "end":
                    (
                        round(
                            safe_float(
                                reference_shot.get(
                                    "end"
                                )
                            ),
                            4,
                        )
                        if reference_shot
                        else None
                    ),
            },
        }

        results.append(
            result
        )

        print(
            f"{event_name:18s} "
            f"{confidence:.3f}"
        )

    # --------------------------------------------------------
    # Significant events
    # --------------------------------------------------------

    significant = merge_events(
        results
    )

    event_counts = Counter(
        item["event_name"]
        for item in significant
    )

    strong_events = [
        item
        for item in significant
        if item[
            "confidence_band"
        ] == "strong"
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {

        "metadata": {

            "engine":
                "audio_visual_event_aligner_v3",

            "model":
                MODEL_NAME,

            "pretrained":
                PRETRAINED,

            "device":
                DEVICE,

            "audio_windows":
                len(audio_windows),

            "significant_events":
                len(significant),

            "concepts":
                list(
                    VISUAL_CONCEPTS.keys()
                ),
        },

        "summary": {

            "event_counts":
                dict(
                    event_counts
                ),

            "strong_event_counts":
                dict(
                    Counter(
                        item[
                            "event_name"
                        ]
                        for item
                        in strong_events
                    )
                ),
        },

        "significant_events":
            significant,

        "all_windows":
            results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    reader.close()

    del model
    del preprocess
    del tokenizer
    del concept_embeddings

    gc.collect()

    if DEVICE == "mps":

        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "✅ AUDIO-VISUAL EVENT ALIGNER V3 COMPLETE"
    )
    print("=" * 72)

    print()

    print(
        f"Windows analyzed   : "
        f"{len(results)}"
    )

    print(
        f"Significant events : "
        f"{len(significant)}"
    )

    print()

    print(
        "Event types:"
    )

    for event, count in (
        event_counts.most_common()
    ):

        print(
            f"  {event:20s}: "
            f"{count}"
        )

    print()

    print(
        "Strong events:"
    )

    for event in strong_events[:30]:

        print(
            f"  "
            f"{event['time']:7.3f}s | "
            f"{event['event_name']:20s} | "
            f"{event['confidence']:.3f} | "
            f"audio={event['audio']['top']} | "
            f"visual={event['visual']['top']}"
        )

    print()

    print(
        "Output:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()