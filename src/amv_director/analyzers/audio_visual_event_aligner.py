from __future__ import annotations

import gc
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

REFERENCE_VIDEO_PATH = (
    BASE_DIR
    / "data/references/reference_video.mp4"
)

MUSIC_PLAN_PATH = (
    BASE_DIR
    / "data/outputs/music_director_plan.json"
)

REFERENCE_SHOT_MAP_PATH = (
    BASE_DIR
    / "data/outputs/reference_shot_map.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data/outputs/reference_audio_visual_events.json"
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
# PROCESSING
# ============================================================

# Three frames give us a little temporal context:
#
#          event
#     -0.22   0   +0.22
#
# This is still cheap enough for your machine.
FRAME_OFFSETS = [
    -0.22,
    0.0,
    0.22,
]

BATCH_SIZE = 8

# A significant audiovisual event must reach this score.
SIGNIFICANT_THRESHOLD = 0.42

# Strong event threshold.
STRONG_THRESHOLD = 0.62

# Minimum separation between repeated detections of the same
# event type when building the compact event list.
EVENT_MERGE_DISTANCE = 0.35


# ============================================================
# SPECIALIZED VISUAL VOCABULARY
# ============================================================

VISUAL_PROMPTS = {

    "gun_weapon": [
        "a character holding a gun",
        "a firearm in a character's hand",
        "a pistol or rifle visible in an anime frame",
        "a character aiming a gun",
        "a character with a firearm",
    ],

    "gun_fire_action": [
        "a character firing a gun",
        "a character shooting a firearm",
        "a gun being fired",
        "a muzzle flash from a gun",
        "a character pulling the trigger",
    ],

    "sword_weapon": [
        "a character holding a sword",
        "a sword or blade visible",
        "a character wielding a blade",
        "two characters with swords",
        "a sword fight",
    ],

    "combat": [
        "anime characters fighting",
        "an anime character attacking another character",
        "an anime combat scene",
        "a character striking another character",
        "two characters fighting",
    ],

    "physical_hit": [
        "a character punching another character",
        "a character being hit",
        "a physical collision between characters",
        "a powerful punch",
        "a character striking an opponent",
    ],

    "explosion_visual": [
        "an explosion in an anime scene",
        "a large fireball",
        "a blast or explosion",
        "an explosion behind a character",
        "a powerful energy explosion",
    ],

    "fast_motion": [
        "a character moving extremely fast",
        "a fast anime action movement",
        "a character lunging forward",
        "a character rapidly moving through the frame",
        "a dynamic action pose",
    ],

    "walking_running": [
        "an anime character walking",
        "an anime character running",
        "a character moving on foot",
        "a character's feet while walking",
        "a character running forward",
    ],

    "scream_visual": [
        "an anime character screaming",
        "a character shouting",
        "a character crying out",
        "an open mouth scream",
        "an intense shouting expression",
    ],

    "glass_visual": [
        "glass shattering in an anime scene",
        "broken glass",
        "glass fragments flying",
        "a window breaking",
        "something made of glass shattering",
    ],

    "impact_visual": [
        "a powerful impact in an anime fight",
        "a character colliding with something",
        "a dramatic hit",
        "an impact frame",
        "a character being knocked back",
    ],

    "face_visual": [
        "a close-up of an anime character's face",
        "an anime character looking intensely",
        "a close-up of anime eyes",
        "an intense facial expression",
        "a dramatic anime close-up",
    ],
}


# ============================================================
# EVENT RULES
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


def softmax(
    values: List[float],
    temperature: float = 0.07,
) -> List[float]:

    arr = np.asarray(
        values,
        dtype=np.float32,
    )

    if len(arr) == 0:
        return []

    temperature = max(
        temperature,
        0.001,
    )

    arr = arr / temperature
    arr -= np.max(arr)

    exp_values = np.exp(
        arr
    )

    total = float(
        np.sum(exp_values)
    )

    if total <= 0:
        return [
            0.0
            for _ in values
        ]

    return (
        exp_values
        / total
    ).tolist()


# ============================================================
# REFERENCE SHOT LOOKUP
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

            if isinstance(
                value,
                list,
            ):

                return value

    if isinstance(data, list):

        if all(
            isinstance(x, dict)
            for x in data
        ):

            return data

    return []


def find_reference_shot(
    shots: List[Dict],
    timestamp: float,
) -> Optional[Dict]:

    containing = []

    for shot in shots:

        start = safe_float(
            shot.get("start")
        )

        end = safe_float(
            shot.get("end")
        )

        if (
            start <= timestamp
            <= end
        ):

            containing.append(
                shot
            )

    if containing:

        return min(
            containing,
            key=lambda shot:
                safe_float(
                    shot.get("duration"),
                    safe_float(
                        shot.get("end")
                    )
                    - safe_float(
                        shot.get("start")
                    ),
                ),
        )

    return None


# ============================================================
# MUSIC SEGMENT LOOKUP
# ============================================================

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

        if (
            start <= timestamp
            <= end
        ):

            return segment

    return None


# ============================================================
# VIDEO READER
# ============================================================

class VideoReader:

    def __init__(
        self,
        video_path: Path,
    ):

        self.cap = cv2.VideoCapture(
            str(video_path)
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Could not open:\n"
                f"{video_path}"
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

        timestamp = clamp(
            timestamp,
            0.0,
            max(
                0.0,
                self.duration - 0.001,
            ),
        )

        self.cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000.0,
        )

        ok, frame = (
            self.cap.read()
        )

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
# LOCAL MOTION
# ============================================================

def frame_difference(
    first: Optional[Image.Image],
    second: Optional[Image.Image],
) -> float:

    if first is None or second is None:
        return 0.0

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
            - b.astype(
                np.float32
            )
        )
    )

    return clamp(
        diff / 35.0
    )


# ============================================================
# MODEL
# ============================================================

def load_mobileclip():

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

    tokenizer = (
        open_clip.get_tokenizer(
            MODEL_NAME
        )
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
# TEXT EMBEDDINGS
# ============================================================

def build_text_embeddings(
    model,
    tokenizer,
):

    print()
    print(
        "Building visual concept embeddings..."
    )

    group_embeddings = {}

    with torch.inference_mode():

        for group, prompts in (
            VISUAL_PROMPTS.items()
        ):

            tokens = tokenizer(
                prompts
            )

            tokens = tokens.to(
                DEVICE
            )

            features = (
                model.encode_text(
                    tokens
                )
            )

            features = (
                features
                / features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            # Store each prompt separately.
            group_embeddings[
                group
            ] = (
                features.detach()
            )

    print(
        "✅ Visual concept embeddings ready"
    )

    return group_embeddings


# ============================================================
# IMAGE EMBEDDING
# ============================================================

def encode_images(
    model,
    preprocess,
    images: List[Image.Image],
) -> List[torch.Tensor]:

    tensors = []

    for image in images:

        tensor = preprocess(
            image
        )

        tensors.append(
            tensor
        )

    if not tensors:

        return []

    features_out = []

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
                / features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

        features_out.extend(
            [
                item.detach()
                for item in features
            ]
        )

        del batch
        del features

    return features_out


# ============================================================
# VISUAL GROUP SCORES
# ============================================================

def classify_visual_concepts(
    image_embeddings: List[torch.Tensor],
    group_embeddings: Dict[str, torch.Tensor],
) -> Dict[str, float]:

    if not image_embeddings:

        return {
            group: 0.0
            for group in group_embeddings
        }

    group_raw_scores = {}

    stacked_images = torch.stack(
        image_embeddings
    )

    for group, text_features in (
        group_embeddings.items()
    ):

        # image × prompts
        similarities = (
            stacked_images
            @ text_features.T
        )

        # Average the best two prompts.
        prompt_scores = (
            similarities.mean(
                dim=0
            )
        )

        values = prompt_scores.tolist()

        values.sort(
            reverse=True
        )

        if len(values) >= 2:

            raw = (
                values[0] * 0.65
                + values[1] * 0.35
            )

        else:

            raw = values[0]

        group_raw_scores[
            group
        ] = float(raw)

    # Convert visual similarities into relative scores.
    names = list(
        group_raw_scores.keys()
    )

    raw_values = [
        group_raw_scores[name]
        for name in names
    ]

    probabilities = softmax(
        raw_values,
        temperature=0.035,
    )

    return {
        names[i]: float(
            probabilities[i]
        )
        for i in range(
            len(names)
        )
    }


# ============================================================
# AUDIO SCORES
# ============================================================

def extract_audio_scores(
    analysis: Dict,
) -> Dict[str, float]:

    scores = analysis.get(
        "group_scores",
        {},
    )

    return {
        str(key): safe_float(
            value
        )
        for key, value
        in scores.items()
    }


# ============================================================
# JOINT EVENT SCORING
# ============================================================

def joint_score(
    audio: Dict[str, float],
    visual: Dict[str, float],
    local_motion: float,
) -> Dict[str, float]:

    # --------------------------------------------------------
    # AUDIO COMPONENTS
    # --------------------------------------------------------

    gunshot = audio.get(
        "gunshot",
        0.0,
    )

    explosion = audio.get(
        "explosion",
        0.0,
    )

    sword = audio.get(
        "sword",
        0.0,
    )

    impact = audio.get(
        "impact",
        0.0,
    )

    whoosh = audio.get(
        "whoosh",
        0.0,
    )

    footsteps = audio.get(
        "footsteps",
        0.0,
    )

    scream = audio.get(
        "scream",
        0.0,
    )

    glass = audio.get(
        "glass",
        0.0,
    )

    # --------------------------------------------------------
    # VISUAL COMPONENTS
    # --------------------------------------------------------

    gun_weapon = visual.get(
        "gun_weapon",
        0.0,
    )

    gun_fire = visual.get(
        "gun_fire_action",
        0.0,
    )

    sword_weapon = visual.get(
        "sword_weapon",
        0.0,
    )

    combat = visual.get(
        "combat",
        0.0,
    )

    physical_hit = visual.get(
        "physical_hit",
        0.0,
    )

    explosion_visual = visual.get(
        "explosion_visual",
        0.0,
    )

    fast_motion = visual.get(
        "fast_motion",
        0.0,
    )

    walking = visual.get(
        "walking_running",
        0.0,
    )

    scream_visual = visual.get(
        "scream_visual",
        0.0,
    )

    glass_visual = visual.get(
        "glass_visual",
        0.0,
    )

    impact_visual = visual.get(
        "impact_visual",
        0.0,
    )

    # --------------------------------------------------------
    # JOINT EVENTS
    # --------------------------------------------------------

    scores = {

        # Gunshot can be weak acoustically, so an impact-like
        # sound plus strong weapon/firing imagery can still
        # produce a weapon-fire hypothesis.
        "WEAPON_FIRE": clamp(
            gunshot * 0.45
            + gun_weapon * 0.22
            + gun_fire * 0.28
            + min(
                impact,
                gun_weapon,
            ) * 0.30
        ),

        "SWORD_IMPACT": clamp(
            sword * 0.55
            + sword_weapon * 0.35
            + min(
                impact,
                sword_weapon,
            ) * 0.20
        ),

        "PHYSICAL_IMPACT": clamp(
            impact * 0.58
            + physical_hit * 0.25
            + combat * 0.12
            + impact_visual * 0.15
        ),

        "EXPLOSION": clamp(
            explosion * 0.68
            + explosion_visual * 0.32
        ),

        "WHOOSH_ACTION": clamp(
            whoosh * 0.50
            + fast_motion * 0.32
            + local_motion * 0.18
        ),

        "FOOTSTEP_ACTION": clamp(
            footsteps * 0.65
            + walking * 0.35
        ),

        "SCREAM": clamp(
            scream * 0.65
            + scream_visual * 0.35
        ),

        "GLASS_BREAK": clamp(
            glass * 0.65
            + glass_visual * 0.35
        ),
    }

    return scores


# ============================================================
# CHOOSE EVENT
# ============================================================

def choose_joint_event(
    scores: Dict[str, float],
    audio_scores: Dict[str, float],
    visual_scores: Dict[str, float],
) -> Tuple[str, float, float]:

    if not scores:

        return (
            "NONE",
            0.0,
            0.0,
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    best_name, best_score = (
        ranked[0]
    )

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
    # Music-only / ambience.
    # --------------------------------------------------------

    music_score = audio_scores.get(
        "music",
        0.0,
    )

    if (
        best_score
        < SIGNIFICANT_THRESHOLD
    ):

        if music_score >= 0.35:

            return (
                "MUSIC_ONLY",
                music_score,
                best_score
                - second_score,
            )

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


# ============================================================
# CONFIDENCE BAND
# ============================================================

def confidence_band(
    confidence: float,
    margin: float,
) -> str:

    if (
        confidence >= STRONG_THRESHOLD
        and margin >= 0.08
    ):

        return "strong"

    if (
        confidence >= SIGNIFICANT_THRESHOLD
        and margin >= 0.04
    ):

        return "medium"

    if confidence >= 0.30:

        return "weak"

    return "none"


# ============================================================
# EVENT RATIONALE
# ============================================================

def build_rationale(
    event_name: str,
    audio: Dict[str, float],
    visual: Dict[str, float],
) -> str:

    if event_name == "WEAPON_FIRE":

        return (
            "Weapon-related visual evidence "
            "is aligned with gun/fire or "
            "impact-like audio."
        )

    if event_name == "SWORD_IMPACT":

        return (
            "Sword/weapon visual evidence "
            "aligns with sword or impact audio."
        )

    if event_name == "PHYSICAL_IMPACT":

        return (
            "Physical-combat imagery aligns "
            "with an impact-like transient."
        )

    if event_name == "EXPLOSION":

        return (
            "Explosion-like audio aligns "
            "with explosion imagery."
        )

    if event_name == "WHOOSH_ACTION":

        return (
            "Fast-motion imagery aligns "
            "with a whoosh-like transient."
        )

    if event_name == "FOOTSTEP_ACTION":

        return (
            "Walking/running imagery aligns "
            "with footsteps-like audio."
        )

    if event_name == "SCREAM":

        return (
            "Scream-like audio aligns "
            "with a visible vocal reaction."
        )

    if event_name == "GLASS_BREAK":

        return (
            "Glass-related audio aligns "
            "with visible glass/debris imagery."
        )

    if event_name == "MUSIC_ONLY":

        return (
            "No strong semantic SFX/visual "
            "pair was found; soundtrack dominates."
        )

    return (
        "No sufficiently strong audiovisual "
        "event pair was found."
    )


# ============================================================
# MERGE SIGNIFICANT EVENTS
# ============================================================

def merge_significant_events(
    events: List[Dict],
) -> List[Dict]:

    candidates = [
        item
        for item in events
        if item.get(
            "event_name"
        )
        not in (
            "NONE",
            "MUSIC_ONLY",
        )
        and item.get(
            "confidence",
            0.0,
        ) >= SIGNIFICANT_THRESHOLD
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

        same_event = (
            candidate["event_name"]
            == previous["event_name"]
        )

        close = (
            candidate["time"]
            - previous["time"]
            < EVENT_MERGE_DISTANCE
        )

        if same_event and close:

            if (
                candidate["confidence"]
                > previous["confidence"]
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
        "🎬 AUDIO-VISUAL EVENT ALIGNER"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load existing analysis
    # --------------------------------------------------------

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

    if not audio_windows:

        raise RuntimeError(
            "No window_analyses found in:\n"
            f"{AUDIO_SCAN_PATH}"
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
    # MobileCLIP
    # --------------------------------------------------------

    model, preprocess, tokenizer = (
        load_mobileclip()
    )

    text_embeddings = (
        build_text_embeddings(
            model,
            tokenizer,
        )
    )

    # --------------------------------------------------------
    # Process windows
    # --------------------------------------------------------

    all_events = []

    total = len(
        audio_windows
    )

    print()
    print(
        "Analyzing audiovisual alignment..."
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

        audio_scores = (
            extract_audio_scores(
                analysis
            )
        )

        print(
            f"[{index:03d}/{total:03d}] "
            f"{timestamp:7.3f}s ... ",
            end="",
            flush=True,
        )

        # ----------------------------------------------------
        # Temporal visual context
        # ----------------------------------------------------

        images = []

        valid_offsets = []

        for offset in FRAME_OFFSETS:

            image = reader.get_frame(
                timestamp + offset
            )

            if image is not None:

                images.append(
                    image
                )

                valid_offsets.append(
                    offset
                )

        # ----------------------------------------------------
        # Local motion
        # ----------------------------------------------------

        local_motion = 0.0

        if len(images) >= 2:

            differences = []

            for i in range(
                len(images) - 1
            ):

                differences.append(
                    frame_difference(
                        images[i],
                        images[i + 1],
                    )
                )

            if differences:

                local_motion = float(
                    np.mean(
                        differences
                    )
                )

        # ----------------------------------------------------
        # Image embeddings
        # ----------------------------------------------------

        image_embeddings = (
            encode_images(
                model,
                preprocess,
                images,
            )
        )

        visual_scores = (
            classify_visual_concepts(
                image_embeddings,
                text_embeddings,
            )
        )

        # ----------------------------------------------------
        # Joint
        # ----------------------------------------------------

        joint_scores = joint_score(
            audio=audio_scores,
            visual=visual_scores,
            local_motion=local_motion,
        )

        event_name, confidence, margin = (
            choose_joint_event(
                joint_scores,
                audio_scores,
                visual_scores,
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
                audio_scores.items(),
                key=lambda x: x[1],
            )
            if audio_scores
            else (
                "none",
                0.0,
            )
        )

        top_visual = (
            max(
                visual_scores.items(),
                key=lambda x: x[1],
            )
            if visual_scores
            else (
                "none",
                0.0,
            )
        )

        result = {

            "time": round(
                timestamp,
                4,
            ),

            "window": {
                "start": round(
                    safe_float(
                        analysis.get(
                            "window_start"
                        )
                    ),
                    4,
                ),
                "end": round(
                    safe_float(
                        analysis.get(
                            "window_end"
                        )
                    ),
                    4,
                ),
            },

            "event_name": event_name,

            "confidence": round(
                confidence,
                4,
            ),

            "confidence_margin": round(
                margin,
                4,
            ),

            "confidence_band": band,

            "local_visual_motion": round(
                local_motion,
                4,
            ),

            "reference": {
                "shot_index":
                    (
                        reference_shots.index(
                            reference_shot
                        ) + 1
                        if (
                            reference_shot
                            is not None
                        )
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

                "music_energy":
                    (
                        safe_float(
                            music_segment.get(
                                "music_energy"
                            )
                        )
                        if music_segment
                        else 0.0
                    ),

                "editorial_intensity":
                    (
                        safe_float(
                            music_segment.get(
                                "editorial_intensity"
                            )
                        )
                        if music_segment
                        else 0.0
                    ),
            },

            "audio": {

                "top_event":
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
                    in audio_scores.items()
                },
            },

            "visual": {

                "top_concept":
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
                    in visual_scores.items()
                },
            },

            "joint_scores": {
                key: round(
                    value,
                    4,
                )
                for key, value
                in sorted(
                    joint_scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            },

            "rationale":
                build_rationale(
                    event_name,
                    audio_scores,
                    visual_scores,
                ),
        }

        all_events.append(
            result
        )

        if event_name not in (
            "NONE",
            "MUSIC_ONLY",
        ):

            print(
                f"{event_name:17s} "
                f"{confidence:.3f}"
            )

        else:

            print(
                f"{event_name:17s}"
            )

    # --------------------------------------------------------
    # Significant compact event map
    # --------------------------------------------------------

    significant_events = (
        merge_significant_events(
            all_events
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    event_counts = Counter(
        event["event_name"]
        for event in significant_events
    )

    strong_counts = Counter(
        event["event_name"]
        for event in significant_events
        if event["confidence_band"]
        == "strong"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {

        "metadata": {

            "engine":
                "audio_visual_event_aligner_v1",

            "model":
                MODEL_NAME,

            "pretrained":
                PRETRAINED,

            "device":
                DEVICE,

            "reference_duration":
                round(
                    reader.duration,
                    4,
                ),

            "windows_analyzed":
                len(all_events),

            "significant_events":
                len(
                    significant_events
                ),

            "significant_threshold":
                SIGNIFICANT_THRESHOLD,

            "strong_threshold":
                STRONG_THRESHOLD,
        },

        "summary": {

            "significant_event_counts":
                dict(
                    event_counts
                ),

            "strong_event_counts":
                dict(
                    strong_counts
                ),
        },

        "significant_events":
            significant_events,

        "all_windows":
            all_events,
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
    del text_embeddings

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
        "✅ AUDIO-VISUAL EVENT ALIGNMENT COMPLETE"
    )
    print("=" * 72)

    print()

    print(
        f"Windows analyzed    : "
        f"{len(all_events)}"
    )

    print(
        f"Significant events  : "
        f"{len(significant_events)}"
    )

    print()

    print(
        "Significant event types:"
    )

    if event_counts:

        for name, count in (
            event_counts.most_common()
        ):

            print(
                f"  {name:20s}: "
                f"{count}"
            )

    else:

        print(
            "  NONE"
        )

    print()

    print(
        "Strong events:"
    )

    strong_events = [
        event
        for event in significant_events
        if event[
            "confidence_band"
        ] == "strong"
    ]

    for event in strong_events[:30]:

        print(
            f"  "
            f"{event['time']:7.3f}s | "
            f"{event['event_name']:20s} | "
            f"{event['confidence']:.3f} | "
            f""
            f"audio={event['audio']['top_event']} "
            f"visual={event['visual']['top_concept']}"
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