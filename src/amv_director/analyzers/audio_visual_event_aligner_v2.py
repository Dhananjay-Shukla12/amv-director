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
    / "data/outputs/reference_audio_visual_events_v2.json"
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

FRAME_OFFSETS = [
    -0.20,
    0.0,
    0.20,
]

BATCH_SIZE = 8

SIGNIFICANT_THRESHOLD = 0.38
STRONG_THRESHOLD = 0.60

EVENT_MERGE_DISTANCE = 0.35


# ============================================================
# VISUAL CONCEPTS
# ============================================================

VISUAL_PROMPTS = {

    "gun_weapon": [
        "a character holding a gun",
        "a firearm in a character's hand",
        "a pistol visible in an anime scene",
        "a rifle visible in an anime scene",
        "a character aiming a firearm",
    ],

    "gun_fire_action": [
        "a character firing a gun",
        "a character shooting a firearm",
        "a gun being fired",
        "a muzzle flash from a gun",
        "a character pulling a gun trigger",
    ],

    "sword_weapon": [
        "a character holding a sword",
        "a sword or blade visible",
        "a character wielding a blade",
        "a sword fight",
        "two characters fighting with swords",
    ],

    "combat": [
        "anime characters fighting",
        "a character attacking another character",
        "an anime combat scene",
        "two characters fighting",
        "a character striking an opponent",
    ],

    "physical_hit": [
        "a character punching another character",
        "a character being hit",
        "a powerful punch",
        "a physical collision between characters",
        "a character striking another character",
    ],

    "explosion_visual": [
        "an explosion in an anime scene",
        "a large fireball",
        "a blast or explosion",
        "a powerful explosion",
        "an explosion behind a character",
    ],

    "fast_motion": [
        "a character moving extremely fast",
        "a fast anime action movement",
        "a character lunging forward",
        "a character rapidly moving",
        "a dynamic action pose",
    ],

    "walking_running": [
        "an anime character walking",
        "an anime character running",
        "a character moving on foot",
        "a character running forward",
        "a character's feet while running",
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
        "a dramatic hit",
        "an impact frame",
        "a character being knocked back",
        "a powerful collision",
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
            matches.append(shot)

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
# VIDEO READER
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

        self.frames = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        self.duration = (
            self.frames / self.fps
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
# LOCAL MOTION
# ============================================================

def motion_between(
    a: Optional[Image.Image],
    b: Optional[Image.Image],
) -> float:

    if a is None or b is None:
        return 0.0

    a_np = np.asarray(a)
    b_np = np.asarray(b)

    a_gray = cv2.cvtColor(
        a_np,
        cv2.COLOR_RGB2GRAY,
    )

    b_gray = cv2.cvtColor(
        b_np,
        cv2.COLOR_RGB2GRAY,
    )

    a_gray = cv2.resize(
        a_gray,
        (256, 144),
    )

    b_gray = cv2.resize(
        b_gray,
        (256, 144),
    )

    diff = np.mean(
        np.abs(
            a_gray.astype(
                np.float32
            )
            - b_gray.astype(
                np.float32
            )
        )
    )

    return clamp(
        diff / 35.0
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


def build_text_embeddings(
    model,
    tokenizer,
):

    embeddings = {}

    with torch.inference_mode():

        for concept, prompts in (
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

            embeddings[
                concept
            ] = features.detach()

    return embeddings


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

    results = []

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
        ).to(DEVICE)

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

        results.extend(
            [
                feature.detach()
                for feature in features
            ]
        )

        del batch
        del features

    return results


# ============================================================
# INDEPENDENT CONCEPT SCORING
# ============================================================

def visual_concept_scores(
    image_embeddings: List[torch.Tensor],
    text_embeddings: Dict[str, torch.Tensor],
) -> Dict[str, float]:

    if not image_embeddings:

        return {
            concept: 0.0
            for concept in text_embeddings
        }

    images = torch.stack(
        image_embeddings
    )

    result = {}

    for concept, text_features in (
        text_embeddings.items()
    ):

        # image × prompt similarity
        similarity = (
            images
            @ text_features.T
        )

        # We do NOT softmax across concepts.
        #
        # Each concept is independent.
        #
        # For several paraphrases, take the strongest
        # prompt with a small contribution from the
        # second strongest prompt.
        prompt_means = similarity.mean(
            dim=0
        )

        values = sorted(
            [
                float(v)
                for v
                in prompt_means.tolist()
            ],
            reverse=True,
        )

        if len(values) >= 2:

            raw = (
                values[0] * 0.70
                + values[1] * 0.30
            )

        else:

            raw = values[0]

        # Map approximate CLIP cosine range into [0,1].
        #
        # This is NOT a probability.
        score = clamp(
            (raw + 1.0) / 2.0
        )

        result[concept] = score

    return result


# ============================================================
# AUDIO
# ============================================================

def audio_scores(
    analysis: Dict,
) -> Dict[str, float]:

    return {
        key: safe_float(value)
        for key, value in analysis.get(
            "group_scores",
            {},
        ).items()
    }


# ============================================================
# EVENT-SPECIFIC JOINT LOGIC
# ============================================================

def joint_event_scores(
    audio: Dict[str, float],
    visual: Dict[str, float],
    motion: float,
) -> Dict[str, float]:

    # ------------------------------
    # AUDIO
    # ------------------------------

    gun = max(
        audio.get("gunshot", 0.0),
        audio.get("pistol_firing", 0.0),
    )

    explosion = audio.get(
        "explosion",
        0.0,
    )

    sword_audio = max(
        audio.get("sword", 0.0),
        audio.get("weapon_clash", 0.0),
    )

    impact_audio = audio.get(
        "impact",
        0.0,
    )

    whoosh_audio = audio.get(
        "whoosh",
        0.0,
    )

    footstep_audio = audio.get(
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

    # ------------------------------
    # VISUAL
    # ------------------------------

    gun_visual = visual.get(
        "gun_weapon",
        0.0,
    )

    gun_fire_visual = visual.get(
        "gun_fire_action",
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
        "explosion_visual",
        0.0,
    )

    fast_visual = visual.get(
        "fast_motion",
        0.0,
    )

    walk_visual = visual.get(
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

    # ========================================================
    # JOINT EVENTS
    # ========================================================

    scores = {

        "WEAPON_FIRE": clamp(
            max(
                gun * 0.65,
                impact_audio * gun_visual * 0.65,
            )
            + gun_visual * 0.15
            + gun_fire_visual * 0.25
        ),

        "SWORD_IMPACT": clamp(
            sword_audio * 0.55
            + sword_visual * 0.30
            + min(
                sword_audio,
                sword_visual,
            ) * 0.35
        ),

        "PHYSICAL_IMPACT": clamp(
            impact_audio * 0.50
            + hit_visual * 0.25
            + impact_visual * 0.20
            + combat_visual * 0.10
        ),

        "EXPLOSION": clamp(
            explosion * 0.65
            + explosion_visual * 0.40
        ),

        "WHOOSH_ACTION": clamp(
            whoosh_audio * 0.50
            + fast_visual * 0.32
            + motion * 0.18
        ),

        "FOOTSTEP_ACTION": clamp(
            footstep_audio * 0.60
            + walk_visual * 0.40
        ),

        "SCREAM": clamp(
            scream_audio * 0.65
            + scream_visual * 0.35
        ),

        "GLASS_BREAK": clamp(
            glass_audio * 0.60
            + glass_visual * 0.40
        ),
    }

    return scores


# ============================================================
# SELECT EVENT
# ============================================================

def select_event(
    scores: Dict[str, float],
    audio: Dict[str, float],
    visual: Dict[str, float],
) -> Tuple[str, float, float]:

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    if not ranked:

        return (
            "NONE",
            0.0,
            0.0,
        )

    best_name, best_score = ranked[0]

    second_score = (
        ranked[1][1]
        if len(ranked) > 1
        else 0.0
    )

    margin = (
        best_score
        - second_score
    )

    # ----------------------------------------------
    # Special protection for WEAPON_FIRE.
    #
    # We don't require the audio classifier itself
    # to say "gunshot". Visual evidence can rescue
    # a weak gunshot acoustic signal.
    # ----------------------------------------------

    gun_visual = max(
        visual.get(
            "gun_weapon",
            0.0,
        ),
        visual.get(
            "gun_fire_action",
            0.0,
        ),
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
        audio.get(
            "impact",
            0.0,
        ),
    )

    weapon_fire_hypothesis = clamp(
        gun_visual * 0.60
        + gun_audio * 0.40
    )

    if (
        gun_visual >= 0.68
        and gun_audio >= 0.18
        and weapon_fire_hypothesis
        >= SIGNIFICANT_THRESHOLD
    ):

        return (
            "WEAPON_FIRE",
            weapon_fire_hypothesis,
            max(
                margin,
                weapon_fire_hypothesis
                - second_score,
            ),
        )

    if (
        best_score
        < SIGNIFICANT_THRESHOLD
    ):

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
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "🎬 AUDIO-VISUAL EVENT ALIGNER V2"
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

    reader = VideoReader(
        REFERENCE_VIDEO_PATH
    )

    model, preprocess, tokenizer = (
        load_model()
    )

    text_embeddings = (
        build_text_embeddings(
            model,
            tokenizer,
        )
    )

    events = []

    print()
    print(
        "Analyzing audiovisual events..."
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

        images = []

        for offset in FRAME_OFFSETS:

            image = reader.get_frame(
                timestamp + offset
            )

            if image is not None:
                images.append(
                    image
                )

        # --------------------------------------------
        # Motion
        # --------------------------------------------

        motion_values = []

        for i in range(
            len(images) - 1
        ):

            motion_values.append(
                motion_between(
                    images[i],
                    images[i + 1],
                )
            )

        motion = (
            float(
                np.mean(
                    motion_values
                )
            )
            if motion_values
            else 0.0
        )

        # --------------------------------------------
        # Visual concepts
        # --------------------------------------------

        image_embeddings = encode_images(
            model,
            preprocess,
            images,
        )

        visual = (
            visual_concept_scores(
                image_embeddings,
                text_embeddings,
            )
        )

        # --------------------------------------------
        # Audio
        # --------------------------------------------

        audio = audio_scores(
            analysis
        )

        # --------------------------------------------
        # Joint
        # --------------------------------------------

        joint = joint_event_scores(
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
                    in visual.items()
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
                        safe_float(
                            reference_shot.get(
                                "start"
                            )
                        )
                        if reference_shot
                        else None
                    ),

                "end":
                    (
                        safe_float(
                            reference_shot.get(
                                "end"
                            )
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
            },
        }

        events.append(
            result
        )

        print(
            f"{event_name:18s} "
            f"{confidence:.3f}"
        )

    # ========================================================
    # COMPACT SIGNIFICANT EVENTS
    # ========================================================

    candidates = [
        event
        for event in events
        if event["event_name"]
        not in (
            "NONE",
            "MUSIC_ONLY",
        )
        and event["confidence"]
        >= SIGNIFICANT_THRESHOLD
    ]

    candidates.sort(
        key=lambda x: (
            x["event_name"],
            x["time"],
        )
    )

    significant = []

    for candidate in candidates:

        if not significant:

            significant.append(
                candidate
            )

            continue

        previous = significant[-1]

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

                significant[-1] = (
                    candidate
                )

        else:

            significant.append(
                candidate
            )

    significant.sort(
        key=lambda x: x["time"]
    )

    counts = Counter(
        event["event_name"]
        for event in significant
    )

    strong = [
        event
        for event in significant
        if event[
            "confidence_band"
        ] == "strong"
    ]

    # ========================================================
    # SAVE
    # ========================================================

    output = {

        "metadata": {

            "engine":
                "audio_visual_event_aligner_v2",

            "model":
                MODEL_NAME,

            "device":
                DEVICE,

            "audio_windows":
                len(audio_windows),

            "significant_events":
                len(significant),
        },

        "summary": {

            "event_counts":
                dict(counts),

            "strong_event_counts":
                dict(
                    Counter(
                        event[
                            "event_name"
                        ]
                        for event
                        in strong
                    )
                ),
        },

        "significant_events":
            significant,

        "all_windows":
            events,
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

    # ========================================================
    # CLEANUP
    # ========================================================

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

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 72)
    print(
        "✅ AUDIO-VISUAL EVENT ALIGNER V2 COMPLETE"
    )
    print("=" * 72)

    print()

    print(
        f"Windows analyzed   : "
        f"{len(events)}"
    )

    print(
        f"Significant events : "
        f"{len(significant)}"
    )

    print()

    print(
        "Event types:"
    )

    for name, count in (
        counts.most_common()
    ):

        print(
            f"  {name:20s}: "
            f"{count}"
        )

    print()

    print(
        "Strong events:"
    )

    for event in strong[:30]:

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
        f"Output:\n{OUTPUT_PATH}"
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()