from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import open_clip


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

REFERENCE_VIDEO = BASE_DIR / "data/references/reference_video.mp4"
CLAP_JSON = BASE_DIR / "data/outputs/reference_clap_sfx_scan.json"

OUTPUT_JSON = BASE_DIR / "data/outputs/reference_av_diagnostics_v4.json"
GUN_SHEET = BASE_DIR / "data/outputs/av_gun_candidates_v4.jpg"
SWORD_SHEET = BASE_DIR / "data/outputs/av_sword_candidates_v4.jpg"
IMPACT_SHEET = BASE_DIR / "data/outputs/av_impact_candidates_v4.jpg"


# ============================================================
# CONFIG
# ============================================================

FRAME_OFFSETS = [-0.20, 0.0, 0.20]

# A little larger than a pure center crop so small weapons
# still have a chance of appearing.
CENTER_CROP_RATIO = 0.78

# Independent positive-vs-negative calibration.
CALIBRATION_SCALE = 10.0

TOP_CANDIDATES = 12


# ============================================================
# VISUAL CONCEPTS
# ============================================================

VISUAL_CONCEPTS = {

    "gun_weapon": {
        "positive": [
            "anime character holding a gun",
            "anime handgun in a character's hand",
            "anime pistol",
            "anime firearm",
            "anime character aiming a gun",
            "anime character pointing a handgun",
            "anime rifle or firearm",
        ],
        "negative": [
            "anime character with no weapon",
            "anime face close up with no weapon",
            "anime character standing empty handed",
            "anime landscape",
            "anime environment",
            "anime sword",
        ],
    },

    "gun_fire": {
        "positive": [
            "anime gun firing",
            "anime handgun firing",
            "anime muzzle flash",
            "bright muzzle flash from a gun",
            "anime firearm muzzle flash",
            "gun firing at close range in anime",
        ],
        "negative": [
            "anime character not firing a weapon",
            "anime character holding a gun but not firing",
            "anime face close up",
            "anime sword attack",
            "anime explosion in the distance",
            "anime environment",
        ],
    },

    "sword_weapon": {
        "positive": [
            "anime character holding a sword",
            "anime sword in hand",
            "anime character attacking with a sword",
            "anime sword fight",
            "anime katana",
            "anime blade weapon",
        ],
        "negative": [
            "anime character with no weapon",
            "anime handgun",
            "anime environment",
            "anime face close up",
            "anime character standing empty handed",
        ],
    },

    "combat": {
        "positive": [
            "anime combat scene",
            "anime fight",
            "anime character attacking another character",
            "anime battle",
            "anime action fight",
        ],
        "negative": [
            "anime peaceful scene",
            "anime character standing still",
            "anime landscape",
            "anime environment",
            "anime calm conversation",
        ],
    },

    "physical_hit": {
        "positive": [
            "anime character being punched",
            "anime punch impact",
            "anime physical attack impact",
            "anime character being struck",
            "anime hand to hand impact",
            "anime close range hit",
        ],
        "negative": [
            "anime character standing still",
            "anime peaceful scene",
            "anime landscape",
            "anime character posing without attack",
            "anime environment",
        ],
    },

    "explosion": {
        "positive": [
            "anime explosion",
            "large explosion in anime",
            "anime fireball",
            "anime blast",
            "anime explosive impact",
            "anime smoke and fire explosion",
        ],
        "negative": [
            "anime character close up",
            "anime calm scene",
            "anime environment",
            "anime sword",
            "anime handgun",
            "anime character standing still",
        ],
    },

    "fast_motion": {
        "positive": [
            "anime high speed action",
            "anime character moving extremely fast",
            "anime motion blur action",
            "anime character dashing",
            "anime fast attack",
        ],
        "negative": [
            "anime static portrait",
            "anime character standing still",
            "anime landscape",
            "anime calm conversation",
            "anime close up without motion",
        ],
    },

    "walking_running": {
        "positive": [
            "anime character running",
            "anime character walking",
            "anime character sprinting",
            "anime character moving forward",
        ],
        "negative": [
            "anime character standing still",
            "anime face close up",
            "anime explosion",
            "anime sword fight",
            "anime gunfight",
        ],
    },

    "scream": {
        "positive": [
            "anime character screaming",
            "anime character shouting",
            "anime character yelling",
            "anime open mouth scream",
            "dramatic anime scream",
        ],
        "negative": [
            "anime character with closed mouth",
            "anime environment",
            "anime peaceful scene",
            "anime landscape",
            "anime character standing silently",
        ],
    },

    "glass_break": {
        "positive": [
            "anime broken glass",
            "anime glass shattering",
            "anime shattered window",
            "anime character breaking glass",
            "anime shards of glass",
        ],
        "negative": [
            "anime character close up",
            "anime explosion",
            "anime sword",
            "anime gun",
            "anime peaceful scene",
        ],
    },

    "impact_visual": {
        "positive": [
            "dramatic anime impact frame",
            "anime attack impact",
            "anime collision",
            "anime impact moment",
            "anime powerful hit",
            "anime visual impact",
        ],
        "negative": [
            "anime static portrait",
            "anime peaceful scene",
            "anime empty environment",
            "anime character standing still",
        ],
    },

    "face_closeup": {
        "positive": [
            "anime character face close up",
            "dramatic anime face closeup",
            "anime eyes close up",
            "anime facial close up",
        ],
        "negative": [
            "anime wide environment",
            "anime full body shot",
            "anime landscape",
            "anime explosion",
        ],
    },

    "environment": {
        "positive": [
            "anime environment",
            "anime landscape",
            "anime city background",
            "anime building interior",
            "anime wide establishing shot",
        ],
        "negative": [
            "anime character face close up",
            "anime combat",
            "anime gunfire",
            "anime sword fight",
            "anime physical attack",
        ],
    },
}


# ============================================================
# HELPERS
# ============================================================

def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def get_timestamp(window: Dict[str, Any]) -> float:
    for key in [
        "timestamp",
        "time",
        "center_time",
        "center",
        "start",
    ]:
        if key in window:
            return safe_float(window[key])

    return 0.0


def get_group_scores(window: Dict[str, Any]) -> Dict[str, float]:
    scores = (
        window.get("group_scores")
        or window.get("scores")
        or window.get("audio_scores")
        or {}
    )

    if not isinstance(scores, dict):
        return {}

    result = {}

    for key, value in scores.items():

        if isinstance(value, dict):
            for nested_key in ["score", "similarity", "probability"]:
                if nested_key in value:
                    result[str(key)] = safe_float(value[nested_key])
                    break
        else:
            result[str(key)] = safe_float(value)

    return result


def audio_score(
    scores: Dict[str, float],
    names: List[str],
) -> float:

    vals = []

    for name in names:
        if name in scores:
            vals.append(scores[name])

    return max(vals) if vals else 0.0


def build_audio_evidence(scores: Dict[str, float]) -> Dict[str, float]:

    return {
        "gunshot": audio_score(
            scores,
            ["gunshot"],
        ),

        "pistol": audio_score(
            scores,
            ["pistol", "pistol_firing"],
        ),

        "impact": audio_score(
            scores,
            ["impact", "heavy_impact", "punch"],
        ),

        "sword": audio_score(
            scores,
            ["sword", "weapon_clash", "sword_clash"],
        ),

        "whoosh": audio_score(
            scores,
            ["whoosh"],
        ),

        "footsteps": audio_score(
            scores,
            ["footsteps"],
        ),

        "glass": audio_score(
            scores,
            ["glass"],
        ),

        "scream": audio_score(
            scores,
            ["scream"],
        ),

        "explosion": audio_score(
            scores,
            ["explosion"],
        ),

        "music": audio_score(
            scores,
            ["music"],
        ),
    }


# ============================================================
# MODEL LOADING
# ============================================================

def load_mobileclip():

    print("Loading MobileCLIP2...")

    candidates = [
        "mobileclip2_s0",
        "mobileclip_s0",
        "MobileCLIP2-S0",
        "MobileCLIP-S0",
    ]

    available = []

    try:
        for name, tag in open_clip.list_pretrained():
            low = name.lower()

            if "mobileclip" in low:
                available.append((name, tag))
    except Exception:
        pass

    # Prefer explicit known names.
    ordered = []

    for candidate in candidates:

        for name, tag in available:

            if name.lower() == candidate.lower():
                ordered.append((name, tag))

    # Then any MobileCLIP2 candidate.
    for item in available:

        if item not in ordered:
            ordered.append(item)

    last_error = None

    for model_name, pretrained_tag in ordered:

        try:

            print(
                f"Trying: model={model_name}, "
                f"pretrained={pretrained_tag}"
            )

            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained_tag,
            )

            tokenizer = open_clip.get_tokenizer(model_name)

            device = "mps" if torch.backends.mps.is_available() else "cpu"

            model = model.to(device)
            model.eval()

            print(f"✅ MobileCLIP loaded on {device}")

            return model, preprocess, tokenizer, device, model_name, pretrained_tag

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Could not load MobileCLIP. "
        f"Last error: {last_error}"
    )


# ============================================================
# TEXT EMBEDDINGS
# ============================================================

def build_visual_embeddings(
    model,
    tokenizer,
    device: str,
) -> Dict[str, Dict[str, torch.Tensor]]:

    print("\nBuilding independent visual concept calibrators...")

    result = {}

    with torch.no_grad():

        for concept, prompts in VISUAL_CONCEPTS.items():

            positive_tokens = tokenizer(prompts["positive"])
            negative_tokens = tokenizer(prompts["negative"])

            positive_tokens = positive_tokens.to(device)
            negative_tokens = negative_tokens.to(device)

            pos = model.encode_text(positive_tokens)
            neg = model.encode_text(negative_tokens)

            pos = normalize(pos).mean(dim=0, keepdim=True)
            neg = normalize(neg).mean(dim=0, keepdim=True)

            pos = normalize(pos)
            neg = normalize(neg)

            result[concept] = {
                "positive": pos,
                "negative": neg,
            }

    print(f"✅ {len(result)} independent concepts ready")

    return result


# ============================================================
# VIDEO
# ============================================================

class VideoReader:

    def __init__(self, path: Path):

        self.cap = cv2.VideoCapture(str(path))

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to open reference video: {path}"
            )

        self.fps = safe_float(
            self.cap.get(cv2.CAP_PROP_FPS),
            29.97,
        )

        self.frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.duration = (
            self.frames / self.fps
            if self.fps > 0
            else 0
        )

    def read(self, t: float) -> np.ndarray | None:

        t = max(0.0, min(t, self.duration - 0.001))

        self.cap.set(
            cv2.CAP_PROP_POS_MSEC,
            t * 1000.0,
        )

        ok, frame = self.cap.read()

        if not ok:
            return None

        return cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

    @staticmethod
    def center_crop(
        frame: np.ndarray,
        ratio: float,
    ) -> np.ndarray:

        h, w = frame.shape[:2]

        crop_w = int(w * ratio)
        crop_h = int(h * ratio)

        x1 = max(0, (w - crop_w) // 2)
        y1 = max(0, (h - crop_h) // 2)

        x2 = x1 + crop_w
        y2 = y1 + crop_h

        return frame[y1:y2, x1:x2]

    def close(self):
        self.cap.release()


# ============================================================
# VISUAL SCORING
# ============================================================

def score_frame(
    frame_rgb: np.ndarray,
    model,
    preprocess,
    embeddings,
    device: str,
) -> Dict[str, float]:

    variants = [
        frame_rgb,
        VideoReader.center_crop(
            frame_rgb,
            CENTER_CROP_RATIO,
        ),
    ]

    image_tensors = []

    for img in variants:

        pil = Image.fromarray(img)

        tensor = preprocess(pil)

        image_tensors.append(tensor)

    images = torch.stack(image_tensors).to(device)

    with torch.no_grad():

        image_features = model.encode_image(images)

        image_features = normalize(image_features)

    scores = {}

    for concept, emb in embeddings.items():

        pos_sim = (
            image_features
            @ emb["positive"].T
        ).squeeze(-1)

        neg_sim = (
            image_features
            @ emb["negative"].T
        ).squeeze(-1)

        raw = (
            pos_sim - neg_sim
        ) * CALIBRATION_SCALE

        calibrated = torch.sigmoid(raw)

        # Max is useful for small objects/weapons.
        # Mean prevents one crop from completely dominating.
        score = (
            calibrated.max() * 0.75
            + calibrated.mean() * 0.25
        )

        scores[concept] = float(score.item())

    return scores


# ============================================================
# RELATIONSHIP EVIDENCE
# ============================================================

def relationship_scores(
    audio: Dict[str, float],
    visual: Dict[str, float],
) -> Dict[str, float]:

    gun_audio = max(
        audio["gunshot"],
        audio["pistol"],
    )

    sword_audio = audio["sword"]

    impact_audio = audio["impact"]

    gun_visual = visual["gun_weapon"]
    gun_fire_visual = visual["gun_fire"]

    sword_visual = visual["sword_weapon"]

    combat_visual = visual["combat"]
    physical_visual = visual["physical_hit"]

    explosion_visual = visual["explosion"]

    scream_visual = visual["scream"]

    glass_visual = visual["glass_break"]

    whoosh_visual = visual["fast_motion"]

    # --------------------------------------------------------
    # IMPORTANT:
    # These are evidence scores, NOT final classifications.
    # --------------------------------------------------------

    gun_audio_visual = (
        0.45 * gun_audio
        + 0.30 * gun_visual
        + 0.25 * gun_fire_visual
    )

    gun_impact_visual = (
        0.20 * impact_audio
        + 0.35 * gun_visual
        + 0.45 * gun_fire_visual
    )

    sword_score = (
        0.35 * sword_audio
        + 0.40 * sword_visual
        + 0.25 * combat_visual
    )

    physical_score = (
        0.45 * impact_audio
        + 0.35 * physical_visual
        + 0.20 * combat_visual
    )

    explosion_score = (
        0.45 * audio["explosion"]
        + 0.45 * explosion_visual
        + 0.10 * impact_audio
    )

    whoosh_score = (
        0.45 * audio["whoosh"]
        + 0.35 * whoosh_visual
        + 0.20 * combat_visual
    )

    scream_score = (
        0.55 * audio["scream"]
        + 0.45 * scream_visual
    )

    glass_score = (
        0.60 * audio["glass"]
        + 0.40 * glass_visual
    )

    return {
        "gun_audio_visual": float(gun_audio_visual),
        "gun_impact_visual": float(gun_impact_visual),
        "sword": float(sword_score),
        "physical_impact": float(physical_score),
        "explosion": float(explosion_score),
        "whoosh": float(whoosh_score),
        "scream": float(scream_score),
        "glass_break": float(glass_score),
    }


# ============================================================
# EVENT HYPOTHESES
# ============================================================

def build_hypotheses(
    audio: Dict[str, float],
    visual: Dict[str, float],
    relations: Dict[str, float],
) -> List[Dict[str, Any]]:

    hypotheses = []

    hypotheses.append({
        "type": "WEAPON_FIRE",
        "score": relations["gun_audio_visual"],
        "evidence": {
            "audio_gunshot": audio["gunshot"],
            "audio_pistol": audio["pistol"],
            "audio_impact": audio["impact"],
            "visual_gun": visual["gun_weapon"],
            "visual_gun_fire": visual["gun_fire"],
        },
    })

    hypotheses.append({
        "type": "SWORD_ACTION",
        "score": relations["sword"],
        "evidence": {
            "audio_sword": audio["sword"],
            "visual_sword": visual["sword_weapon"],
            "visual_combat": visual["combat"],
        },
    })

    hypotheses.append({
        "type": "PHYSICAL_IMPACT",
        "score": relations["physical_impact"],
        "evidence": {
            "audio_impact": audio["impact"],
            "visual_physical_hit": visual["physical_hit"],
            "visual_combat": visual["combat"],
            "visual_impact": visual["impact_visual"],
        },
    })

    hypotheses.append({
        "type": "EXPLOSION",
        "score": relations["explosion"],
        "evidence": {
            "audio_explosion": audio["explosion"],
            "visual_explosion": visual["explosion"],
        },
    })

    hypotheses.append({
        "type": "WHOOSH_ACTION",
        "score": relations["whoosh"],
        "evidence": {
            "audio_whoosh": audio["whoosh"],
            "visual_fast_motion": visual["fast_motion"],
            "visual_combat": visual["combat"],
        },
    })

    hypotheses.append({
        "type": "SCREAM",
        "score": relations["scream"],
        "evidence": {
            "audio_scream": audio["scream"],
            "visual_scream": visual["scream"],
        },
    })

    hypotheses.append({
        "type": "GLASS_BREAK",
        "score": relations["glass_break"],
        "evidence": {
            "audio_glass": audio["glass"],
            "visual_glass": visual["glass_break"],
        },
    })

    hypotheses.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return hypotheses


# ============================================================
# AUDIO INPUT
# ============================================================

def load_audio_windows() -> List[Dict[str, Any]]:

    with open(CLAP_JSON, "r") as f:
        data = json.load(f)

    windows = (
        data.get("window_analyses")
        or data.get("windows")
        or data.get("analyses")
        or []
    )

    if not windows:
        raise RuntimeError(
            "Could not find audio windows in "
            f"{CLAP_JSON}"
        )

    return windows


# ============================================================
# CONTACT SHEET
# ============================================================

def make_contact_sheet(
    video: VideoReader,
    candidates: List[Dict[str, Any]],
    path: Path,
    title: str,
):

    if not candidates:
        print(f"⚠️ No candidates for {title}")
        return

    font = ImageFont.load_default()

    cells = []

    for item in candidates[:TOP_CANDIDATES]:

        t = item["timestamp"]

        frame = video.read(t)

        if frame is None:
            continue

        image = Image.fromarray(frame)

        image.thumbnail((360, 205))

        canvas = Image.new(
            "RGB",
            (380, 260),
            "black",
        )

        x = (380 - image.width) // 2
        y = 5

        canvas.paste(
            image,
            (x, y),
        )

        draw = ImageDraw.Draw(canvas)

        text = (
            f"{t:.3f}s\n"
            f"winner: {item['winner']}\n"
            f"score: {item['winner_score']:.3f}\n"
            f"gun={item['visual'].get('gun_weapon', 0):.3f}  "
            f"fire={item['visual'].get('gun_fire', 0):.3f}\n"
            f"sword={item['visual'].get('sword_weapon', 0):.3f}  "
            f"impact={item['visual'].get('physical_hit', 0):.3f}"
        )

        draw.multiline_text(
            (8, 215),
            text,
            fill="white",
            font=font,
            spacing=2,
        )

        cells.append(canvas)

    cols = 3
    rows = math.ceil(len(cells) / cols)

    sheet = Image.new(
        "RGB",
        (cols * 380, rows * 260 + 35),
        "black",
    )

    draw = ImageDraw.Draw(sheet)

    draw.text(
        (10, 8),
        title,
        fill="white",
        font=font,
    )

    for i, cell in enumerate(cells):

        x = (i % cols) * 380
        y = 35 + (i // cols) * 260

        sheet.paste(
            cell,
            (x, y),
        )

    sheet.save(path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("🎬 AUDIO-VISUAL DIAGNOSTIC ALIGNER V4")
    print("=" * 72)

    print(f"\nReference video : {REFERENCE_VIDEO}")
    print(f"CLAP scan       : {CLAP_JSON}")

    windows = load_audio_windows()

    print(f"\nAudio windows   : {len(windows)}")

    model, preprocess, tokenizer, device, model_name, pretrained_tag = (
        load_mobileclip()
    )

    embeddings = build_visual_embeddings(
        model,
        tokenizer,
        device,
    )

    video = VideoReader(
        REFERENCE_VIDEO
    )

    print(
        f"\nReference video : "
        f"{video.duration:.3f}s @ {video.fps:.3f}fps"
    )

    results = []

    print("\nAnalyzing evidence independently...\n")

    for index, window in enumerate(
        windows,
        start=1,
    ):

        timestamp = get_timestamp(window)

        raw_audio_scores = get_group_scores(window)

        audio = build_audio_evidence(
            raw_audio_scores
        )

        frame_scores = []

        for offset in FRAME_OFFSETS:

            frame = video.read(
                timestamp + offset
            )

            if frame is None:
                continue

            visual_scores = score_frame(
                frame,
                model,
                preprocess,
                embeddings,
                device,
            )

            frame_scores.append(
                visual_scores
            )

        visual = {}

        if frame_scores:

            for concept in VISUAL_CONCEPTS:

                values = [
                    x[concept]
                    for x in frame_scores
                ]

                visual[concept] = float(
                    max(values) * 0.75
                    + np.mean(values) * 0.25
                )

        else:

            visual = {
                concept: 0.0
                for concept in VISUAL_CONCEPTS
            }

        relations = relationship_scores(
            audio,
            visual,
        )

        hypotheses = build_hypotheses(
            audio,
            visual,
            relations,
        )

        winner = hypotheses[0]

        top_visual = sorted(
            visual.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        top_audio = sorted(
            audio.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        result = {
            "timestamp": round(timestamp, 3),

            "audio": audio,

            "visual": visual,

            "relationship": relations,

            "hypotheses": hypotheses,

            "top_visual": [
                {
                    "concept": name,
                    "score": round(score, 4),
                }
                for name, score in top_visual
            ],

            "top_audio": [
                {
                    "concept": name,
                    "score": round(score, 4),
                }
                for name, score in top_audio
            ],

            "winner": winner["type"],
            "winner_score": round(
                winner["score"],
                4,
            ),
        }

        results.append(result)

        print(
            f"[{index:03d}/{len(windows):03d}] "
            f"{timestamp:7.3f}s | "
            f"{winner['type']:<20} "
            f"{winner['score']:.3f} | "
            f"visual: "
            f"{top_visual[0][0]}="
            f"{top_visual[0][1]:.2f}"
        )

    video.close()

    # ========================================================
    # SORT IMPORTANT CANDIDATES
    # ========================================================

    gun_candidates = sorted(
        results,
        key=lambda x: (
            x["relationship"]["gun_audio_visual"]
            + x["visual"]["gun_weapon"]
            + x["visual"]["gun_fire"]
        ),
        reverse=True,
    )

    sword_candidates = sorted(
        results,
        key=lambda x: (
            x["relationship"]["sword"]
            + x["visual"]["sword_weapon"]
        ),
        reverse=True,
    )

    impact_candidates = sorted(
        results,
        key=lambda x: (
            x["relationship"]["physical_impact"]
            + x["visual"]["impact_visual"]
        ),
        reverse=True,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output = {
        "meta": {
            "version": "v4",
            "purpose": (
                "Independent audio/visual evidence diagnostics "
                "before final audiovisual event classification."
            ),
            "model": model_name,
            "pretrained": pretrained_tag,
            "device": device,
            "reference_duration": video.duration,
        },

        "summary": {
            "windows": len(results),

            "top_winners": {},
            
            "best_weapon_fire_candidate": (
                gun_candidates[0]["timestamp"]
                if gun_candidates
                else None
            ),

            "best_sword_candidate": (
                sword_candidates[0]["timestamp"]
                if sword_candidates
                else None
            ),

            "best_impact_candidate": (
                impact_candidates[0]["timestamp"]
                if impact_candidates
                else None
            ),
        },

        "windows": results,
    }

    counts = {}

    for item in results:

        winner = item["winner"]

        counts[winner] = (
            counts.get(winner, 0)
            + 1
        )

    output["summary"]["top_winners"] = counts

    with open(
        OUTPUT_JSON,
        "w",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # ========================================================
    # CONTACT SHEETS
    # ========================================================

    make_contact_sheet(
        video=VideoReader(REFERENCE_VIDEO),
        candidates=gun_candidates,
        path=GUN_SHEET,
        title="TOP GUN / WEAPON FIRE VISUAL CANDIDATES",
    )

    make_contact_sheet(
        video=VideoReader(REFERENCE_VIDEO),
        candidates=sword_candidates,
        path=SWORD_SHEET,
        title="TOP SWORD / BLADE VISUAL CANDIDATES",
    )

    make_contact_sheet(
        video=VideoReader(REFERENCE_VIDEO),
        candidates=impact_candidates,
        path=IMPACT_SHEET,
        title="TOP PHYSICAL IMPACT VISUAL CANDIDATES",
    )

    print("\n" + "=" * 72)
    print("✅ AUDIO-VISUAL DIAGNOSTIC V4 COMPLETE")
    print("=" * 72)

    print(
        f"\nWindows analyzed : {len(results)}"
    )

    print("\nWinner distribution:")

    for key, value in sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(
            f"  {key:<24}: {value}"
        )

    print("\nBest candidates:")

    if gun_candidates:
        g = gun_candidates[0]

        print(
            f"  GUN      : "
            f"{g['timestamp']:.3f}s | "
            f"{g['relationship']['gun_audio_visual']:.3f} | "
            f"gun={g['visual']['gun_weapon']:.3f} | "
            f"fire={g['visual']['gun_fire']:.3f} | "
            f"gunshot={g['audio']['gunshot']:.3f}"
        )

    if sword_candidates:
        s = sword_candidates[0]

        print(
            f"  SWORD    : "
            f"{s['timestamp']:.3f}s | "
            f"{s['relationship']['sword']:.3f} | "
            f"sword={s['visual']['sword_weapon']:.3f} | "
            f"sword_audio={s['audio']['sword']:.3f}"
        )

    if impact_candidates:
        i = impact_candidates[0]

        print(
            f"  IMPACT   : "
            f"{i['timestamp']:.3f}s | "
            f"{i['relationship']['physical_impact']:.3f} | "
            f"physical={i['visual']['physical_hit']:.3f} | "
            f"impact_audio={i['audio']['impact']:.3f}"
        )

    print("\nOutputs:")

    print(
        f"  {OUTPUT_JSON}"
    )

    print(
        f"  {GUN_SHEET}"
    )

    print(
        f"  {SWORD_SHEET}"
    )

    print(
        f"  {IMPACT_SHEET}"
    )

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()