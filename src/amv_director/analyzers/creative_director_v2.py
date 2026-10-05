from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

MUSIC_LOCK = (
    BASE_DIR
    / "data/outputs/music_locked_edit_plan_v3.json"
)

AV_GRAMMAR = (
    BASE_DIR
    / "data/outputs/reference_av_grammar_v1.json"
)

REFERENCE_SHOT_PLAN = (
    BASE_DIR
    / "data/outputs/reference_conditioned_shot_plan.json"
)

ALL_CLIPS = (
    BASE_DIR
    / "data/outputs/clip_analysis/all_clips_analysis.json"
)

SOURCE_QUALITY = (
    BASE_DIR
    / "data/outputs/clip_analysis/source_quality_analysis.json"
)

OUTPUT = (
    BASE_DIR
    / "data/outputs/creative_director_plan_v2.json"
)


# ============================================================
# DIRECTOR SETTINGS
# ============================================================

MAX_SCENE_REPEAT_SOFT = 3
MAX_CONSECUTIVE_SAME_SCENE = 1

QUALITY_FLOOR = 0.30

# How much the Director should prioritize:
# reference behavior > music role > source quality > novelty
WEIGHT_REFERENCE = 0.34
WEIGHT_MUSIC = 0.28
WEIGHT_QUALITY = 0.22
WEIGHT_VARIATION = 0.16


# ============================================================
# HELPERS
# ============================================================

def f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_json(path: Path) -> Any:
    with open(path, "r") as fh:
        return json.load(fh)


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def recursive_find_lists(
    obj: Any,
    names: set[str],
) -> Dict[str, List[Any]]:

    found: Dict[str, List[Any]] = {}

    def walk(x: Any):

        if isinstance(x, dict):

            for key, value in x.items():

                if (
                    key.lower() in names
                    and isinstance(value, list)
                ):
                    found.setdefault(
                        key.lower(),
                        value,
                    )

                walk(value)

        elif isinstance(x, list):

            for item in x:
                walk(item)

    walk(obj)

    return found


def duration_safe(
    start: Any,
    end: Any,
) -> float:

    return max(
        0.05,
        f(end) - f(start),
    )


def clamp(
    value: float,
    low: float,
    high: float,
) -> float:

    return max(
        low,
        min(high, value),
    )


# ============================================================
# MUSIC
# ============================================================

def music_segment_at(
    timestamp: float,
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:

    for segment in segments:

        start = f(segment.get("start"))
        end = f(segment.get("end"))

        if start <= timestamp <= end:
            return segment

    if not segments:
        return {}

    return min(
        segments,
        key=lambda x: abs(
            timestamp - f(
                x.get("start")
            )
        ),
    )


# ============================================================
# AV EFFECTS
# ============================================================

def get_effects(
    music_lock: Dict[str, Any],
) -> List[Dict[str, Any]]:

    effect_lane = music_lock.get(
        "effect_lane",
        {},
    )

    events = effect_lane.get(
        "events",
        [],
    )

    return safe_list(events)


def effects_for_window(
    start: float,
    end: float,
    effects: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    result = []

    for event in effects:

        timestamp = f(
            event.get("timestamp")
        )

        if start - 0.12 <= timestamp <= end + 0.12:
            result.append(event)

    return sorted(
        result,
        key=lambda x: f(
            x.get("timestamp")
        ),
    )


# ============================================================
# SOURCE SCENE NORMALIZATION
# ============================================================

def normalize_source_scenes(
    data: Any,
) -> List[Dict[str, Any]]:

    scenes = []

    def scan(obj: Any):

        if isinstance(obj, list):

            for item in obj:
                scan(item)

            return

        if not isinstance(obj, dict):
            return

        # ----------------------------------------------------
        # Direct scene-like object
        # ----------------------------------------------------

        if (
            "scene_id" in obj
            and (
                "start" in obj
                and "end" in obj
            )
        ):

            video_name = (
                obj.get("video")
                or obj.get("video_name")
                or obj.get("source_video")
                or obj.get("clip")
                or "unknown_source"
            )

            scenes.append({
                "video": str(video_name),

                "scene_id": obj.get(
                    "scene_id"
                ),

                "start": f(
                    obj.get("start")
                ),

                "end": f(
                    obj.get("end")
                ),

                "duration": duration_safe(
                    obj.get("start"),
                    obj.get("end"),
                ),

                "brightness": f(
                    obj.get("brightness")
                ),

                "contrast": f(
                    obj.get("contrast")
                ),

                "sharpness": f(
                    obj.get("sharpness")
                ),

                "edge_density": f(
                    obj.get("edge_density")
                ),

                "motion": f(
                    obj.get("motion")
                ),

                "labels": safe_list(
                    obj.get("labels")
                ),

                "scores": (
                    obj.get(
                        "scores",
                        {}
                    )
                    if isinstance(
                        obj.get("scores", {}),
                        dict,
                    )
                    else {}
                ),
            })

        for value in obj.values():
            scan(value)

    scan(data)

    # De-duplicate scene records.
    unique = {}

    for scene in scenes:

        key = (
            scene["video"],
            scene["scene_id"],
            round(scene["start"], 3),
            round(scene["end"], 3),
        )

        unique[key] = scene

    return list(
        unique.values()
    )


# ============================================================
# QUALITY
# ============================================================

def build_quality_map(
    quality_data: Any,
) -> Dict[Tuple[str, Any], float]:

    result = {}

    def scan(obj: Any):

        if isinstance(obj, list):

            for item in obj:
                scan(item)

            return

        if not isinstance(obj, dict):
            return

        if "scene_id" in obj:

            quality = None

            for key in (
                "quality",
                "quality_score",
                "overall_quality",
                "score",
            ):

                if key in obj:
                    quality = f(
                        obj[key]
                    )
                    break

            if quality is not None:

                video = (
                    obj.get("video")
                    or obj.get("video_name")
                    or obj.get("source_video")
                    or "unknown_source"
                )

                result[
                    (
                        str(video),
                        obj["scene_id"],
                    )
                ] = quality

        for value in obj.values():
            scan(value)

    scan(quality_data)

    return result


# ============================================================
# REFERENCE MATCH NORMALIZATION
# ============================================================

def normalize_reference_matches(
    data: Any,
) -> List[Dict[str, Any]]:

    candidates = []

    arrays = recursive_find_lists(
        data,
        {
            "matches",
            "shot_matches",
            "reference_shots",
            "assignments",
            "shots",
        },
    )

    raw_lists = list(
        arrays.values()
    )

    # Search every list recursively because the exact previous
    # matcher schema may evolve.
    for arr in raw_lists:

        for item in arr:

            if not isinstance(item, dict):
                continue

            source_scene = (
                item.get("source_scene")
                or item.get("matched_source")
                or item.get("source")
                or {}
            )

            if isinstance(
                source_scene,
                dict,
            ):

                video = (
                    source_scene.get("video")
                    or source_scene.get("video_name")
                    or source_scene.get("source_video")
                    or source_scene.get("clip")
                )

                scene_id = (
                    source_scene.get("scene_id")
                    or source_scene.get("id")
                )

                source_start = f(
                    source_scene.get(
                        "start"
                    )
                )

                source_end = f(
                    source_scene.get(
                        "end"
                    )
                )

            else:

                video = None
                scene_id = None
                source_start = 0.0
                source_end = 0.0

            # Alternate flat schema.
            if video is None:

                video = (
                    item.get("video")
                    or item.get("video_name")
                    or item.get("source_video")
                )

            if scene_id is None:

                scene_id = (
                    item.get("scene_id")
                    or item.get("source_scene_id")
                )

            reference_time = (
                item.get("reference_timestamp")
                or item.get("timestamp")
                or item.get("reference_time")
                or item.get("start")
                or 0.0
            )

            score = (
                item.get("score")
                or item.get("match_score")
                or item.get("similarity")
                or 0.0
            )

            if (
                video is None
                or scene_id is None
            ):
                continue

            candidates.append({

                "reference_time": f(
                    reference_time
                ),

                "video": str(video),

                "scene_id": scene_id,

                "source_start": source_start,

                "source_end": source_end,

                "score": f(score),

            })

    # De-duplicate.
    unique = {}

    for item in candidates:

        key = (
            round(item["reference_time"], 3),
            item["video"],
            item["scene_id"],
        )

        if (
            key not in unique
            or item["score"]
            > unique[key]["score"]
        ):

            unique[key] = item

    return list(
        unique.values()
    )


# ============================================================
# REFERENCE MATCH
# ============================================================

def nearest_reference_match(
    timestamp: float,
    matches: List[Dict[str, Any]],
) -> Dict[str, Any] | None:

    if not matches:
        return None

    return min(
        matches,
        key=lambda x: abs(
            timestamp
            - x["reference_time"]
        ),
    )


# ============================================================
# SOURCE SELECTION
# ============================================================

def source_visual_energy(
    scene: Dict[str, Any],
) -> float:

    motion = clamp(
        f(scene.get("motion")),
        0.0,
        1.0,
    )

    edge = clamp(
        f(scene.get("edge_density")),
        0.0,
        1.0,
    )

    contrast = clamp(
        f(scene.get("contrast")),
        0.0,
        1.0,
    )

    sharpness = clamp(
        f(scene.get("sharpness")),
        0.0,
        1.0,
    )

    return (
        0.35 * motion
        + 0.25 * edge
        + 0.20 * contrast
        + 0.20 * sharpness
    )


def source_visual_role(
    scene: Dict[str, Any],
) -> str:

    labels = [
        str(x).lower()
        for x in safe_list(
            scene.get("labels")
        )
    ]

    motion = f(
        scene.get("motion")
    )

    brightness = f(
        scene.get("brightness")
    )

    text = " ".join(labels)

    if any(
        keyword in text
        for keyword in (
            "explosion",
            "impact",
            "attack",
            "fight",
            "combat",
        )
    ):

        return "action"

    if any(
        keyword in text
        for keyword in (
            "face",
            "close",
            "eyes",
            "character",
        )
    ):

        if motion > 0.55:
            return "character_action"

        return "character"

    if motion > 0.60:
        return "motion"

    if brightness < 0.15:
        return "dark_atmosphere"

    return "environment"


def desired_visual_role(
    music_role: str,
    sequence: str | None,
    shot_index: int,
) -> str:

    # --------------------------------------------------------
    # Effects/grammar take priority over generic music role.
    # --------------------------------------------------------

    if sequence == "BUILD_TO_IMPACT":
        return "action"

    if sequence == "DIRECT_IMPACT":
        return "action"

    if sequence == "MOTION_ACCENT":
        return "motion"

    if sequence == "ACCENT":
        return "character"

    if music_role == "cinematic_zone":

        cycle = shot_index % 5

        if cycle in {0, 3}:
            return "environment"

        return "character"

    if music_role == "build_zone":
        return "character_action"

    if music_role == "impact_zone":
        return "action"

    if music_role == "accent_zone":
        return "character"

    if music_role == "action_zone":
        return "action"

    return "character"


def role_compatibility(
    source_role: str,
    desired_role: str,
) -> float:

    matrix = {

        "action": {
            "action": 1.0,
            "motion": 0.90,
            "character_action": 0.80,
            "character": 0.55,
            "environment": 0.30,
            "dark_atmosphere": 0.35,
        },

        "motion": {
            "action": 0.90,
            "motion": 1.0,
            "character_action": 0.90,
            "character": 0.65,
            "environment": 0.45,
            "dark_atmosphere": 0.40,
        },

        "character_action": {
            "action": 0.85,
            "motion": 0.90,
            "character_action": 1.0,
            "character": 0.85,
            "environment": 0.40,
            "dark_atmosphere": 0.45,
        },

        "character": {
            "action": 0.55,
            "motion": 0.65,
            "character_action": 0.85,
            "character": 1.0,
            "environment": 0.55,
            "dark_atmosphere": 0.60,
        },

        "environment": {
            "action": 0.30,
            "motion": 0.45,
            "character_action": 0.40,
            "character": 0.55,
            "environment": 1.0,
            "dark_atmosphere": 0.85,
        },

        "dark_atmosphere": {
            "action": 0.35,
            "motion": 0.40,
            "character_action": 0.45,
            "character": 0.60,
            "environment": 0.85,
            "dark_atmosphere": 1.0,
        },
    }

    return (
        matrix
        .get(
            desired_role,
            {},
        )
        .get(
            source_role,
            0.40,
        )
    )


def candidate_score(
    scene: Dict[str, Any],
    desired_role_name: str,
    reference_match: Dict[str, Any] | None,
    quality_map: Dict[Tuple[str, Any], float],
    usage_counts: Dict[Tuple[str, Any], int],
    previous_scene: Tuple[str, Any] | None,
) -> Tuple[float, Dict[str, Any]]:

    video = scene["video"]
    scene_id = scene["scene_id"]

    key = (
        video,
        scene_id,
    )

    quality = quality_map.get(
        key,
        0.60,
    )

    quality = clamp(
        quality,
        0.0,
        1.0,
    )

    source_role_name = source_visual_role(
        scene
    )

    compatibility = role_compatibility(
        source_role_name,
        desired_role_name,
    )

    energy = source_visual_energy(
        scene
    )

    # --------------------------------------------------------
    # Reference conditioning
    # --------------------------------------------------------

    reference_score = 0.50

    if reference_match:

        if (
            reference_match["video"]
            == video
            and reference_match["scene_id"]
            == scene_id
        ):

            reference_score = max(
                0.50,
                reference_match["score"],
            )

        else:

            reference_score = (
                0.35
                + 0.30 * compatibility
            )

    # --------------------------------------------------------
    # Variation
    # --------------------------------------------------------

    use_count = usage_counts.get(
        key,
        0,
    )

    variation = 1.0

    if use_count >= MAX_SCENE_REPEAT_SOFT:
        variation -= 0.35

    elif use_count > 0:
        variation -= (
            0.12
            * use_count
        )

    if (
        previous_scene is not None
        and previous_scene == key
    ):

        variation -= 0.90

    variation = clamp(
        variation,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    total = (
        WEIGHT_REFERENCE
        * reference_score

        + WEIGHT_MUSIC
        * compatibility

        + WEIGHT_QUALITY
        * quality

        + WEIGHT_VARIATION
        * variation
    )

    diagnostics = {

        "reference_score": round(
            reference_score,
            4,
        ),

        "music_role_compatibility": round(
            compatibility,
            4,
        ),

        "quality": round(
            quality,
            4,
        ),

        "visual_energy": round(
            energy,
            4,
        ),

        "variation": round(
            variation,
            4,
        ),

        "source_role": source_role_name,

        "reuse_count_before": use_count,

    }

    return (
        total,
        diagnostics,
    )


# ============================================================
# TEMPORAL WINDOW
# ============================================================

def choose_source_window(
    scene: Dict[str, Any],
    target_duration: float,
) -> Dict[str, Any]:

    start = f(
        scene.get("start")
    )

    end = f(
        scene.get("end")
    )

    available = max(
        0.05,
        end - start,
    )

    target = max(
        0.05,
        target_duration,
    )

    if available <= target:

        return {

            "start": round(
                start,
                4,
            ),

            "end": round(
                end,
                4,
            ),

            "duration": round(
                available,
                4,
            ),

            "mode": "full_scene",

        }

    # --------------------------------------------------------
    # Centered crop from the source scene.
    #
    # Later the renderer can choose more sophisticated temporal
    # windows using the temporal_window_analyzer.
    # --------------------------------------------------------

    excess = (
        available
        - target
    )

    crop_start = (
        start
        + excess * 0.35
    )

    crop_end = (
        crop_start
        + target
    )

    crop_end = min(
        crop_end,
        end,
    )

    return {

        "start": round(
            crop_start,
            4,
        ),

        "end": round(
            crop_end,
            4,
        ),

        "duration": round(
            crop_end
            - crop_start,
            4,
        ),

        "mode": "center_weighted_window",

    }


# ============================================================
# DIRECTOR INTENT
# ============================================================

def build_edit_intent(
    shot_index: int,
    music_segment: Dict[str, Any],
    effects: List[Dict[str, Any]],
    source_scene: Dict[str, Any],
) -> Dict[str, Any]:

    role = music_segment.get(
        "editorial_role",
        "cinematic_zone",
    )

    intensity = f(
        music_segment.get(
            "editorial_intensity",
            0.0,
        )
    )

    trend = f(
        music_segment.get(
            "energy_trend",
            0.0,
        )
    )

    sequence = None

    if effects:

        sequence = effects[0].get(
            "sequence_type"
        )

    desired_role_name = desired_visual_role(
        role,
        sequence,
        shot_index,
    )

    source_energy = source_visual_energy(
        source_scene
    )

    # --------------------------------------------------------
    # Camera behavior
    # --------------------------------------------------------

    if sequence == "BUILD_TO_IMPACT":

        camera = [
            "controlled_push",
            "directional_motion",
            "micro_acceleration",
            "impact_snap",
        ]

    elif sequence == "DIRECT_IMPACT":

        camera = [
            "hold",
            "impact_snap",
        ]

    elif sequence == "MOTION_ACCENT":

        camera = [
            "directional_move",
            "motion_follow",
        ]

    elif role == "cinematic_zone":

        camera = [
            "slow_drift",
            "subtle_parallax_feel",
        ]

    elif role == "build_zone":

        camera = [
            "gradual_push",
            "rising_motion",
        ]

    elif role == "impact_zone":

        camera = [
            "compressed_push",
            "impact_emphasis",
        ]

    elif role == "action_zone":

        camera = [
            "directional_motion",
            "fast_reframe",
        ]

    else:

        camera = [
            "controlled_hold",
        ]

    # --------------------------------------------------------
    # Cut behavior
    # --------------------------------------------------------

    if role in {
        "impact_zone",
        "action_zone",
    }:

        cut_behavior = "hard_resolve"

    elif sequence in {
        "DIRECT_IMPACT",
        "BUILD_TO_IMPACT",
    }:

        cut_behavior = "impact_resolve"

    elif role == "cinematic_zone":

        cut_behavior = "soft_editorial_resolve"

    else:

        cut_behavior = "clean_music_cut"

    # --------------------------------------------------------
    # Effect behavior
    # --------------------------------------------------------

    effect_actions = []

    for effect in effects:

        edit_sequence = effect.get(
            "edit_sequence",
            [],
        )

        for action in edit_sequence:

            if action not in effect_actions:

                effect_actions.append(
                    action
                )

    # --------------------------------------------------------
    # Creative variation
    # --------------------------------------------------------

    if (
        source_energy > 0.70
        and intensity > 0.65
    ):

        creative_mode = "aggressive"

    elif (
        intensity < 0.30
    ):

        creative_mode = "restrained"

    else:

        creative_mode = "controlled"

    return {

        "visual_role": desired_role_name,

        "music_role": role,

        "music_intensity": round(
            intensity,
            4,
        ),

        "energy_trend": round(
            trend,
            4,
        ),

        "reference_sequence": sequence,

        "camera_behavior": camera,

        "cut_behavior": cut_behavior,

        "effect_actions": effect_actions,

        "creative_mode": creative_mode,

        "director_priority": [

            "music_timing",

            "reference_editing_grammar",

            "source_visual_quality",

            "semantic_fit",

            "controlled_creative_variation",

        ],

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("🎬 CREATIVE DIRECTOR V2")
    print("=" * 72)

    music = load_json(
        MUSIC_LOCK
    )

    grammar = load_json(
        AV_GRAMMAR
    )

    reference_plan = load_json(
        REFERENCE_SHOT_PLAN
    )

    all_clips = load_json(
        ALL_CLIPS
    )

    try:

        source_quality = load_json(
            SOURCE_QUALITY
        )

    except FileNotFoundError:

        source_quality = {}

    # --------------------------------------------------------
    # Music
    # --------------------------------------------------------

    segments = music.get(
        "segments",
        [],
    )

    shot_windows = music.get(
        "shot_windows",
        [],
    )

    if not shot_windows:

        raise RuntimeError(
            "music_locked_edit_plan_v3.json "
            "contains no shot windows."
        )

    effects = get_effects(
        music
    )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    source_scenes = normalize_source_scenes(
        all_clips
    )

    quality_map = build_quality_map(
        source_quality
    )

    reference_matches = (
        normalize_reference_matches(
            reference_plan
        )
    )

    av_events = grammar.get(
        "events",
        [],
    )

    print(
        f"\nShot windows      : "
        f"{len(shot_windows)}"
    )

    print(
        f"Reference AV      : "
        f"{len(av_events)}"
    )

    print(
        f"Source scenes     : "
        f"{len(source_scenes)}"
    )

    print(
        f"Reference matches : "
        f"{len(reference_matches)}"
    )

    # --------------------------------------------------------
    # Source scenes must exist.
    # --------------------------------------------------------

    if not source_scenes:

        raise RuntimeError(
            "No source scenes found in "
            f"{ALL_CLIPS}"
        )

    # --------------------------------------------------------
    # Director state
    # --------------------------------------------------------

    usage_counts: Dict[
        Tuple[str, Any],
        int
    ] = {}

    previous_scene = None

    planned_shots = []

    # ========================================================
    # PLAN EACH SHOT
    # ========================================================

    for shot_index, window in enumerate(
        shot_windows,
        start=1,
    ):

        start = f(
            window.get("start")
        )

        end = f(
            window.get("end")
        )

        duration = max(
            0.05,
            end - start,
        )

        midpoint = (
            start
            + duration / 2.0
        )

        music_segment = music_segment_at(
            midpoint,
            segments,
        )

        music_role = music_segment.get(
            "editorial_role",
            "cinematic_zone",
        )

        local_effects = effects_for_window(
            start,
            end,
            effects,
        )

        sequence = (
            local_effects[0].get(
                "sequence_type"
            )
            if local_effects
            else None
        )

        desired_role = desired_visual_role(
            music_role,
            sequence,
            shot_index,
        )

        reference_match = nearest_reference_match(
            midpoint,
            reference_matches,
        )

        # ----------------------------------------------------
        # Score all source scenes.
        # ----------------------------------------------------

        scored = []

        for scene in source_scenes:

            quality = quality_map.get(
                (
                    scene["video"],
                    scene["scene_id"],
                ),
                0.60,
            )

            if quality < QUALITY_FLOOR:
                continue

            score, diagnostics = candidate_score(
                scene,
                desired_role,
                reference_match,
                quality_map,
                usage_counts,
                previous_scene,
            )

            scored.append({

                "scene": scene,

                "score": score,

                "diagnostics": diagnostics,

            })

        if not scored:

            raise RuntimeError(
                f"No usable source scene "
                f"for shot {shot_index}."
            )

        scored.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        winner = scored[0]

        source = winner["scene"]

        source_key = (
            source["video"],
            source["scene_id"],
        )

        usage_counts[source_key] = (
            usage_counts.get(
                source_key,
                0,
            )
            + 1
        )

        previous_scene = source_key

        # ----------------------------------------------------
        # Source temporal window.
        # ----------------------------------------------------

        source_window = choose_source_window(
            source,
            duration,
        )

        # ----------------------------------------------------
        # Director intent.
        # ----------------------------------------------------

        intent = build_edit_intent(
            shot_index,
            music_segment,
            local_effects,
            source,
        )

        planned_shots.append({

            "shot_id": shot_index,

            "timeline": {

                "start": round(
                    start,
                    4,
                ),

                "end": round(
                    end,
                    4,
                ),

                "duration": round(
                    duration,
                    4,
                ),

                "midpoint": round(
                    midpoint,
                    4,
                ),

            },

            "music": {

                "segment": music_segment.get(
                    "segment"
                ),

                "role": music_role,

                "intensity": f(
                    music_segment.get(
                        "editorial_intensity",
                        0.0,
                    )
                ),

                "sync_type": music_segment.get(
                    "sync_type"
                ),

                "sync_priority": music_segment.get(
                    "sync_priority"
                ),

            },

            "reference": {

                "nearest_match": (
                    reference_match
                    if reference_match
                    else None
                ),

                "active_av_events": local_effects,

            },

            "director": intent,

            "source": {

                "video": source["video"],

                "scene_id": source["scene_id"],

                "original_start": source["start"],

                "original_end": source["end"],

                "source_role": source_visual_role(
                    source
                ),

                "selected_window": source_window,

                "selection_score": round(
                    winner["score"],
                    4,
                ),

                "selection_diagnostics":
                    winner["diagnostics"],

            },

            "alternatives": [

                {

                    "video": item["scene"]["video"],

                    "scene_id":
                        item["scene"]["scene_id"],

                    "score": round(
                        item["score"],
                        4,
                    ),

                    "source_role":
                        source_visual_role(
                            item["scene"]
                        ),

                }

                for item
                in scored[1:4]

            ],

        })

        print(
            f"[{shot_index:02d}/"
            f"{len(shot_windows):02d}] "
            f"{start:6.3f}→{end:6.3f} | "
            f"{music_role:<16} | "
            f"{desired_role:<18} | "
            f"{source['video']}:"
            f"{source['scene_id']} | "
            f"{source_visual_role(source)}"
        )

    # ========================================================
    # AUDIT
    # ========================================================

    total_duration = sum(
        shot["timeline"]["duration"]
        for shot in planned_shots
    )

    scene_usage = {}

    for shot in planned_shots:

        key = (
            shot["source"]["video"],
            shot["source"]["scene_id"],
        )

        scene_usage[key] = (
            scene_usage.get(
                key,
                0,
            )
            + 1
        )

    repeated = {
        f"{video}:{scene_id}": count
        for (
            video,
            scene_id,
        ), count in scene_usage.items()
        if count > 1
    }

    max_repeat = (
        max(
            scene_usage.values()
        )
        if scene_usage
        else 0
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    result = {

        "meta": {

            "version": "v2",

            "purpose": (
                "Creative fusion of music timing, "
                "reference editing grammar and source "
                "footage into a shot-level director plan."
            ),

            "principle": (
                "Reference determines editing language; "
                "music determines timing; source footage "
                "determines visual material; the Director "
                "chooses the creative combination."
            ),

        },

        "summary": {

            "shots": len(
                planned_shots
            ),

            "planned_duration": round(
                total_duration,
                4,
            ),

            "unique_source_scenes": len(
                scene_usage
            ),

            "reused_source_scenes": len(
                repeated
            ),

            "max_scene_reuse": max_repeat,

        },

        "shot_plan": planned_shots,

        "source_usage": [

            {

                "video": video,

                "scene_id": scene_id,

                "usage_count": count,

            }

            for (
                video,
                scene_id,
            ), count

            in sorted(
                scene_usage.items(),
                key=lambda x: x[1],
                reverse=True,
            )

        ],

    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT,
        "w",
    ) as fh:

        json.dump(
            result,
            fh,
            indent=2,
        )

    # ========================================================
    # REPORT
    # ========================================================

    print("\n" + "=" * 72)
    print("✅ CREATIVE DIRECTOR V2 COMPLETE")
    print("=" * 72)

    print(
        f"\nShots planned       : "
        f"{len(planned_shots)}"
    )

    print(
        f"Duration            : "
        f"{total_duration:.3f}s"
    )

    print(
        f"Unique source scenes: "
        f"{len(scene_usage)}"
    )

    print(
        f"Reused scenes       : "
        f"{len(repeated)}"
    )

    print(
        f"Maximum reuse       : "
        f"{max_repeat}"
    )

    print("\nFirst 20 Director decisions:")

    for shot in planned_shots[:20]:

        source = shot["source"]

        print(
            f"  "
            f"{shot['shot_id']:02d} | "
            f"{shot['timeline']['start']:6.3f}"
            f"→"
            f"{shot['timeline']['end']:6.3f} | "
            f"{shot['music']['role']:<16} | "
            f"{shot['director']['visual_role']:<18} | "
            f"{source['video']}:"
            f"{source['scene_id']} | "
            f"{source['selected_window']['mode']}"
        )

    print(
        f"\nOutput: {OUTPUT}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()