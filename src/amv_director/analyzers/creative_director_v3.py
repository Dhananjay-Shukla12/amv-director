from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

MUSIC_PLAN = (
    BASE_DIR
    / "data/outputs/music_director_plan.json"
)

MUSIC_LOCK = (
    BASE_DIR
    / "data/outputs/music_locked_edit_plan_v3.json"
)

AV_GRAMMAR = (
    BASE_DIR
    / "data/outputs/reference_av_grammar_v1.json"
)

REFERENCE_MATCH = (
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
    / "data/outputs/creative_director_plan_v3.json"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_SCENE_REUSE = 3
QUALITY_FLOOR = 0.30
NO_CONSECUTIVE_SAME_SCENE = True

REFERENCE_DISTANCE = 1.25

WEIGHT_MUSIC = 0.30
WEIGHT_REFERENCE = 0.27
WEIGHT_QUALITY = 0.20
WEIGHT_ROLE = 0.15
WEIGHT_VARIATION = 0.08


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


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, value))


def basename(value: Any) -> str | None:

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return Path(text).name


def safe_list(value: Any) -> List[Any]:

    return value if isinstance(value, list) else []


# ============================================================
# MUSIC
# ============================================================

def get_music_segments(
    music_plan: Dict[str, Any],
) -> List[Dict[str, Any]]:

    segments = music_plan.get("segments", [])

    if not isinstance(segments, list):
        return []

    return segments


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
            timestamp - f(x.get("start"))
        ),
    )


# ============================================================
# AV EFFECT EVENTS
# ============================================================

def get_effect_events(
    music_lock: Dict[str, Any],
) -> List[Dict[str, Any]]:

    lane = music_lock.get(
        "effect_lane",
        {},
    )

    return safe_list(
        lane.get("events", [])
    )


def effects_for_window(
    start: float,
    end: float,
    effects: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    result = []

    for effect in effects:

        t = f(
            effect.get("timestamp")
        )

        if (
            start - 0.10
            <= t
            <= end + 0.10
        ):
            result.append(effect)

    return sorted(
        result,
        key=lambda x: f(
            x.get("timestamp")
        ),
    )


# ============================================================
# SOURCE SCENE EXTRACTION
# ============================================================

VIDEO_KEYS = {
    "video",
    "video_name",
    "source_video",
    "filename",
    "file_name",
    "file",
    "path",
    "source",
    "clip",
    "name",
}


SCENE_CONTAINER_KEYS = {
    "scenes",
    "shots",
    "segments",
    "scene_analysis",
    "shot_analysis",
    "results",
}


def find_video_name(
    obj: Dict[str, Any],
) -> str | None:

    for key in VIDEO_KEYS:

        value = obj.get(key)

        if isinstance(value, str):

            candidate = basename(value)

            if candidate:
                return candidate

    return None


def normalize_scene(
    obj: Dict[str, Any],
    inherited_video: str | None,
) -> Dict[str, Any] | None:

    if (
        "scene_id" not in obj
        or "start" not in obj
        or "end" not in obj
    ):
        return None

    video = (
        find_video_name(obj)
        or inherited_video
        or "unknown_source"
    )

    labels = safe_list(
        obj.get("labels", [])
    )

    scores = obj.get(
        "scores",
        {},
    )

    if not isinstance(scores, dict):
        scores = {}

    return {

        "video": video,

        "scene_id": obj.get(
            "scene_id"
        ),

        "start": f(
            obj.get("start")
        ),

        "end": f(
            obj.get("end")
        ),

        "duration": max(
            0.05,
            f(obj.get("end"))
            - f(obj.get("start")),
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

        "labels": [
            str(x)
            for x in labels
        ],

        "scores": scores,

        "raw_role": (
            obj.get("role")
            or obj.get("semantic_role")
            or obj.get("shot_role")
        ),

        "raw_shot_type": (
            obj.get("shot_type")
            or obj.get("dominant_shot_type")
        ),
    }


def extract_source_scenes(
    data: Any,
) -> List[Dict[str, Any]]:

    scenes = []

    def walk(
        obj: Any,
        inherited_video: str | None = None,
    ):

        if isinstance(obj, list):

            for item in obj:

                walk(
                    item,
                    inherited_video,
                )

            return

        if not isinstance(obj, dict):
            return

        local_video = (
            find_video_name(obj)
            or inherited_video
        )

        normalized = normalize_scene(
            obj,
            local_video,
        )

        if normalized is not None:
            scenes.append(
                normalized
            )

        for key, value in obj.items():

            child_video = local_video

            if key.lower() in VIDEO_KEYS:
                candidate = (
                    basename(value)
                    if isinstance(value, str)
                    else None
                )

                if candidate:
                    child_video = candidate

            walk(
                value,
                child_video,
            )

    walk(data)

    unique = {}

    for scene in scenes:

        key = (
            scene["video"],
            scene["scene_id"],
            round(
                scene["start"],
                3,
            ),
            round(
                scene["end"],
                3,
            ),
        )

        unique[key] = scene

    return list(
        unique.values()
    )


# ============================================================
# QUALITY EXTRACTION
# ============================================================

def extract_quality_map(
    data: Any,
) -> Dict[Tuple[str, Any], float]:

    result = {}

    def walk(
        obj: Any,
        inherited_video: str | None = None,
    ):

        if isinstance(obj, list):

            for item in obj:
                walk(
                    item,
                    inherited_video,
                )

            return

        if not isinstance(obj, dict):
            return

        local_video = (
            find_video_name(obj)
            or inherited_video
        )

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
                        obj.get(key)
                    )
                    break

            if quality is not None:

                result[
                    (
                        local_video
                        or "unknown_source",
                        obj["scene_id"],
                    )
                ] = clamp(
                    quality
                )

        for key, value in obj.items():

            walk(
                value,
                local_video,
            )

    walk(data)

    return result


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def minmax_map(
    scenes: List[Dict[str, Any]],
    key: str,
) -> Dict[int, float]:

    values = [
        f(scene.get(key))
        for scene in scenes
    ]

    if not values:
        return {}

    lo = min(values)
    hi = max(values)

    result = {}

    for index, value in enumerate(values):

        if hi - lo < 1e-8:
            result[index] = 0.5
        else:
            result[index] = (
                value - lo
            ) / (
                hi - lo
            )

    return result


def attach_normalized_features(
    scenes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    mappings = {
        key: minmax_map(
            scenes,
            key,
        )

        for key in (
            "motion",
            "brightness",
            "contrast",
            "sharpness",
            "edge_density",
        )
    }

    for index, scene in enumerate(scenes):

        motion = mappings[
            "motion"
        ].get(index, 0.5)

        brightness = mappings[
            "brightness"
        ].get(index, 0.5)

        contrast = mappings[
            "contrast"
        ].get(index, 0.5)

        sharpness = mappings[
            "sharpness"
        ].get(index, 0.5)

        edge = mappings[
            "edge_density"
        ].get(index, 0.5)

        scene["visual_energy"] = clamp(
            0.35 * motion
            + 0.20 * edge
            + 0.20 * contrast
            + 0.15 * sharpness
            + 0.10 * brightness
        )

    return scenes


# ============================================================
# SOURCE SEMANTIC ROLE
# ============================================================

def source_role(
    scene: Dict[str, Any],
) -> str:

    raw_role = scene.get(
        "raw_role"
    )

    if raw_role:

        text = str(
            raw_role
        ).lower()

        if any(
            x in text
            for x in (
                "impact",
                "combat",
                "attack",
                "action",
                "fight",
                "explosion",
            )
        ):
            return "action"

        if any(
            x in text
            for x in (
                "environment",
                "wide",
                "landscape",
                "establishing",
            )
        ):
            return "environment"

        if any(
            x in text
            for x in (
                "close",
                "face",
                "portrait",
            )
        ):
            return "closeup"

    shot_type = str(
        scene.get(
            "raw_shot_type",
            ""
        )
    ).lower()

    labels = " ".join(
        str(x).lower()
        for x in scene.get(
            "labels",
            [],
        )
    )

    text = (
        shot_type
        + " "
        + labels
    )

    if any(
        x in text
        for x in (
            "explosion",
            "impact",
            "attack",
            "combat",
            "fight",
            "punch",
            "action",
        )
    ):
        return "action"

    if any(
        x in text
        for x in (
            "face",
            "closeup",
            "close-up",
            "eyes",
            "portrait",
        )
    ):
        return "closeup"

    if any(
        x in text
        for x in (
            "environment",
            "landscape",
            "establishing",
            "building",
            "city",
        )
    ):
        return "environment"

    if any(
        x in text
        for x in (
            "motion",
            "running",
            "running",
            "movement",
            "dash",
        )
    ):
        return "motion"

    if scene["visual_energy"] >= 0.70:
        return "action"

    if scene["visual_energy"] <= 0.22:
        return "environment"

    return "character"


# ============================================================
# DESIRED DIRECTOR ROLE
# ============================================================

def desired_role(
    music_role: str,
    effects: List[Dict[str, Any]],
) -> str:

    sequence = None

    if effects:

        sequence = effects[0].get(
            "sequence_type"
        )

    if sequence in {
        "BUILD_TO_IMPACT",
        "DIRECT_IMPACT",
    }:
        return "action"

    if sequence == "MOTION_ACCENT":
        return "motion"

    if music_role == "impact_zone":
        return "action"

    if music_role == "action_zone":
        return "action"

    if music_role == "build_zone":
        return "character"

    if music_role == "accent_zone":
        return "closeup"

    if music_role == "cinematic_zone":

        return "cinematic"

    return "character"


# ============================================================
# ROLE COMPATIBILITY
# ============================================================

ROLE_MATRIX = {

    "cinematic": {
        "environment": 1.00,
        "character": 0.82,
        "closeup": 0.76,
        "motion": 0.50,
        "action": 0.35,
    },

    "character": {
        "character": 1.00,
        "closeup": 0.93,
        "motion": 0.70,
        "environment": 0.58,
        "action": 0.55,
    },

    "closeup": {
        "closeup": 1.00,
        "character": 0.92,
        "environment": 0.48,
        "motion": 0.55,
        "action": 0.62,
    },

    "action": {
        "action": 1.00,
        "motion": 0.92,
        "character": 0.72,
        "closeup": 0.64,
        "environment": 0.30,
    },

    "motion": {
        "motion": 1.00,
        "action": 0.92,
        "character": 0.72,
        "closeup": 0.58,
        "environment": 0.42,
    },
}


def role_compatibility(
    desired: str,
    actual: str,
) -> float:

    return ROLE_MATRIX.get(
        desired,
        {},
    ).get(
        actual,
        0.45,
    )


# ============================================================
# REFERENCE MATCH EXTRACTION
# ============================================================

def normalize_reference_matches(
    data: Any,
) -> List[Dict[str, Any]]:

    matches = []

    def walk(obj: Any):

        if isinstance(obj, list):

            for item in obj:
                walk(item)

            return

        if not isinstance(obj, dict):
            return

        source_obj = (
            obj.get("source_scene")
            or obj.get("matched_source")
            or obj.get("source_match")
            or obj.get("match")
            or obj.get("source")
        )

        if not isinstance(
            source_obj,
            dict,
        ):
            source_obj = {}

        video = (
            find_video_name(
                source_obj
            )
            or find_video_name(obj)
        )

        scene_id = (
            source_obj.get(
                "scene_id"
            )
            or source_obj.get(
                "id"
            )
            or obj.get(
                "scene_id"
            )
            or obj.get(
                "source_scene_id"
            )
        )

        reference_obj = (
            obj.get("reference")
            if isinstance(
                obj.get("reference"),
                dict,
            )
            else {}
        )

        reference_time = None

        for candidate in (
            obj.get(
                "reference_timestamp"
            ),
            obj.get(
                "reference_time"
            ),
            obj.get(
                "timestamp"
            ),
            reference_obj.get(
                "timestamp"
            ),
            reference_obj.get(
                "start"
            ),
            obj.get(
                "reference_start"
            ),
        ):

            if candidate is not None:

                reference_time = f(
                    candidate
                )

                break

        score = None

        for candidate in (
            obj.get("score"),
            obj.get("match_score"),
            obj.get("similarity"),
            source_obj.get("score"),
            0.0,
        ):

            if candidate is not None:

                score = f(
                    candidate
                )

                break

        if (
            video is not None
            and scene_id is not None
            and reference_time is not None
        ):

            matches.append({

                "reference_time":
                    reference_time,

                "video":
                    video,

                "scene_id":
                    scene_id,

                "score":
                    clamp(score),

            })

        for value in obj.values():
            walk(value)

    walk(data)

    unique = {}

    for match in matches:

        key = (
            round(
                match["reference_time"],
                3,
            ),
            match["video"],
            match["scene_id"],
        )

        if (
            key not in unique
            or match["score"]
            > unique[key]["score"]
        ):

            unique[key] = match

    return list(
        unique.values()
    )


def nearest_reference(
    timestamp: float,
    matches: List[Dict[str, Any]],
) -> Dict[str, Any] | None:

    if not matches:
        return None

    nearest = min(
        matches,
        key=lambda x: abs(
            timestamp
            - x["reference_time"]
        ),
    )

    if (
        abs(
            timestamp
            - nearest["reference_time"]
        )
        > REFERENCE_DISTANCE
    ):
        return None

    return nearest


# ============================================================
# SOURCE SELECTION SCORE
# ============================================================

def score_source(
    scene: Dict[str, Any],
    desired: str,
    quality_map: Dict[
        Tuple[str, Any],
        float,
    ],
    usage: Dict[
        Tuple[str, Any],
        int,
    ],
    previous_key: Tuple[str, Any] | None,
    reference_match: Dict[str, Any] | None,
) -> Tuple[float, Dict[str, Any]]:

    key = (
        scene["video"],
        scene["scene_id"],
    )

    use_count = usage.get(
        key,
        0,
    )

    # Hard reuse limit.
    if use_count >= MAX_SCENE_REUSE:
        return -999.0, {
            "rejected": "max_reuse"
        }

    quality = quality_map.get(
        key,
        0.60,
    )

    quality = clamp(
        quality
    )

    actual_role = source_role(
        scene
    )

    role_score = role_compatibility(
        desired,
        actual_role,
    )

    # --------------------------------------------------------
    # Reference conditioning
    # --------------------------------------------------------

    reference_score = 0.45

    if reference_match:

        if (
            reference_match["video"]
            == scene["video"]
            and reference_match["scene_id"]
            == scene["scene_id"]
        ):

            reference_score = (
                0.72
                + 0.28
                * reference_match["score"]
            )

        else:

            reference_score = (
                0.35
                + 0.25
                * role_score
            )

    # --------------------------------------------------------
    # Variation
    # --------------------------------------------------------

    variation = 1.0

    if use_count == 1:
        variation -= 0.18

    elif use_count == 2:
        variation -= 0.40

    if (
        previous_key is not None
        and key == previous_key
    ):

        if NO_CONSECUTIVE_SAME_SCENE:
            return -999.0, {
                "rejected":
                    "consecutive_same_scene"
            }

        variation -= 0.80

    variation = clamp(
        variation
    )

    score = (

        WEIGHT_MUSIC
        * role_score

        + WEIGHT_REFERENCE
        * reference_score

        + WEIGHT_QUALITY
        * quality

        + WEIGHT_ROLE
        * role_score

        + WEIGHT_VARIATION
        * variation
    )

    diagnostics = {

        "quality":
            round(
                quality,
                4,
            ),

        "reference_score":
            round(
                reference_score,
                4,
            ),

        "role_score":
            round(
                role_score,
                4,
            ),

        "variation":
            round(
                variation,
                4,
            ),

        "visual_energy":
            round(
                scene["visual_energy"],
                4,
            ),

        "source_role":
            actual_role,

        "reuse_before":
            use_count,

    }

    return score, diagnostics


# ============================================================
# TEMPORAL WINDOW
# ============================================================

def choose_window(
    scene: Dict[str, Any],
    target_duration: float,
) -> Dict[str, Any]:

    start = scene["start"]
    end = scene["end"]

    available = (
        scene["duration"]
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

    # Slightly center-weighted window.
    excess = (
        available
        - target
    )

    start_ratio = 0.35

    selected_start = (
        start
        + excess
        * start_ratio
    )

    selected_end = (
        selected_start
        + target
    )

    return {

        "start": round(
            selected_start,
            4,
        ),

        "end": round(
            selected_end,
            4,
        ),

        "duration": round(
            selected_end
            - selected_start,
            4,
        ),

        "mode":
            "temporal_window",

    }


# ============================================================
# DIRECTOR INTENT
# ============================================================

def build_intent(
    music_segment: Dict[str, Any],
    effects: List[Dict[str, Any]],
    scene: Dict[str, Any],
) -> Dict[str, Any]:

    music_role = music_segment.get(
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

    desired = desired_role(
        music_role,
        effects,
    )

    # --------------------------------------------------------
    # Camera language
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
            "controlled_hold",
            "impact_snap",
            "short_shake",
        ]

    elif sequence == "MOTION_ACCENT":

        camera = [
            "directional_motion",
            "motion_follow",
            "motion_blur",
        ]

    elif music_role == "cinematic_zone":

        camera = [
            "slow_drift",
            "subtle_push",
            "restrained_motion",
        ]

    elif music_role == "build_zone":

        camera = [
            "gradual_push",
            "rising_motion",
        ]

    elif music_role == "impact_zone":

        camera = [
            "compressed_push",
            "impact_emphasis",
        ]

    elif music_role == "action_zone":

        camera = [
            "directional_motion",
            "fast_reframe",
        ]

    else:

        camera = [
            "controlled_hold",
        ]

    # --------------------------------------------------------
    # Editing behavior
    # --------------------------------------------------------

    if sequence in {
        "BUILD_TO_IMPACT",
        "DIRECT_IMPACT",
    }:

        cut_behavior = (
            "impact_resolve"
        )

    elif music_role == "cinematic_zone":

        cut_behavior = (
            "cinematic_resolve"
        )

    else:

        cut_behavior = (
            "music_cut"
        )

    effect_actions = []

    for effect in effects:

        for action in effect.get(
            "edit_sequence",
            [],
        ):

            if action not in effect_actions:
                effect_actions.append(
                    action
                )

    if intensity >= 0.75:

        creative_mode = "high_intensity"

    elif intensity >= 0.45:

        creative_mode = "controlled"

    else:

        creative_mode = "restrained"

    return {

        "visual_role":
            desired,

        "music_role":
            music_role,

        "music_intensity":
            round(
                intensity,
                4,
            ),

        "energy_trend":
            round(
                trend,
                4,
            ),

        "reference_sequence":
            sequence,

        "camera_behavior":
            camera,

        "cut_behavior":
            cut_behavior,

        "effect_actions":
            effect_actions,

        "creative_mode":
            creative_mode,

        "principle":
            (
                "Use the source shot naturally; "
                "apply reference-style motion/effects "
                "only when supported by the timeline."
            ),

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("🎬 CREATIVE DIRECTOR V3")
    print("=" * 72)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    music_plan = load_json(
        MUSIC_PLAN
    )

    music_lock = load_json(
        MUSIC_LOCK
    )

    grammar = load_json(
        AV_GRAMMAR
    )

    reference_data = load_json(
        REFERENCE_MATCH
    )

    all_clips = load_json(
        ALL_CLIPS
    )

    try:
        quality_data = load_json(
            SOURCE_QUALITY
        )
    except FileNotFoundError:
        quality_data = {}

    # --------------------------------------------------------
    # Music
    # --------------------------------------------------------

    segments = get_music_segments(
        music_plan
    )

    shot_windows = music_lock.get(
        "shot_windows",
        [],
    )

    effects = get_effect_events(
        music_lock
    )

    if not segments:

        raise RuntimeError(
            "No music segments found in "
            "music_director_plan.json"
        )

    if not shot_windows:

        raise RuntimeError(
            "No shot windows found in "
            "music_locked_edit_plan_v3.json"
        )

    music_roles = {

        segment.get(
            "editorial_role",
            "UNKNOWN",
        )

        for segment in segments

    }

    if len(music_roles) < 2:

        raise RuntimeError(
            "Music role extraction failed. "
            f"Only found: {music_roles}"
        )

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    source_scenes = extract_source_scenes(
        all_clips
    )

    source_scenes = attach_normalized_features(
        source_scenes
    )

    quality_map = extract_quality_map(
        quality_data
    )

    reference_matches = (
        normalize_reference_matches(
            reference_data
        )
    )

    av_events = grammar.get(
        "events",
        [],
    )

    print(
        f"\nMusic segments      : "
        f"{len(segments)}"
    )

    print(
        f"Music roles         : "
        f"{', '.join(sorted(music_roles))}"
    )

    print(
        f"Shot windows        : "
        f"{len(shot_windows)}"
    )

    print(
        f"AV effect events    : "
        f"{len(effects)}"
    )

    print(
        f"Source scenes       : "
        f"{len(source_scenes)}"
    )

    print(
        f"Reference matches   : "
        f"{len(reference_matches)}"
    )

    # --------------------------------------------------------
    # Detect extraction problems early
    # --------------------------------------------------------

    known_sources = sorted({
        scene["video"]
        for scene in source_scenes
        if scene["video"]
        != "unknown_source"
    })

    unknown_count = sum(
        1
        for scene in source_scenes
        if scene["video"]
        == "unknown_source"
    )

    print(
        f"Known source videos : "
        f"{len(known_sources)}"
    )

    print(
        f"Unknown-source rows : "
        f"{unknown_count}"
    )

    if not known_sources:

        raise RuntimeError(
            "Source video metadata could not "
            "be recovered from all_clips_analysis.json."
        )

    # --------------------------------------------------------
    # Director state
    # --------------------------------------------------------

    usage = {}

    previous_key = None

    planned = []

    # ========================================================
    # SHOT-BY-SHOT DIRECTOR
    # ========================================================

    for index, window in enumerate(
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

        music_segment = (
            music_segment_at(
                midpoint,
                segments,
            )
        )

        music_role = music_segment.get(
            "editorial_role",
            "cinematic_zone",
        )

        local_effects = (
            effects_for_window(
                start,
                end,
                effects,
            )
        )

        desired = desired_role(
            music_role,
            local_effects,
        )

        reference_match = (
            nearest_reference(
                midpoint,
                reference_matches,
            )
        )

        # ----------------------------------------------------
        # Score candidates
        # ----------------------------------------------------

        candidates = []

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

            score, diagnostics = (
                score_source(
                    scene,
                    desired,
                    quality_map,
                    usage,
                    previous_key,
                    reference_match,
                )
            )

            if score <= -900:
                continue

            candidates.append({

                "scene":
                    scene,

                "score":
                    score,

                "diagnostics":
                    diagnostics,

            })

        if not candidates:

            raise RuntimeError(
                f"No usable source scene "
                f"for shot {index}."
            )

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        winner = candidates[0]

        source = winner["scene"]

        source_key = (
            source["video"],
            source["scene_id"],
        )

        usage[source_key] = (
            usage.get(
                source_key,
                0,
            )
            + 1
        )

        previous_key = source_key

        source_window = choose_window(
            source,
            duration,
        )

        intent = build_intent(
            music_segment,
            local_effects,
            source,
        )

        planned.append({

            "shot_id":
                index,

            "timeline": {

                "start":
                    round(
                        start,
                        4,
                    ),

                "end":
                    round(
                        end,
                        4,
                    ),

                "duration":
                    round(
                        duration,
                        4,
                    ),

                "midpoint":
                    round(
                        midpoint,
                        4,
                    ),

            },

            "music": {

                "segment":
                    music_segment.get(
                        "segment"
                    ),

                "role":
                    music_role,

                "intensity":
                    f(
                        music_segment.get(
                            "editorial_intensity",
                            0.0,
                        )
                    ),

                "energy_trend":
                    f(
                        music_segment.get(
                            "energy_trend",
                            0.0,
                        )
                    ),

                "sync_type":
                    music_segment.get(
                        "sync_type"
                    ),

                "sync_priority":
                    music_segment.get(
                        "sync_priority"
                    ),

            },

            "reference": {

                "nearest_match":
                    reference_match,

                "active_av_events":
                    local_effects,

            },

            "director":
                intent,

            "source": {

                "video":
                    source["video"],

                "scene_id":
                    source["scene_id"],

                "source_role":
                    source_role(
                        source
                    ),

                "visual_energy":
                    round(
                        source[
                            "visual_energy"
                        ],
                        4,
                    ),

                "original_start":
                    source["start"],

                "original_end":
                    source["end"],

                "selected_window":
                    source_window,

                "selection_score":
                    round(
                        winner["score"],
                        4,
                    ),

                "selection_diagnostics":
                    winner[
                        "diagnostics"
                    ],

            },

            "alternatives": [

                {

                    "video":
                        item[
                            "scene"
                        ]["video"],

                    "scene_id":
                        item[
                            "scene"
                        ]["scene_id"],

                    "role":
                        source_role(
                            item["scene"]
                        ),

                    "score":
                        round(
                            item["score"],
                            4,
                        ),

                }

                for item
                in candidates[1:4]

            ],

        })

        print(
            f"[{index:02d}/"
            f"{len(shot_windows):02d}] "
            f"{start:6.3f}→"
            f"{end:6.3f} | "
            f"{music_role:<16} | "
            f"{desired:<11} | "
            f"{source['video']}:"
            f"{source['scene_id']} | "
            f"{source_role(source)}"
        )

    # ========================================================
    # AUDIT
    # ========================================================

    role_counts = {}

    for shot in planned:

        role = shot["music"]["role"]

        role_counts[role] = (
            role_counts.get(
                role,
                0,
            )
            + 1
        )

    source_usage = {}

    for shot in planned:

        key = (
            shot["source"]["video"],
            shot["source"]["scene_id"],
        )

        source_usage[key] = (
            source_usage.get(
                key,
                0,
            )
            + 1
        )

    unique_sources = len(
        source_usage
    )

    reused_sources = sum(
        1
        for count
        in source_usage.values()
        if count > 1
    )

    max_reuse = max(
        source_usage.values()
    )

    total_duration = sum(
        shot["timeline"]["duration"]
        for shot in planned
    )

    # --------------------------------------------------------
    # Hard validations
    # --------------------------------------------------------

    if max_reuse > MAX_SCENE_REUSE:

        raise RuntimeError(
            "Scene reuse limit violated: "
            f"{max_reuse} > "
            f"{MAX_SCENE_REUSE}"
        )

    unknown_planned = sum(
        1
        for shot in planned
        if shot["source"]["video"]
        == "unknown_source"
    )

    if unknown_planned > 0:

        raise RuntimeError(
            "Director still produced "
            f"{unknown_planned} unknown-source shots."
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    result = {

        "meta": {

            "version":
                "v3",

            "purpose":
                (
                    "Corrected Creative Director that "
                    "keeps the original music structure, "
                    "music-lock timing, reference AV grammar "
                    "and source-video identity separate."
                ),

            "principles":
                {

                    "music":
                        "controls timing",

                    "reference":
                        "teaches editing language",

                    "source":
                        "provides visual material",

                    "director":
                        "chooses the combination",

                    "effects":
                        "are event-driven, not onset-driven",

                },

        },

        "summary": {

            "shots":
                len(planned),

            "duration":
                round(
                    total_duration,
                    4,
                ),

            "music_roles":
                role_counts,

            "unique_source_scenes":
                unique_sources,

            "reused_source_scenes":
                reused_sources,

            "maximum_scene_reuse":
                max_reuse,

            "reference_matches":
                len(
                    reference_matches
                ),

            "av_effect_events":
                len(effects),

        },

        "shot_plan":
            planned,

        "source_usage":

            [

                {

                    "video":
                        video,

                    "scene_id":
                        scene_id,

                    "usage_count":
                        count,

                }

                for (
                    video,
                    scene_id,
                ), count

                in sorted(
                    source_usage.items(),
                    key=lambda x: (
                        x[1],
                        x[0][0],
                        x[0][1],
                    ),
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
    print("✅ CREATIVE DIRECTOR V3 COMPLETE")
    print("=" * 72)

    print(
        f"\nShots planned        : "
        f"{len(planned)}"
    )

    print(
        f"Duration             : "
        f"{total_duration:.3f}s"
    )

    print(
        f"Unique source scenes : "
        f"{unique_sources}"
    )

    print(
        f"Reused source scenes : "
        f"{reused_sources}"
    )

    print(
        f"Maximum reuse        : "
        f"{max_reuse}"
    )

    print("\nMusic role distribution:")

    for role, count in sorted(
        role_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"  {role:<22}: "
            f"{count}"
        )

    print("\nFirst 20 director decisions:")

    for shot in planned[:20]:

        print(
            f"  "
            f"{shot['shot_id']:02d} | "
            f"{shot['timeline']['start']:6.3f}"
            f"→"
            f"{shot['timeline']['end']:6.3f} | "
            f"{shot['music']['role']:<16} | "
            f"{shot['director']['visual_role']:<11} | "
            f"{shot['source']['video']}:"
            f"{shot['source']['scene_id']}"
        )

    print(
        f"\nOutput: {OUTPUT}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()