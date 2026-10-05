import argparse
import json
import math
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

MUSIC_PLAN_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "music_director_plan.json"
)

CLIP_ANALYSIS_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "clip_analysis"
    / "all_clips_analysis.json"
)

GRAMMAR_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "reference_edit_grammar.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "outputs"
    / "creative_director_plan.json"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_QUALITY = 0.30

MAX_SCENE_REUSE = 3

REUSE_PENALTY = 0.12

CONSECUTIVE_SAME_SCENE_PENALTY = 0.50

DARK_SCENE_PENALTY = 0.15

SHORT_SCENE_PENALTY = 0.15

EXACT_WINDOW_REPEAT_PENALTY = 0.50

MIN_CLIP_DURATION = 0.12

EPSILON = 1e-6


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(
            value,
            (list, tuple, np.ndarray)
        ):
            if len(value) == 0:
                return default

            value = value[0]

        return float(value)

    except Exception:
        return default


def clamp(
    value,
    low=0.0,
    high=1.0
):
    return max(
        low,
        min(high, value)
    )


def percentile(
    values,
    p,
    default=0.0
):
    values = [
        safe_float(x)
        for x in values
    ]

    if not values:
        return default

    return float(
        np.percentile(
            values,
            p
        )
    )


def normalize_value(
    value,
    low,
    high
):

    if high <= low:
        return 0.5

    return clamp(
        (
            value - low
        )
        /
        (
            high - low
        )
    )


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# FLATTEN SOURCE SCENES
# ============================================================

def flatten_scenes(data):

    if not isinstance(
        data,
        list
    ):

        raise ValueError(
            "all_clips_analysis.json "
            "must contain a list."
        )

    scenes = []

    for video_data in data:

        video = video_data.get(
            "video",
            "unknown.mp4"
        )

        path = video_data.get(
            "path",
            ""
        )

        video_metrics = video_data.get(
            "video_metrics",
            {}
        )

        raw_scenes = video_data.get(
            "scenes",
            []
        )

        for scene in raw_scenes:

            scene_id = scene.get(
                "scene_id"
            )

            start = safe_float(
                scene.get(
                    "start",
                    0
                )
            )

            end = safe_float(
                scene.get(
                    "end",
                    start
                )
            )

            duration = safe_float(
                scene.get(
                    "duration",
                    end - start
                )
            )

            labels = scene.get(
                "labels",
                []
            )

            if not isinstance(
                labels,
                list
            ):

                labels = []

            scenes.append(
                {
                    "video":
                        video,

                    "path":
                        path,

                    "scene_id":
                        scene_id,

                    "start":
                        start,

                    "end":
                        end,

                    "duration":
                        duration,

                    "brightness":
                        safe_float(
                            scene.get(
                                "brightness",
                                0
                            )
                        ),

                    "contrast":
                        safe_float(
                            scene.get(
                                "contrast",
                                0
                            )
                        ),

                    "sharpness":
                        safe_float(
                            scene.get(
                                "sharpness",
                                0
                            )
                        ),

                    "edge_density":
                        safe_float(
                            scene.get(
                                "edge_density",
                                0
                            )
                        ),

                    "motion":
                        safe_float(
                            scene.get(
                                "motion",
                                0
                            )
                        ),

                    "labels":
                        [
                            str(x).lower()
                            for x in labels
                        ],

                    "source_video_metrics":
                        video_metrics,
                }
            )

    return scenes


# ============================================================
# SOURCE NORMALIZATION
# ============================================================

def build_normalization_stats(
    scenes
):

    return {
        "brightness": {
            "p10": percentile(
                [
                    s["brightness"]
                    for s in scenes
                ],
                10
            ),

            "p90": percentile(
                [
                    s["brightness"]
                    for s in scenes
                ],
                90
            ),
        },

        "contrast": {
            "p10": percentile(
                [
                    s["contrast"]
                    for s in scenes
                ],
                10
            ),

            "p90": percentile(
                [
                    s["contrast"]
                    for s in scenes
                ],
                90
            ),
        },

        "sharpness": {
            "p10": percentile(
                [
                    s["sharpness"]
                    for s in scenes
                ],
                10
            ),

            "p90": percentile(
                [
                    s["sharpness"]
                    for s in scenes
                ],
                90
            ),
        },

        "edge_density": {
            "p10": percentile(
                [
                    s["edge_density"]
                    for s in scenes
                ],
                10
            ),

            "p90": percentile(
                [
                    s["edge_density"]
                    for s in scenes
                ],
                90
            ),
        },

        "motion": {
            "p10": percentile(
                [
                    s["motion"]
                    for s in scenes
                ],
                10
            ),

            "p90": percentile(
                [
                    s["motion"]
                    for s in scenes
                ],
                90
            ),
        },

        "duration": {
            "p10": percentile(
                [
                    s["duration"]
                    for s in scenes
                ],
                10
            ),

            "p90": percentile(
                [
                    s["duration"]
                    for s in scenes
                ],
                90
            ),
        },
    }


# ============================================================
# ADD NORMALIZED VISUAL FEATURES
# ============================================================

def enrich_scene(
    scene,
    stats
):

    motion = normalize_value(
        scene["motion"],
        stats["motion"]["p10"],
        stats["motion"]["p90"],
    )

    brightness = normalize_value(
        scene["brightness"],
        stats["brightness"]["p10"],
        stats["brightness"]["p90"],
    )

    contrast = normalize_value(
        scene["contrast"],
        stats["contrast"]["p10"],
        stats["contrast"]["p90"],
    )

    sharpness = normalize_value(
        scene["sharpness"],
        stats["sharpness"]["p10"],
        stats["sharpness"]["p90"],
    )

    edge_density = normalize_value(
        scene["edge_density"],
        stats["edge_density"]["p10"],
        stats["edge_density"]["p90"],
    )

    duration_norm = normalize_value(
        scene["duration"],
        stats["duration"]["p10"],
        stats["duration"]["p90"],
    )

    labels = set(
        scene["labels"]
    )

    mostly_black = (
        "mostly_black"
        in labels
    )

    very_dark = (
        "very_dark"
        in labels
    )

    soft = (
        "soft_or_blurred"
        in labels
    )

    high_intensity_label = (
        "high_visual_intensity"
        in labels
    )

    medium_motion_label = (
        "medium_motion"
        in labels
    )

    low_motion_label = (
        "low_motion"
        in labels
    )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    quality = (
        0.28 * sharpness
        + 0.18 * contrast
        + 0.18 * edge_density
        + 0.16 * brightness
        + 0.20 * motion
    )

    if mostly_black:
        quality -= 0.50

    if very_dark:
        quality -= 0.20

    if soft:
        quality -= 0.10

    quality = clamp(
        quality
    )

    # --------------------------------------------------------
    # Visual intensity
    # --------------------------------------------------------

    intensity = (
        0.35 * motion
        + 0.22 * contrast
        + 0.18 * sharpness
        + 0.15 * edge_density
        + 0.10 * brightness
    )

    if high_intensity_label:
        intensity += 0.10

    if medium_motion_label:
        intensity += 0.04

    if low_motion_label:
        intensity -= 0.05

    intensity = clamp(
        intensity
    )

    scene["normalized"] = {
        "motion":
            round(
                motion,
                4
            ),

        "brightness":
            round(
                brightness,
                4
            ),

        "contrast":
            round(
                contrast,
                4
            ),

        "sharpness":
            round(
                sharpness,
                4
            ),

        "edge_density":
            round(
                edge_density,
                4
            ),

        "duration":
            round(
                duration_norm,
                4
            ),
    }

    scene["quality_score"] = round(
        quality,
        4
    )

    scene["visual_intensity"] = round(
        intensity,
        4
    )

    return scene


# ============================================================
# EDITORIAL VISUAL PROFILE
# ============================================================

def target_profile(
    segment
):

    role = segment.get(
        "editorial_role",
        "build_zone"
    )

    intensity = safe_float(
        segment.get(
            "editorial_intensity",
            0.5
        )
    )

    effect_budget = segment.get(
        "effect_budget",
        "moderate"
    )

    # --------------------------------------------------------
    # Base profiles
    # --------------------------------------------------------

    if role == "cinematic_zone":

        profile = {
            "motion": 0.25,
            "brightness": 0.45,
            "contrast": 0.55,
            "sharpness": 0.70,
            "edge_density": 0.45,
            "quality": 0.80,
        }

    elif role == "build_zone":

        profile = {
            "motion": 0.45,
            "brightness": 0.50,
            "contrast": 0.60,
            "sharpness": 0.68,
            "edge_density": 0.55,
            "quality": 0.78,
        }

    elif role == "accent_zone":

        profile = {
            "motion": 0.68,
            "brightness": 0.58,
            "contrast": 0.72,
            "sharpness": 0.76,
            "edge_density": 0.68,
            "quality": 0.82,
        }

    elif role == "action_zone":

        profile = {
            "motion": 0.78,
            "brightness": 0.55,
            "contrast": 0.74,
            "sharpness": 0.75,
            "edge_density": 0.72,
            "quality": 0.82,
        }

    elif role == "impact_zone":

        profile = {
            "motion": 0.88,
            "brightness": 0.62,
            "contrast": 0.82,
            "sharpness": 0.82,
            "edge_density": 0.78,
            "quality": 0.86,
        }

    else:

        profile = {
            "motion": 0.50,
            "brightness": 0.50,
            "contrast": 0.60,
            "sharpness": 0.70,
            "edge_density": 0.55,
            "quality": 0.78,
        }

    # --------------------------------------------------------
    # Energy adjustment
    # --------------------------------------------------------

    profile["motion"] = clamp(
        profile["motion"]
        * 0.75
        + intensity * 0.25
    )

    # --------------------------------------------------------
    # Effect budget adjustment
    # --------------------------------------------------------

    if effect_budget == "maximum":

        profile["contrast"] = clamp(
            profile["contrast"]
            + 0.05
        )

        profile["sharpness"] = clamp(
            profile["sharpness"]
            + 0.04
        )

    elif effect_budget == "minimal":

        profile["motion"] *= 0.85

    return profile


# ============================================================
# SCENE COMPATIBILITY
# ============================================================

def duration_fit(
    scene,
    requested_duration
):

    scene_duration = scene[
        "duration"
    ]

    if scene_duration <= 0:
        return 0.0

    # Ideal case: scene can contain the entire window.
    if scene_duration >= requested_duration:

        extra = (
            scene_duration
            - requested_duration
        )

        # Prefer scenes that don't require
        # an excessive crop.
        fit = 1.0 - clamp(
            extra
            / max(
                requested_duration,
                0.5
            ),
            0.0,
            1.0
        ) * 0.25

        return clamp(
            fit
        )

    # For short scenes we allow them, but they
    # are penalized because they may need chaining.
    ratio = (
        scene_duration
        /
        max(
            requested_duration,
            EPSILON
        )
    )

    return clamp(
        ratio
    )


def scene_profile_similarity(
    scene,
    profile
):

    n = scene[
        "normalized"
    ]

    values = [

        abs(
            n["motion"]
            - profile["motion"]
        ),

        abs(
            n["brightness"]
            - profile["brightness"]
        ),

        abs(
            n["contrast"]
            - profile["contrast"]
        ),

        abs(
            n["sharpness"]
            - profile["sharpness"]
        ),

        abs(
            n["edge_density"]
            - profile["edge_density"]
        ),
    ]

    difference = float(
        np.mean(values)
    )

    return clamp(
        1.0 - difference
    )


# ============================================================
# SCENE SCORING
# ============================================================

def score_scene(
    scene,
    segment,
    profile,
    used_counts,
    previous_scene_key,
    used_windows
):

    requested_duration = safe_float(
        segment.get(
            "duration",
            0.5
        )
    )

    quality = scene[
        "quality_score"
    ]

    visual_similarity = (
        scene_profile_similarity(
            scene,
            profile
        )
    )

    fit = duration_fit(
        scene,
        requested_duration
    )

    # Quality should matter a lot.
    score = (
        0.34 * quality
        + 0.34 * visual_similarity
        + 0.20 * fit
        + 0.08 * scene[
            "normalized"
        ]["sharpness"]
    )

    # --------------------------------------------------------
    # Reuse penalty
    # --------------------------------------------------------

    key = (
        scene["video"],
        scene["scene_id"]
    )

    reuse_count = used_counts.get(
        key,
        0
    )

    if reuse_count >= MAX_SCENE_REUSE:

        return -999.0

    score -= (
        REUSE_PENALTY
        * reuse_count
    )

    # --------------------------------------------------------
    # Consecutive same scene
    # --------------------------------------------------------

    if (
        previous_scene_key
        == key
    ):

        score -= (
            CONSECUTIVE_SAME_SCENE_PENALTY
        )

    # --------------------------------------------------------
    # Dark-scene penalty
    # --------------------------------------------------------

    labels = set(
        scene["labels"]
    )

    if "mostly_black" in labels:

        score -= 0.60

    elif "very_dark" in labels:

        score -= DARK_SCENE_PENALTY

    # --------------------------------------------------------
    # Short-scene penalty
    # --------------------------------------------------------

    if (
        scene["duration"]
        + 0.05
        < requested_duration
    ):

        score -= SHORT_SCENE_PENALTY

    # --------------------------------------------------------
    # Exact window reuse
    # --------------------------------------------------------

    window_key = (
        scene["video"],
        scene["scene_id"],
        round(
            scene["start"],
            3
        ),
        round(
            scene["end"],
            3
        )
    )

    if window_key in used_windows:

        score -= (
            EXACT_WINDOW_REPEAT_PENALTY
        )

    return score


# ============================================================
# WINDOW SELECTION
# ============================================================

def choose_window(
    scene,
    requested_duration,
    reuse_count
):

    start = scene[
        "start"
    ]

    end = scene[
        "end"
    ]

    scene_duration = (
        end - start
    )

    requested_duration = max(
        requested_duration,
        MIN_CLIP_DURATION
    )

    # --------------------------------------------------------
    # Scene is long enough
    # --------------------------------------------------------

    if (
        scene_duration
        >= requested_duration
    ):

        available = (
            scene_duration
            - requested_duration
        )

        # Rotate crop position when a scene is reused.
        positions = [
            0.15,
            0.50,
            0.75,
        ]

        position = positions[
            min(
                reuse_count,
                len(positions) - 1
            )
        ]

        source_start = (
            start
            + available
            * position
        )

        source_end = (
            source_start
            + requested_duration
        )

        return (
            float(source_start),
            float(source_end)
        )

    # --------------------------------------------------------
    # Scene is shorter.
    #
    # The caller may use this scene as a partial
    # component of a multi-clip segment.
    # --------------------------------------------------------

    source_start = start
    source_end = end

    return (
        float(source_start),
        float(source_end)
    )


# ============================================================
# BUILD SINGLE SOURCE CLIP
# ============================================================

def make_clip(
    scene,
    segment,
    reuse_count
):

    requested_duration = safe_float(
        segment.get(
            "duration",
            0.5
        )
    )

    source_start, source_end = (
        choose_window(
            scene,
            requested_duration,
            reuse_count
        )
    )

    duration = (
        source_end
        - source_start
    )

    return {
        "source_video":
            scene["video"],

        "source_path":
            scene["path"],

        "scene_id":
            scene["scene_id"],

        "source_start":
            round(
                source_start,
                4
            ),

        "source_end":
            round(
                source_end,
                4
            ),

        "duration":
            round(
                duration,
                4
            ),

        "scene_duration":
            round(
                scene["duration"],
                4
            ),

        "quality_score":
            round(
                scene["quality_score"],
                4
            ),

        "visual_intensity":
            round(
                scene["visual_intensity"],
                4
            ),

        "normalized_features":
            scene["normalized"],

        "labels":
            scene["labels"],
    }


# ============================================================
# BUILD CREATIVE PLAN
# ============================================================

def build_plan(
    music_plan,
    scenes
):

    segments = music_plan.get(
        "segments",
        []
    )

    if not segments:

        raise ValueError(
            "No segments found in music plan."
        )

    used_counts = {}

    used_windows = set()

    previous_scene_key = None

    output_segments = []

    # --------------------------------------------------------
    # Process every editorial segment
    # --------------------------------------------------------

    for segment in segments:

        requested_duration = safe_float(
            segment.get(
                "duration",
                0
            )
        )

        profile = target_profile(
            segment
        )

        candidates = []

        for scene in scenes:

            score = score_scene(
                scene,
                segment,
                profile,
                used_counts,
                previous_scene_key,
                used_windows
            )

            if score <= -900:
                continue

            candidates.append(
                (
                    score,
                    scene
                )
            )

        if not candidates:

            raise RuntimeError(
                "No usable source scene found "
                f"for segment "
                f"{segment.get('segment')}"
            )

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        # ----------------------------------------------------
        # Candidate diversification.
        #
        # Don't blindly select #1 if #2 is very close.
        # This introduces controlled creativity while keeping
        # the editorial intent intact.
        # ----------------------------------------------------

        top_score = candidates[0][0]

        near_best = [
            item
            for item in candidates[:8]
            if item[0]
            >= top_score - 0.035
        ]

        # Deterministic selection:
        # rotate through near-best candidates based on
        # segment index instead of using random().
        segment_index = int(
            segment.get(
                "segment",
                len(output_segments) + 1
            )
        )

        chosen_score, chosen_scene = (
            near_best[
                (
                    segment_index * 7
                )
                % len(near_best)
            ]
        )

        key = (
            chosen_scene["video"],
            chosen_scene["scene_id"]
        )

        reuse_count = used_counts.get(
            key,
            0
        )

        # ----------------------------------------------------
        # If scene is too short for the segment,
        # create a controlled multi-clip segment.
        # ----------------------------------------------------

        clips = []

        if (
            chosen_scene["duration"]
            + 0.05
            >= requested_duration
        ):

            clip = make_clip(
                chosen_scene,
                segment,
                reuse_count
            )

            clips.append(
                clip
            )

        else:

            remaining = (
                requested_duration
            )

            candidate_pool = [
                item[1]
                for item in candidates
            ]

            local_used = set()

            while (
                remaining
                > MIN_CLIP_DURATION
                and candidate_pool
            ):

                best_scene = None
                best_scene_score = -999.0

                for scene in candidate_pool:

                    key2 = (
                        scene["video"],
                        scene["scene_id"]
                    )

                    if key2 in local_used:
                        continue

                    # Prefer scenes that can fill more
                    # of the remaining duration.
                    fill = min(
                        scene["duration"],
                        remaining
                    )

                    duration_score = (
                        fill
                        /
                        max(
                            remaining,
                            EPSILON
                        )
                    )

                    visual_score = (
                        scene_profile_similarity(
                            scene,
                            profile
                        )
                    )

                    candidate_value = (
                        0.65 * duration_score
                        + 0.35 * visual_score
                        + 0.20 * scene[
                            "quality_score"
                        ]
                    )

                    # Avoid immediate repetition.
                    if (
                        previous_scene_key
                        == key2
                    ):
                        candidate_value -= 0.40

                    if candidate_value > best_scene_score:

                        best_scene_score = (
                            candidate_value
                        )

                        best_scene = scene

                if best_scene is None:
                    break

                key2 = (
                    best_scene["video"],
                    best_scene["scene_id"]
                )

                local_used.add(
                    key2
                )

                best_reuse = used_counts.get(
                    key2,
                    0
                )

                take = min(
                    best_scene["duration"],
                    remaining
                )

                temp_segment = dict(
                    segment
                )

                temp_segment[
                    "duration"
                ] = take

                clip = make_clip(
                    best_scene,
                    temp_segment,
                    best_reuse
                )

                clips.append(
                    clip
                )

                remaining -= take

                if take <= MIN_CLIP_DURATION:
                    break

                if (
                    len(clips)
                    >= 5
                ):
                    break

            # Safety
            actual = sum(
                x["duration"]
                for x in clips
            )

            if actual < (
                requested_duration
                - 0.08
            ):

                # Use the original best scene for the
                # remaining time if possible.
                remaining = (
                    requested_duration
                    - actual
                )

                if remaining > 0:

                    temp_segment = dict(
                        segment
                    )

                    temp_segment[
                        "duration"
                    ] = remaining

                    clip = make_clip(
                        chosen_scene,
                        temp_segment,
                        reuse_count
                    )

                    clips.append(
                        clip
                    )

        # ----------------------------------------------------
        # Register usage
        # ----------------------------------------------------

        for clip in clips:

            clip_key = (
                clip["source_video"],
                clip["scene_id"]
            )

            used_counts[
                clip_key
            ] = (
                used_counts.get(
                    clip_key,
                    0
                )
                + 1
            )

            window_key = (
                clip["source_video"],
                clip["scene_id"],
                round(
                    clip["source_start"],
                    3
                ),
                round(
                    clip["source_end"],
                    3
                )
            )

            used_windows.add(
                window_key
            )

            previous_scene_key = (
                clip_key
            )

        # ----------------------------------------------------
        # Segment result
        # ----------------------------------------------------

        actual_duration = sum(
            x["duration"]
            for x in clips
        )

        source_scene_ids = [
            f"{x['source_video']}::"
            f"{x['scene_id']}"
            for x in clips
        ]

        output_segments.append(
            {
                "segment":
                    segment["segment"],

                "start":
                    segment["start"],

                "end":
                    segment["end"],

                "duration":
                    segment["duration"],

                "editorial_role":
                    segment[
                        "editorial_role"
                    ],

                "intensity_label":
                    segment[
                        "intensity_label"
                    ],

                "editorial_intensity":
                    segment[
                        "editorial_intensity"
                    ],

                "music_energy":
                    segment[
                        "music_energy"
                    ],

                "sync_type":
                    segment[
                        "sync_type"
                    ],

                "effect_budget":
                    segment[
                        "effect_budget"
                    ],

                "transition_context":
                    segment.get(
                        "transition_context",
                        "stable"
                    ),

                "creative_freedom":
                    segment.get(
                        "creative_freedom",
                        "moderate"
                    ),

                "visual_target":
                    {
                        key:
                            round(
                                value,
                                4
                            )
                        if isinstance(
                            value,
                            (int, float)
                        )
                        else value
                        for key, value
                        in profile.items()
                    },

                "selection_score":
                    round(
                        float(
                            chosen_score
                        ),
                        4
                    ),

                "actual_source_duration":
                    round(
                        actual_duration,
                        4
                    ),

                "source_scene_ids":
                    source_scene_ids,

                "clips":
                    clips,

                "selection_reason":
                    (
                        "Selected source footage "
                        "to match the editorial role, "
                        "music intensity, visual motion, "
                        "image quality, and reference "
                        "pacing while avoiding excessive "
                        "scene repetition."
                    ),
            }
        )

    return output_segments


# ============================================================
# PLAN SUMMARY
# ============================================================

def build_summary(
    segments
):

    all_clips = []

    for segment in segments:

        all_clips.extend(
            segment.get(
                "clips",
                []
            )
        )

    scene_keys = set(
        (
            clip[
                "source_video"
            ],
            clip[
                "scene_id"
            ]
        )
        for clip in all_clips
    )

    reuse_counts = {}

    for clip in all_clips:

        key = (
            clip[
                "source_video"
            ],
            clip[
                "scene_id"
            ]
        )

        reuse_counts[key] = (
            reuse_counts.get(
                key,
                0
            )
            + 1
        )

    reused = {
        key: count
        for key, count
        in reuse_counts.items()
        if count > 1
    }

    qualities = [
        safe_float(
            clip.get(
                "quality_score",
                0
            )
        )
        for clip in all_clips
    ]

    intensities = [
        safe_float(
            clip.get(
                "visual_intensity",
                0
            )
        )
        for clip in all_clips
    ]

    requested_duration = sum(
        safe_float(
            segment.get(
                "duration",
                0
            )
        )
        for segment in segments
    )

    actual_duration = sum(
        safe_float(
            clip.get(
                "duration",
                0
            )
        )
        for clip in all_clips
    )

    role_counts = {}

    for segment in segments:

        role = segment.get(
            "editorial_role",
            "unknown"
        )

        role_counts[role] = (
            role_counts.get(
                role,
                0
            )
            + 1
        )

    return {

        "segment_count":
            len(segments),

        "clip_count":
            len(all_clips),

        "unique_source_scenes":
            len(scene_keys),

        "reused_source_scenes":
            len(reused),

        "requested_duration":
            round(
                requested_duration,
                4
            ),

        "actual_source_duration":
            round(
                actual_duration,
                4
            ),

        "duration_difference":
            round(
                actual_duration
                - requested_duration,
                4
            ),

        "average_source_quality":
            round(
                float(
                    np.mean(
                        qualities
                    )
                ),
                4
            )
            if qualities
            else 0.0,

        "average_visual_intensity":
            round(
                float(
                    np.mean(
                        intensities
                    )
                ),
                4
            )
            if intensities
            else 0.0,

        "maximum_scene_reuse":
            max(
                reuse_counts.values()
            )
            if reuse_counts
            else 0,

        "editorial_role_distribution":
            role_counts,

        "top_reused_scenes":
            [
                {
                    "video":
                        key[0],

                    "scene_id":
                        key[1],

                    "count":
                        count,
                }

                for key, count
                in sorted(
                    reused.items(),
                    key=lambda x:
                        x[1],
                    reverse=True
                )[:10]
            ],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Creative Director: select source footage "
            "based on music/editorial intent."
        )
    )

    parser.add_argument(
        "--music-plan",
        default=str(
            MUSIC_PLAN_PATH
        )
    )

    parser.add_argument(
        "--clip-analysis",
        default=str(
            CLIP_ANALYSIS_PATH
        )
    )

    parser.add_argument(
        "--grammar",
        default=str(
            GRAMMAR_PATH
        )
    )

    parser.add_argument(
        "--output",
        default=str(
            OUTPUT_PATH
        )
    )

    args = parser.parse_args()

    music_path = Path(
        args.music_plan
    ).expanduser().resolve()

    clips_path = Path(
        args.clip_analysis
    ).expanduser().resolve()

    grammar_path = Path(
        args.grammar
    ).expanduser().resolve()

    output_path = Path(
        args.output
    ).expanduser().resolve()

    print("=" * 82)
    print("🎬 CREATIVE DIRECTOR")
    print("=" * 82)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    music_plan = load_json(
        music_path
    )

    clip_data = load_json(
        clips_path
    )

    # Grammar is currently used as a style reference.
    # We load it now so the architecture remains explicit.
    grammar = load_json(
        grammar_path
    )

    # --------------------------------------------------------
    # Scenes
    # --------------------------------------------------------

    scenes = flatten_scenes(
        clip_data
    )

    if not scenes:

        raise RuntimeError(
            "No source scenes found."
        )

    print()
    print(
        f"Source videos         : "
        f"{len(clip_data)}"
    )

    print(
        f"Source scenes         : "
        f"{len(scenes)}"
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    stats = build_normalization_stats(
        scenes
    )

    scenes = [
        enrich_scene(
            scene,
            stats
        )
        for scene in scenes
    ]

    good_scenes = [
        scene
        for scene in scenes
        if scene[
            "quality_score"
        ] >= MIN_QUALITY
    ]

    print(
        f"Quality-filtered      : "
        f"{len(good_scenes)}"
    )

    # Use the quality-filtered pool where possible.
    if len(good_scenes) >= 10:

        scenes_for_selection = (
            good_scenes
        )

    else:

        scenes_for_selection = scenes

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    segments = build_plan(
        music_plan,
        scenes_for_selection
    )

    summary = build_summary(
        segments
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {

        "meta": {

            "name":
                "creative_director_plan",

            "version":
                "1.0",

            "purpose":
                (
                    "Select source footage that satisfies "
                    "the editorial and musical intent without "
                    "copying the reference footage."
                ),

            "reference_grammar":
                str(
                    grammar_path
                ),

            "music_plan":
                str(
                    music_path
                ),

            "source_analysis":
                str(
                    clips_path
                ),
        },

        "summary":
            summary,

        "source_statistics":
            {
                "scene_count":
                    len(scenes),

                "quality_filtered_count":
                    len(good_scenes),

                "minimum_quality":
                    MIN_QUALITY,
            },

        "selection_philosophy":
            {
                "reference_controls":
                    [
                        "rhythm",
                        "pacing",
                        "intensity",
                        "editorial role",
                    ],

                "source_controls":
                    [
                        "subject",
                        "available motion",
                        "visual composition",
                        "source quality",
                    ],

                "music_controls":
                    [
                        "timing",
                        "accent opportunities",
                        "energy progression",
                    ],

                "creativity_controls":
                    [
                        "controlled candidate variation",
                        "scene reuse avoidance",
                        "window variation",
                        "visual contrast",
                    ],
            },

        "segments":
            segments,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 82)
    print("✅ CREATIVE DIRECTOR PLAN COMPLETE")
    print("=" * 82)

    print(
        f"Editorial segments    : "
        f"{summary['segment_count']}"
    )

    print(
        f"Source clips          : "
        f"{summary['clip_count']}"
    )

    print(
        f"Unique source scenes  : "
        f"{summary['unique_source_scenes']}"
    )

    print(
        f"Reused source scenes  : "
        f"{summary['reused_source_scenes']}"
    )

    print(
        f"Requested duration    : "
        f"{summary['requested_duration']:.3f}s"
    )

    print(
        f"Source duration       : "
        f"{summary['actual_source_duration']:.3f}s"
    )

    print(
        f"Duration difference   : "
        f"{summary['duration_difference']:+.3f}s"
    )

    print(
        f"Average source quality: "
        f"{summary['average_source_quality']:.3f}"
    )

    print(
        f"Average visual energy : "
        f"{summary['average_visual_intensity']:.3f}"
    )

    print()
    print("EDITORIAL DISTRIBUTION")

    for role, count in (
        summary[
            "editorial_role_distribution"
        ]
        .items()
    ):

        print(
            f"  {role:<20}: "
            f"{count}"
        )

    print()
    print("FIRST 20 DIRECTOR DECISIONS")

    for segment in segments[:20]:

        clips = segment.get(
            "clips",
            []
        )

        selected = (
            clips[0]
            if clips
            else None
        )

        if selected:

            source = (
                f"{selected['source_video']} "
                f"S{selected['scene_id']}"
            )

            source_window = (
                f"{selected['source_start']:.3f}"
                f"→"
                f"{selected['source_end']:.3f}s"
            )

        else:

            source = "NONE"
            source_window = "-"

        print(
            f"  #{segment['segment']:02d} "
            f"{segment['start']:.3f}→"
            f"{segment['end']:.3f}s "
            f"{segment['editorial_role']:<15} "
            f"music={segment['music_energy']:.2f} "
            f"→ {source} "
            f"[{source_window}]"
        )

    print()
    print("Saved:")
    print(output_path)
    print("=" * 82)


if __name__ == "__main__":
    main()