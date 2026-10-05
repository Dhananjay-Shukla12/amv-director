from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

ANALYSIS_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "clip_analysis"
    / "all_clips_analysis.json"
)


# ---------------------------------------------------------
# Editing-role profiles
# ---------------------------------------------------------
#
# Values describe what the role should generally prefer.
#
# duration:
#   preferred minimum / maximum in seconds
#
# motion:
#   preferred normalized motion
#
# contrast:
#   preferred normalized contrast
#
# sharpness:
#   preferred normalized sharpness
#
# brightness:
#   optional preference
#
# This is intentionally deterministic.
# Later the AI Director will learn these preferences
# from the reference Edit DNA.
# ---------------------------------------------------------

ROLE_PROFILES = {
    "impact": {
        "duration": (0.35, 1.50),
        "motion": (0.70, 1.00),
        "contrast": (0.60, 1.00),
        "sharpness": (0.55, 1.00),
        "brightness": (0.20, 0.90),
    },

    "action": {
        "duration": (0.50, 2.50),
        "motion": (0.65, 1.00),
        "contrast": (0.45, 1.00),
        "sharpness": (0.45, 1.00),
        "brightness": (0.15, 0.95),
    },

    "cinematic": {
        "duration": (2.00, 8.00),
        "motion": (0.15, 0.70),
        "contrast": (0.45, 1.00),
        "sharpness": (0.50, 1.00),
        "brightness": (0.15, 0.85),
    },

    "calm": {
        "duration": (2.50, 12.00),
        "motion": (0.00, 0.35),
        "contrast": (0.20, 0.80),
        "sharpness": (0.35, 1.00),
        "brightness": (0.15, 0.80),
    },

    "transition": {
        "duration": (0.30, 1.20),
        "motion": (0.35, 0.85),
        "contrast": (0.35, 0.90),
        "sharpness": (0.25, 0.90),
        "brightness": (0.10, 0.90),
    },
}


def load_scenes():
    if not ANALYSIS_FILE.exists():
        raise FileNotFoundError(
            f"Analysis file not found:\n{ANALYSIS_FILE}\n\n"
            "Run clip_analyzer.py first."
        )

    with ANALYSIS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = []

    for video_result in data:
        video_name = video_result["video"]

        for scene in video_result["scenes"]:
            scene_copy = dict(scene)
            scene_copy["video"] = video_name
            scenes.append(scene_copy)

    return scenes


def get_score(scene, name):
    return float(scene.get("scores", {}).get(name, 0.0))


def range_score(value, minimum, maximum):
    """
    Returns a score describing how well a value fits inside
    the preferred range.

    1.0 = comfortably inside the preferred range
    0.0 = far outside
    """

    if minimum <= value <= maximum:
        return 1.0

    if value < minimum:
        distance = minimum - value
        scale = max(minimum, 0.25)
    else:
        distance = value - maximum
        scale = max(maximum, 0.50)

    score = 1.0 - (distance / scale)

    return max(0.0, min(1.0, score))


def duration_fit(duration, preferred_range):
    minimum, maximum = preferred_range

    # Strong preference around the middle of the intended range.
    center = (minimum + maximum) / 2.0
    half_range = max((maximum - minimum) / 2.0, 0.25)

    distance = abs(duration - center)

    score = 1.0 - (distance / half_range)

    return max(0.0, min(1.0, score))


def score_scene(scene, role):
    profile = ROLE_PROFILES[role]

    duration = float(scene.get("duration", 0.0))

    motion = get_score(scene, "motion")
    contrast = get_score(scene, "contrast")
    sharpness = get_score(scene, "sharpness")
    brightness = get_score(scene, "brightness")

    scores = {
        "duration": duration_fit(
            duration,
            profile["duration"],
        ),
        "motion": range_score(
            motion,
            *profile["motion"],
        ),
        "contrast": range_score(
            contrast,
            *profile["contrast"],
        ),
        "sharpness": range_score(
            sharpness,
            *profile["sharpness"],
        ),
        "brightness": range_score(
            brightness,
            *profile["brightness"],
        ),
    }

    # Role-specific importance.
    if role == "impact":
        total = (
            scores["duration"] * 0.20
            + scores["motion"] * 0.40
            + scores["contrast"] * 0.20
            + scores["sharpness"] * 0.15
            + scores["brightness"] * 0.05
        )

    elif role == "action":
        total = (
            scores["duration"] * 0.15
            + scores["motion"] * 0.50
            + scores["contrast"] * 0.15
            + scores["sharpness"] * 0.15
            + scores["brightness"] * 0.05
        )

    elif role == "cinematic":
        total = (
            scores["duration"] * 0.30
            + scores["motion"] * 0.10
            + scores["contrast"] * 0.20
            + scores["sharpness"] * 0.25
            + scores["brightness"] * 0.15
        )

    elif role == "calm":
        total = (
            scores["duration"] * 0.40
            + scores["motion"] * 0.05
            + scores["contrast"] * 0.15
            + scores["sharpness"] * 0.20
            + scores["brightness"] * 0.20
        )

    else:  # transition
        total = (
            scores["duration"] * 0.30
            + scores["motion"] * 0.25
            + scores["contrast"] * 0.15
            + scores["sharpness"] * 0.10
            + scores["brightness"] * 0.20
        )

    return total, scores


# def find_candidates(
#     scenes,
#     role,
#     top_k=5,
# ):
#     candidates = []

#     for scene in scenes:

#         total, component_scores = score_scene(
#             scene,
#             role,
#         )

#         result = dict(scene)

#         result["retrieval_score"] = round(total, 4)

#         result["fit"] = {
#             key: round(value, 3)
#             for key, value in component_scores.items()
#         }

#         candidates.append(result)

#     candidates.sort(
#         key=lambda item: item["retrieval_score"],
#         reverse=True,
#     )

#     return candidates[:top_k]

def find_candidates(
    scenes,
    role,
    top_k=5,
):
    candidates = []

    profile = ROLE_PROFILES[role]
    min_duration, max_duration = profile["duration"]

    # Allow a small tolerance because detected scene boundaries
    # are not always perfectly aligned.
    tolerance = 0.15

    for scene in scenes:

        duration = float(scene.get("duration", 0.0))

        if duration < (min_duration - tolerance):
            continue

        if duration > (max_duration + tolerance):
            continue

        total, component_scores = score_scene(
            scene,
            role,
        )

        result = dict(scene)

        result["retrieval_score"] = round(total, 4)

        result["fit"] = {
            key: round(value, 3)
            for key, value in component_scores.items()
        }

        candidates.append(result)

    candidates.sort(
        key=lambda item: item["retrieval_score"],
        reverse=True,
    )

    return candidates[:top_k]

def print_candidates(candidates):
    print()
    print("=" * 76)
    print("🎬 TOP CLIP CANDIDATES")
    print("=" * 76)

    for index, scene in enumerate(candidates, start=1):

        print()
        print(f"#{index}")

        print(f"Video       : {scene['video']}")

        print(
            f"Scene       : {scene['scene_id']} "
            f"({scene['start']:.3f}s → {scene['end']:.3f}s)"
        )

        print(
            f"Duration    : {scene['duration']:.3f}s"
        )

        print(
            f"Retrieval   : {scene['retrieval_score']:.3f}"
        )

        raw = scene.get("scores", {})

        print(
            f"Raw scores  : "
            f"M={raw.get('motion', 0):.3f} "
            f"C={raw.get('contrast', 0):.3f} "
            f"S={raw.get('sharpness', 0):.3f}"
        )

        fit = scene.get("fit", {})

        print(
            f"Role fit    : "
            f"D={fit.get('duration', 0):.3f} "
            f"M={fit.get('motion', 0):.3f} "
            f"C={fit.get('contrast', 0):.3f} "
            f"S={fit.get('sharpness', 0):.3f}"
        )

        labels = scene.get("labels", [])

        print(
            f"Labels      : {', '.join(labels)}"
        )

    print()
    print("=" * 76)


def main():

    role = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "impact"
    )

    if role not in ROLE_PROFILES:
        print(
            f"❌ Unknown role: {role}\n"
            f"Available: {', '.join(ROLE_PROFILES)}"
        )
        sys.exit(1)

    try:
        scenes = load_scenes()

        candidates = find_candidates(
            scenes,
            role=role,
            top_k=5,
        )

        print(f"Role        : {role}")
        print(f"Scenes      : {len(scenes)}")

        print_candidates(candidates)

    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()