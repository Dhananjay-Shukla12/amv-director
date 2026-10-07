from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_grammar() -> dict:
    return {
        "meta": {
            "version": "1.0",
            "purpose": (
                "Describe the reference AMV's typography language "
                "for downstream Lyric/Dialogue Director decisions."
            ),
            "source_reference": "reference_full.mp4",
        },

        "design_principles": [
            "Typography is an editorial visual layer, not generic subtitles.",
            "Text treatment changes according to editorial role and music intensity.",
            "Short words can function as visual accents.",
            "Important words receive disproportionate visual emphasis.",
            "Typography may occupy the foreground, background, or character space.",
            "Text can remain stable for a phrase or become animated graphic material.",
            "Typography should support the composition rather than obscure the subject.",
        ],

        "archetypes": [
            {
                "id": "dialogue_caption",
                "role": "dialogue",
                "scale": "small_to_medium",
                "position": [
                    "lower_center",
                    "lower_third",
                    "subject_adjacent",
                ],
                "alignment": "center_or_left",
                "weight": "bold",
                "italic": True,
                "outline": "strong",
                "shadow": "strong",
                "glow": "low",
                "animation": [
                    "quick_reveal",
                    "stable_hold",
                    "quick_exit",
                ],
                "best_for": [
                    "spoken_dialogue",
                    "short_phrase",
                ],
            },

            {
                "id": "reaction_word",
                "role": "accent",
                "scale": "small",
                "position": [
                    "subject_adjacent",
                    "off_center",
                ],
                "alignment": "center",
                "weight": "bold",
                "outline": "medium",
                "shadow": "medium",
                "glow": "medium",
                "animation": [
                    "fade_in",
                    "micro_scale",
                    "stable_hold",
                    "fade_out",
                ],
                "best_for": [
                    "single_word",
                    "reaction",
                    "micro_accent",
                ],
            },

            {
                "id": "giant_graphic_text",
                "role": "graphic",
                "scale": "very_large",
                "position": [
                    "background",
                    "center",
                    "cropped_edges",
                ],
                "alignment": "center",
                "weight": "heavy",
                "outline": "medium",
                "shadow": "medium",
                "glow": "medium",
                "blur": "possible",
                "rotation": "possible",
                "animation": [
                    "slide",
                    "zoom",
                    "drift",
                    "fade",
                ],
                "best_for": [
                    "introduction",
                    "transition",
                    "section_marker",
                    "dramatic_phrase",
                ],
            },

            {
                "id": "dramatic_statement",
                "role": "statement",
                "scale": "medium_to_large",
                "position": [
                    "center",
                    "subject_adjacent",
                ],
                "alignment": "center",
                "weight": "heavy",
                "outline": "strong",
                "shadow": "strong",
                "glow": "low",
                "animation": [
                    "fast_reveal",
                    "stable_hold",
                    "cut_or_fade",
                ],
                "best_for": [
                    "dramatic_dialogue",
                    "important_phrase",
                ],
            },

            {
                "id": "impact_word",
                "role": "impact",
                "scale": "large",
                "position": [
                    "center",
                    "lower_center",
                    "subject_adjacent",
                ],
                "alignment": "center",
                "weight": "heavy",
                "outline": "strong",
                "shadow": "strong",
                "glow": "low_to_medium",
                "color_strategy": "semantic_or_energy_based",
                "animation": [
                    "impact_in",
                    "micro_scale_pulse",
                    "hold",
                    "hard_exit",
                ],
                "best_for": [
                    "power_words",
                    "chorus_words",
                    "impact_beats",
                    "emphasis_words",
                ],
            },

            {
                "id": "outro_text",
                "role": "outro",
                "scale": "medium",
                "position": [
                    "center",
                    "lower_center",
                ],
                "alignment": "center",
                "weight": "bold",
                "outline": "medium",
                "shadow": "medium",
                "glow": "low",
                "animation": [
                    "fade_in",
                    "stable_hold",
                    "fade_out",
                ],
                "best_for": [
                    "outro",
                    "call_to_action",
                ],
            },
        ],

        "reference_observations": [
            {
                "time_range": [4.0, 6.0],
                "archetype": "dialogue_caption",
                "observed_behavior": (
                    "Short phrases appear in the lower portion of the frame "
                    "with strong white typography and pronounced shadow/outline."
                ),
            },
            {
                "time_range": [7.5, 9.0],
                "archetype": "reaction_word",
                "observed_behavior": (
                    "A compact reaction word appears beside the subject "
                    "and remains visually separated from the main subject."
                ),
            },
            {
                "time_range": [12.0, 12.8],
                "archetype": "giant_graphic_text",
                "observed_behavior": (
                    "Oversized background typography is blurred and becomes "
                    "part of the transition/composition."
                ),
            },
            {
                "time_range": [15.2, 16.6],
                "archetype": "dramatic_statement",
                "observed_behavior": (
                    "A centered dramatic phrase is integrated into a close-up "
                    "rather than behaving like a conventional subtitle."
                ),
            },
            {
                "time_range": [17.8, 20.4],
                "archetype": "impact_word",
                "observed_behavior": (
                    "A dramatic statement contains visually emphasized words, "
                    "with stronger color and scale treatment for the key word."
                ),
            },
            {
                "time_range": [39.8, 42.8],
                "archetype": "impact_word",
                "observed_behavior": (
                    "Sequential emphasis words are displayed over high-energy "
                    "visuals, with strong color contrast and large typography."
                ),
            },
            {
                "time_range": [55.6, 57.4],
                "archetype": "outro_text",
                "observed_behavior": (
                    "Outro text remains visible across multiple frames "
                    "while the background settles."
                ),
            },
        ],

        "selection_rules": {
            "short_phrase": [
                "reaction_word",
                "dialogue_caption",
            ],
            "dramatic_phrase": [
                "dramatic_statement",
                "impact_word",
            ],
            "single_emphasis_word": [
                "impact_word",
                "reaction_word",
            ],
            "section_transition": [
                "giant_graphic_text",
            ],
            "outro": [
                "outro_text",
            ],
        },

        "future_director_inputs": [
            "lyric_timestamp",
            "phrase_length",
            "word_importance",
            "music_energy",
            "beat_strength",
            "shot_role",
            "subject_position",
            "available_negative_space",
            "reference_typography_style",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Reference Typography Grammar V1."
    )

    parser.add_argument(
        "--output",
        default="data/outputs/reference_typography_grammar_v1.json",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grammar = build_grammar()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            grammar,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("===================================")
    print("REFERENCE TYPOGRAPHY GRAMMAR V1")
    print("===================================")
    print(f"Archetypes: {len(grammar['archetypes'])}")
    print(
        f"Observations: "
        f"{len(grammar['reference_observations'])}"
    )
    print(f"Output: {output_path}")
    print("✅ Grammar generated")


if __name__ == "__main__":
    main()