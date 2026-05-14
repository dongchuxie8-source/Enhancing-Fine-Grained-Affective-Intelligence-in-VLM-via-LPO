"""
Build ordered preference datasets from AffectNet annotations.

This script constructs ranked description lists for each facial expression image
based on continuous valence annotations, enabling LPO training.

Usage:
    python data/build_preference.py --input_csv data/affectnet_annotations.csv \
                                     --output_dir data/ \
                                     --num_samples 1000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Emotion intensity description templates
INTENSITY_TEMPLATES = {
    "Happy": {
        "high": "This image shows an intensely joyful expression with a radiant smile and sparkling eyes.",
        "medium": "This image shows a clearly happy expression with a visible smile.",
        "low": "This image shows a slightly happy expression with upturned lips.",
        "neutral": "This image shows a nearly neutral expression with mild pleasantness.",
    },
    "Sad": {
        "high": "This image shows an intensely sad expression with furrowed brows and a downturned mouth.",
        "medium": "This image shows a clearly sad expression with a low mood.",
        "low": "This image shows a slightly sad expression with mild melancholy.",
        "neutral": "This image shows a nearly neutral expression with a hint of sadness.",
    },
    "Angry": {
        "high": "This image shows an intensely angry expression with tight lips and narrowed eyes.",
        "medium": "This image shows a clearly angry expression with visible tension.",
        "low": "This image shows a slightly angry expression with mild irritation.",
        "neutral": "This image shows a nearly neutral expression with a hint of annoyance.",
    },
    "Fear": {
        "high": "This image shows an intensely fearful expression with wide eyes and an open mouth.",
        "medium": "This image shows a clearly fearful expression with visible anxiety.",
        "low": "This image shows a slightly fearful expression with mild unease.",
        "neutral": "This image shows a nearly neutral expression with a hint of concern.",
    },
    "Surprise": {
        "high": "This image shows an intensely surprised expression with raised eyebrows and a wide-open mouth.",
        "medium": "This image shows a clearly surprised expression with raised eyebrows.",
        "low": "This image shows a slightly surprised expression with mild astonishment.",
        "neutral": "This image shows a nearly neutral expression with a hint of surprise.",
    },
    "Disgust": {
        "high": "This image shows an intensely disgusted expression with a wrinkled nose and curled lip.",
        "medium": "This image shows a clearly disgusted expression with visible aversion.",
        "low": "This image shows a slightly disgusted expression with mild distaste.",
        "neutral": "This image shows a nearly neutral expression with a hint of discomfort.",
    },
    "Contempt": {
        "high": "This image shows an intensely contemptuous expression with a pronounced sneer.",
        "medium": "This image shows a clearly contemptuous expression with a slight smirk.",
        "low": "This image shows a slightly contemptuous expression with mild disdain.",
        "neutral": "This image shows a nearly neutral expression with a hint of superiority.",
    },
    "Neutral": {
        "high": "This image shows a completely neutral expression with no emotional cues.",
        "medium": "This image shows a mostly neutral expression with very subtle cues.",
        "low": "This image shows a nearly neutral expression with slight emotional hints.",
        "neutral": "This image shows a neutral expression with ambiguous emotional signals.",
    },
}


def get_intensity_level(valence: float) -> str:
    """Map continuous valence to discrete intensity level."""
    abs_v = abs(valence)
    if abs_v > 0.8:
        return "high"
    elif abs_v > 0.5:
        return "medium"
    elif abs_v > 0.2:
        return "low"
    else:
        return "neutral"


def build_ranked_texts(emotion: str, valence: float) -> list:
    """
    Build ordered description list based on emotion and valence.

    The first description best matches the true intensity level,
    followed by progressively less accurate descriptions.
    """
    if emotion not in INTENSITY_TEMPLATES:
        emotion = "Neutral"

    templates = INTENSITY_TEMPLATES[emotion]
    level = get_intensity_level(valence)

    # Order: best match first, then decreasing relevance
    level_order = {
        "high": ["high", "medium", "low", "neutral"],
        "medium": ["medium", "high", "low", "neutral"],
        "low": ["low", "medium", "neutral", "high"],
        "neutral": ["neutral", "low", "medium", "high"],
    }

    return [templates[l] for l in level_order[level]]


def build_dataset(csv_path: str, num_samples: int = 1000, seed: int = 42):
    """Build the full preference dataset from annotations CSV."""
    print(f"Reading annotations from {csv_path}")
    df = pd.read_csv(csv_path)

    # Sample evenly across emotions
    emotions = df["emotion_name"].unique()
    samples_per_emotion = num_samples // len(emotions)

    preference_data = []
    for emotion in emotions:
        emotion_df = df[df["emotion_name"] == emotion]
        n_sample = min(samples_per_emotion, len(emotion_df))
        samples = emotion_df.sample(n=n_sample, random_state=seed)

        for idx, row in samples.iterrows():
            ranked_texts = build_ranked_texts(emotion, row["valence"])
            preference_data.append({
                "id": f"emotion_{len(preference_data)}",
                "image": row.get("image_path", f"images/{idx}.jpg"),
                "conversations": [
                    {"from": "human", "value": "Carefully observe this image and analyze the fine-grained emotion intensity."},
                    {"from": "gpt", "value": "Based on the facial muscle features, the emotion intensity ranking is as follows: "},
                ],
                "ranked_texts": ranked_texts,
            })

    print(f"Built {len(preference_data)} preference samples")
    return preference_data


def main():
    parser = argparse.ArgumentParser(description="Build LPO preference dataset")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to annotations CSV")
    parser.add_argument("--output_dir", type=str, default="data/", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Total number of samples")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    data = build_dataset(args.input_csv, args.num_samples, args.seed)

    # Split
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=args.seed)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=args.seed)

    # Save
    splits = {"train_lpo.json": train_data, "val_lpo.json": val_data, "test_lpo.json": test_data}
    for filename, split_data in splits.items():
        path = output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {filename}: {len(split_data)} samples")

    print("\nDone! Dataset construction complete.")


if __name__ == "__main__":
    main()
