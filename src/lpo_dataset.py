"""
LPO Dataset and Data Collator for Listwise Preference Optimization.

This module provides dataset classes for loading ordered preference data
and collating them into batches suitable for LPO training.
"""

import os
import copy
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import transformers

logger = logging.getLogger(__name__)


IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"


def expand2square(pil_img: Image.Image, background_color: tuple) -> Image.Image:
    """Pad a non-square image to square with the given background color."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class LPODataset(Dataset):
    """
    Dataset for Listwise Preference Optimization.

    Each sample consists of one image + R ranked candidate descriptions,
    sorted by emotion intensity from highest to lowest.

    Expected JSON format:
    [
        {
            "id": "emotion_0",
            "image": "path/to/image.jpg",
            "conversations": [
                {"from": "human", "value": "Analyze the emotion intensity..."},
                {"from": "gpt", "value": "Based on facial features, the intensity ranking is: "}
            ],
            "ranked_texts": [
                "Intensely happy expression...",
                "Clearly happy expression...",
                "Slightly happy expression...",
                "Nearly neutral expression..."
            ]
        },
        ...
    ]
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        image_folder: Optional[str] = None,
        image_processor=None,
        n_rank: int = 4,
        max_length: int = 512,
    ):
        """
        Args:
            data_path: Path to the JSON file containing preference data.
            tokenizer: Tokenizer for text processing.
            image_folder: Root folder for images.
            image_processor: Processor for image preprocessing.
            n_rank: Number of ranked candidates per sample.
            max_length: Maximum token length for text sequences.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.n_rank = n_rank
        self.max_length = max_length

        logger.info(f"Loading LPO data from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        if image_folder:
            self._filter_missing_images()
        logger.info(f"Loaded {len(self.data)} samples with {n_rank} candidates each")

    def _filter_missing_images(self):
        """Remove samples whose image files are missing."""
        valid_samples = []
        missing_count = 0
        for item in self.data:
            if "image" not in item:
                valid_samples.append(item)
                continue
            image_path = os.path.join(self.image_folder, item["image"])
            if os.path.exists(image_path):
                valid_samples.append(item)
            else:
                missing_count += 1
        if missing_count > 0:
            logger.warning(f"Filtered out {missing_count} samples with missing images")
        self.data = valid_samples

    def __len__(self) -> int:
        return len(self.data)

    def _load_image(self, image_file: str) -> Optional[torch.Tensor]:
        """Load and preprocess an image."""
        if not self.image_folder or not self.image_processor:
            return None

        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert("RGB")

        # Pad to square if needed
        if hasattr(self.image_processor, "image_mean"):
            background_color = tuple(int(x * 255) for x in self.image_processor.image_mean)
            image = expand2square(image, background_color)

        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image_tensor

    def _tokenize_conversation(self, human_text: str, gpt_text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single conversation turn."""
        # Build the full prompt
        prompt = f"USER: {human_text}\nASSISTANT: {gpt_text}"

        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels: mask the user prompt part
        labels = input_ids.clone()
        # Find where ASSISTANT: starts and mask everything before it
        assistant_token = self.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
        input_ids_list = input_ids.tolist()

        # Simple heuristic: find the last occurrence of assistant tokens
        assistant_start = -1
        for i in range(len(input_ids_list) - len(assistant_token)):
            if input_ids_list[i:i + len(assistant_token)] == assistant_token:
                assistant_start = i + len(assistant_token)

        if assistant_start > 0:
            labels[:assistant_start] = IGNORE_INDEX

        # Mask padding
        labels[attention_mask == 0] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # Load image if available
        image = None
        if "image" in item:
            image = self._load_image(item["image"])

        # Get conversation components
        human_value = item["conversations"][0]["value"]
        gpt_prefix = item["conversations"][1]["value"]
        ranked_texts = item["ranked_texts"]

        # Ensure we have exactly n_rank candidates
        if len(ranked_texts) < self.n_rank:
            ranked_texts = ranked_texts + [ranked_texts[-1]] * (self.n_rank - len(ranked_texts))
        ranked_texts = ranked_texts[:self.n_rank]

        # Tokenize each ranked response
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for k in range(self.n_rank):
            full_response = gpt_prefix + ranked_texts[k]
            tokens = self._tokenize_conversation(human_value, full_response)
            input_ids_list.append(tokens["input_ids"])
            labels_list.append(tokens["labels"])
            attention_mask_list.append(tokens["attention_mask"])

        result = {
            "input_ids": torch.stack(input_ids_list),        # [R, L]
            "labels": torch.stack(labels_list),              # [R, L]
            "attention_mask": torch.stack(attention_mask_list),  # [R, L]
        }

        if image is not None:
            result["image"] = image

        return result


@dataclass
class LPODataCollator:
    """
    Data collator for LPO training.

    Flattens B samples × R candidates into [B*R, L] tensors for efficient
    batched forward passes through the model.
    """
    tokenizer: transformers.PreTrainedTokenizer
    n_rank: int = 4

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        B = len(instances)
        R = self.n_rank

        # Flatten: [B, R, L] -> [B*R, L]
        all_input_ids = []
        all_labels = []
        all_attention_mask = []

        for instance in instances:
            for k in range(R):
                all_input_ids.append(instance["input_ids"][k])
                all_labels.append(instance["labels"][k])
                all_attention_mask.append(instance["attention_mask"][k])

        batch = {
            "input_ids": torch.stack(all_input_ids),           # [B*R, L]
            "labels": torch.stack(all_labels),                 # [B*R, L]
            "attention_mask": torch.stack(all_attention_mask),  # [B*R, L]
            "n_rank": R,
        }

        # Handle images: replicate each image R times
        if "image" in instances[0]:
            images = []
            for instance in instances:
                img = instance["image"]
                for _ in range(R):
                    images.append(img)
            batch["images"] = torch.stack(images)  # [B*R, C, H, W]

        return batch


def make_lpo_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path: str,
    image_folder: Optional[str] = None,
    image_processor=None,
    n_rank: int = 4,
    max_length: int = 512,
    val_split_ratio: float = 0.1,
) -> Dict:
    """
    Create train/eval datasets and collator for LPO training.

    Args:
        tokenizer: Tokenizer for text processing.
        data_path: Path to the full dataset JSON.
        image_folder: Root folder for images.
        image_processor: Processor for image preprocessing.
        n_rank: Number of ranked candidates.
        max_length: Maximum sequence length.
        val_split_ratio: Fraction of data to use for validation.

    Returns:
        Dictionary with train_dataset, eval_dataset, and data_collator.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    if val_split_ratio > 0:
        n_total = len(full_data)
        n_val = int(n_total * val_split_ratio)
        n_train = n_total - n_val
        train_data = full_data[:n_train]
        val_data = full_data[n_train:]

        # Write split files
        train_path = data_path.replace(".json", "_train_split.json")
        val_path = data_path.replace(".json", "_val_split.json")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Split data: {n_train} train, {n_val} val")
    else:
        train_path = data_path
        val_path = None

    train_dataset = LPODataset(
        data_path=train_path,
        tokenizer=tokenizer,
        image_folder=image_folder,
        image_processor=image_processor,
        n_rank=n_rank,
        max_length=max_length,
    )

    eval_dataset = None
    if val_path:
        eval_dataset = LPODataset(
            data_path=val_path,
            tokenizer=tokenizer,
            image_folder=image_folder,
            image_processor=image_processor,
            n_rank=n_rank,
            max_length=max_length,
        )

    data_collator = LPODataCollator(tokenizer=tokenizer, n_rank=n_rank)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
