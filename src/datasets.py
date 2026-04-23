"""Dataset loading and preprocessing for spatial reasoning probing study."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image


@dataclass
class VSRExample:
    """Single VSR example with image, caption, and spatial relation label."""
    image_id: str
    image: Optional[Image.Image]  # PIL Image or None if load fails
    caption: str
    relation: str
    label: bool  # True = caption correctly describes image, False = negative


@dataclass
class SpartQAExample:
    """Single SpartQA example with context, question, and answer."""
    context: str
    question: str
    answer: str  # "Yes", "No", or "Unknown"


class VSRDataset:
    """
    Visual Spatial Reasoning dataset loader.
    
    Paper: Liu et al. (2022) "Visual Spatial Reasoning"
    https://arxiv.org/abs/2205.00363
    
    Format: (image, caption, relation, label)
    Relations: 66 spatial relation types (on, under, above, below, left of, etc.)
    Size: ~10k examples
    """

    def __init__(self, split: str = "train", variant: str = "random"):
        """
        Load VSR dataset from HuggingFace.

        Args:
            split: "train" or "test"
            variant: "random" (standard split) or "zeroshot" (test generalization)
        """
        dataset_id = f"cambridgeltl/vsr_{variant}"
        print(f"Loading {dataset_id} split={split}...")
        self.hf_dataset = load_dataset(dataset_id, split=split)
        print(f"Loaded {len(self.hf_dataset)} examples")
        
        self.split = split
        self.variant = variant
        self._failed_image_indices = []

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> VSRExample:
        """Load example by index, handling image load failures."""
        row = self.hf_dataset[idx]
        image = None
        
        try:
            if isinstance(row["image"], Image.Image):
                image = row["image"]
            else:
                # Might be PIL Image data
                image = row["image"]
        except Exception as e:
            self._failed_image_indices.append(idx)
            print(f"Warning: failed to load image at idx {idx}: {e}")

        return VSRExample(
            image_id=row.get("image_id", str(idx)),
            image=image,
            caption=row["caption"],
            relation=row["relation"],
            label=row["label"],
        )

    def get_texts(self) -> list[str]:
        """Extract all captions as text."""
        return [row["caption"] for row in self.hf_dataset]

    def get_images(self) -> list[Optional[Image.Image]]:
        """Extract all images, with None for failed loads."""
        images = []
        for idx in range(len(self.hf_dataset)):
            try:
                row = self.hf_dataset[idx]
                images.append(row["image"])
            except Exception:
                images.append(None)
        return images

    def get_relations(self) -> np.ndarray:
        """Get relation type label for each example (string array)."""
        return np.array([row["relation"] for row in self.hf_dataset])

    def get_binary_labels(self) -> np.ndarray:
        """Get True/False labels (1/0) for each example."""
        return np.array([int(row["label"]) for row in self.hf_dataset])

    def get_unique_relations(self) -> list[str]:
        """Get sorted list of unique relation types in this split."""
        relations = set(row["relation"] for row in self.hf_dataset)
        return sorted(list(relations))

    def relation_counts(self) -> dict[str, int]:
        """Count examples per relation type."""
        counts = {}
        for row in self.hf_dataset:
            rel = row["relation"]
            counts[rel] = counts.get(rel, 0) + 1
        return counts

    def label_balance(self) -> dict[str, Tuple[int, int]]:
        """Get (true_count, false_count) per relation type."""
        balance = {}
        for row in self.hf_dataset:
            rel = row["relation"]
            label = row["label"]
            if rel not in balance:
                balance[rel] = [0, 0]
            balance[rel][int(label)] += 1
        return {rel: tuple(counts) for rel, counts in balance.items()}


class SpartQADataset:
    """
    SPARTQA: Textual Question Answering Benchmark for Spatial Reasoning.
    
    Paper: Mirzaee et al. (2021)
    https://arxiv.org/abs/2104.05832
    
    Format: (context, question, answer: Yes/No/Unknown)
    Multi-hop spatial reasoning in text.
    """

    def __init__(self):
        """Load SpartQA — requires manual download from GitHub."""
        print("SpartQA: Check HuggingFace or download from https://github.com/hlr/SPARTQA")
        self.hf_dataset = None
        
        # Try HuggingFace first
        try:
            self.hf_dataset = load_dataset("hlr/spartqa")
            print("Loaded SpartQA from HuggingFace")
        except Exception as e:
            print(f"SpartQA not found on HuggingFace: {e}")
            print("Manual download may be required.")

    def __len__(self) -> int:
        if self.hf_dataset is None:
            return 0
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> SpartQAExample:
        if self.hf_dataset is None:
            raise RuntimeError("SpartQA not loaded")
        
        row = self.hf_dataset[idx]
        return SpartQAExample(
            context=row.get("context", ""),
            question=row.get("question", ""),
            answer=row.get("answer", ""),
        )


class StepGameDataset:
    """
    StepGame: A New Benchmark for Robust Multi-Hop Spatial Reasoning in Texts.
    
    Paper: Shi et al. (2022)
    https://arxiv.org/abs/2204.08292
    
    Format: (story, question, answer) with hop count metadata
    Used to analyze accuracy vs. reasoning complexity.
    """

    def __init__(self):
        """Load StepGame — requires manual download from GitHub."""
        print("StepGame: Download from https://github.com/ZhengxiangShi/StepGame")
        self.hf_dataset = None
        
        # Try HuggingFace
        try:
            self.hf_dataset = load_dataset("stepgame")
            print("Loaded StepGame from HuggingFace")
        except Exception as e:
            print(f"StepGame not found on HuggingFace: {e}")
            print("Manual download may be required.")

    def __len__(self) -> int:
        if self.hf_dataset is None:
            return 0
        return len(self.hf_dataset)


def load_vsr(split: str = "train", variant: str = "random") -> VSRDataset:
    """
    Convenience function to load VSR.

    Args:
        split: "train" or "test"
        variant: "random" (standard) or "zeroshot" (test on unseen relations)

    Returns:
        VSRDataset instance
    """
    return VSRDataset(split=split, variant=variant)


def load_spartqa() -> SpartQADataset:
    """Load SpartQA if available."""
    return SpartQADataset()


def load_stepgame() -> StepGameDataset:
    """Load StepGame if available."""
    return StepGameDataset()
