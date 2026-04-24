"""Embedding extraction from text models with caching and L2 normalization."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import os

import numpy as np
import torch
from tqdm import tqdm


def get_device() -> str:
    """Detect available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_text(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of text strings.

        Args:
            texts: List of text strings
            batch_size: Batch size for inference

        Returns:
            (N, dim) array of L2-normalized embeddings
        """
        raise NotImplementedError

    def embed_image_text(
        self, images: list, texts: list[str], batch_size: int = 32
    ) -> np.ndarray:
        """
        Embed image-text pairs jointly. Only implemented for multimodal models.

        Args:
            images: List of PIL Images
            texts: List of text strings
            batch_size: Batch size

        Returns:
            (N, dim) array of L2-normalized embeddings
        """
        raise NotImplementedError("embed_image_text not implemented for this model")


class SBERTEmbedder(BaseEmbedder):
    """
    Sentence-BERT embedder using sentence-transformers library.
    
    Model: sentence-transformers/all-mpnet-base-v2
    Reference: Reimers & Gurevych (2019)
    https://arxiv.org/abs/1908.10084
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize SBERT model.

        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading SBERT model: {model_name}")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=get_device())
        self.model_name = model_name
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed_text(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed texts using SBERT mean pooling."""
        print(f"Embedding {len(texts)} texts with SBERT (batch_size={batch_size})...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalize
        )
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(texts), self.model.get_sentence_embedding_dimension())
        return embeddings


class CLIPTextEmbedder(BaseEmbedder):
    """
    CLIP text encoder (text-only).
    
    Model: openai/clip-vit-base-patch32 (text encoder)
    Reference: Radford et al. (2021)
    https://arxiv.org/abs/2103.00020
    
    Note: Text encoder extracts pooled [EOS] token embedding (512d).
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP text encoder.

        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading CLIP text encoder: {model_name}")
        from transformers import CLIPTokenizer, CLIPTextModel

        self.device = get_device()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_name = model_name
        print(f"Model loaded. Embedding dimension: 512")

    def embed_text(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed texts using CLIP text encoder."""
        print(f"Embedding {len(texts)} texts with CLIP text encoder (batch_size={batch_size})...")
        
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i : i + batch_size]
                
                # Tokenize (max 77 tokens — CLIP limitation)
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # Get pooled embeddings (cls token)
                batch_emb = outputs.pooler_output.cpu().numpy()  # (batch_size, 512)
                
                # L2 normalize
                batch_emb = batch_emb / np.linalg.norm(batch_emb, axis=1, keepdims=True)
                
                embeddings.append(batch_emb)
        
        result = np.vstack(embeddings)
        assert result.shape == (len(texts), 512)
        return result


class CLIPMultimodalEmbedder(BaseEmbedder):
    """
    CLIP with image and text joint embedding.

    Model: openai/clip-vit-base-patch32 (full model)

    Supports three extraction modes:
      embed_image()      — image encoder only (512d) — tests visual spatial encoding directly
      embed_text()       — text encoder only (512d) — same as CLIPTextEmbedder
      embed_image_text() — concat fusion (1024d)    — probe accesses both modalities independently

    Averaging was the original default but is methodologically unsound for probing:
    it can cancel signal that lives in opposite directions across modalities.
    Concat is preferred — it lets the probe learn to weight each modality.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP multimodal (image + text).

        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading CLIP multimodal: {model_name}")
        from transformers import CLIPProcessor, CLIPModel

        self.device = get_device()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_name = model_name
        print(f"Model loaded. Embedding dimension: 512")

    def embed_text(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed texts using CLIP text encoder (no image)."""
        print(f"Embedding {len(texts)} texts with CLIP text encoder (batch_size={batch_size})...")
        
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i : i + batch_size]
                
                inputs = self.processor(
                    text=batch,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self.device)
                
                outputs = self.model.get_text_features(**inputs)
                batch_emb = outputs.cpu().numpy()
                
                # L2 normalize
                batch_emb = batch_emb / np.linalg.norm(batch_emb, axis=1, keepdims=True)
                
                embeddings.append(batch_emb)
        
        result = np.vstack(embeddings)
        assert result.shape == (len(texts), 512)
        return result

    def embed_image(self, images: list, batch_size: int = 32) -> np.ndarray:
        """
        Embed images using CLIP image encoder only (512d).

        Tests whether the visual encoder directly encodes spatial relations,
        independent of any text signal.

        Args:
            images: List of PIL Images (None entries get zero vectors)
            batch_size: Batch size

        Returns:
            (N, 512) array of L2-normalized image embeddings
        """
        print(f"Embedding {len(images)} images with CLIP image encoder (batch_size={batch_size})...")
        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size)):
                batch_images = images[i : i + batch_size]
                valid = [(j, img) for j, img in enumerate(batch_images) if img is not None]

                batch_emb = np.zeros((len(batch_images), 512), dtype=np.float32)
                if valid:
                    idxs, valid_images = zip(*valid)
                    inputs = self.processor(
                        images=list(valid_images),
                        return_tensors="pt",
                    ).to(self.device)
                    outputs = self.model.get_image_features(**inputs).cpu().numpy()
                    outputs = outputs / np.linalg.norm(outputs, axis=1, keepdims=True)
                    batch_emb[list(idxs)] = outputs

                embeddings.append(batch_emb)

        result = np.vstack(embeddings)
        assert result.shape == (len(images), 512)
        return result

    def embed_image_text(
        self,
        images: list,
        texts: list[str],
        batch_size: int = 32,
        fusion: str = "concat",
    ) -> np.ndarray:
        """
        Embed image-text pairs jointly with CLIP using concat fusion (1024d).

        Concat is the correct default for probing: it preserves image and text
        signals in separate halves of the vector so the probe can weight each
        independently. Averaging was the original default but cancels signal
        that lives in opposite directions across modalities.

        Args:
            images: List of PIL Images (None entries are treated as failed loads)
            texts: List of text strings
            batch_size: Batch size
            fusion: "concat" (recommended) or "average"

        Returns:
            L2-normalized embeddings, shape (N, 1024) for concat, (N, 512) for average
        """
        assert fusion in ("average", "concat"), f"fusion must be 'average' or 'concat', got {fusion!r}"
        assert len(images) == len(texts), "images and texts must have same length"

        out_dim = 512 if fusion == "average" else 1024
        print(f"Embedding {len(images)} image-text pairs with CLIP multimodal "
              f"(fusion={fusion}, batch_size={batch_size})...")

        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size)):
                batch_images = images[i : i + batch_size]
                batch_texts = texts[i : i + batch_size]

                valid_pairs = [
                    (img, txt) for img, txt in zip(batch_images, batch_texts) if img is not None
                ]

                if not valid_pairs:
                    batch_emb = np.zeros((len(batch_images), out_dim), dtype=np.float32)
                else:
                    valid_images, valid_texts = zip(*valid_pairs)

                    inputs = self.processor(
                        text=list(valid_texts),
                        images=list(valid_images),
                        padding=True,
                        truncation=True,
                        max_length=77,
                        return_tensors="pt",
                    ).to(self.device)

                    outputs = self.model(**inputs)
                    image_emb = outputs.image_embeds.cpu().numpy()  # (n_valid, 512)
                    text_emb = outputs.text_embeds.cpu().numpy()    # (n_valid, 512)

                    # L2-normalize each modality before fusion
                    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)
                    text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)

                    if fusion == "average":
                        fused = (image_emb + text_emb) / 2.0
                    else:  # concat
                        fused = np.concatenate([image_emb, text_emb], axis=1)  # (n_valid, 1024)

                    # L2-normalize the fused vector
                    fused = fused / np.linalg.norm(fused, axis=1, keepdims=True)

                    batch_emb = np.zeros((len(batch_images), out_dim), dtype=np.float32)
                    valid_idx = [
                        j for j, img in enumerate(batch_images) if img is not None
                    ]
                    batch_emb[valid_idx] = fused

                embeddings.append(batch_emb)

        result = np.vstack(embeddings)
        assert result.shape == (len(images), out_dim)
        return result


def get_embedder(model_name: str) -> BaseEmbedder:
    """
    Factory function to get embedder by model name.

    Args:
        model_name: One of:
            - "sbert" (Sentence-BERT)
            - "clip-text" (CLIP text encoder only)
            - "clip-multimodal" (CLIP with image+text)

    Returns:
        Initialized embedder instance
    """
    if model_name == "sbert":
        return SBERTEmbedder()
    elif model_name == "clip-text":
        return CLIPTextEmbedder()
    elif model_name == "clip-multimodal":
        return CLIPMultimodalEmbedder()
    else:
        raise ValueError(f"Unknown model: {model_name}")
