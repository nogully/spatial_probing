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
    
    Note: Encodes image and text separately, then mixes (e.g., average pool).
    This studies whether having access to images during training improves
    spatial relation encoding in the text embedding space.
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

    def embed_image_text(
        self, images: list, texts: list[str], batch_size: int = 32
    ) -> np.ndarray:
        """
        Embed image-text pairs jointly with CLIP.
        
        Args:
            images: List of PIL Images
            texts: List of text strings
            batch_size: Batch size

        Returns:
            (N, 512) L2-normalized embeddings
        """
        assert len(images) == len(texts), "images and texts must have same length"
        
        print(f"Embedding {len(images)} image-text pairs with CLIP multimodal (batch_size={batch_size})...")
        
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size)):
                batch_images = images[i : i + batch_size]
                batch_texts = texts[i : i + batch_size]
                
                # Filter out None images
                valid_pairs = [
                    (img, txt) for img, txt in zip(batch_images, batch_texts) if img is not None
                ]
                
                if not valid_pairs:
                    # All images in batch failed — use text-only
                    batch_emb = np.zeros((len(batch_images), 512), dtype=np.float32)
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
                    
                    # Get multimodal embeddings
                    # CLIP projects both to same embedding space
                    # Use image embeddings if available, else text embeddings
                    image_emb = outputs.image_embeds  # (n_valid, 512)
                    text_emb = outputs.text_embeds    # (n_valid, 512)
                    
                    # Average image and text embeddings
                    multimodal_emb = (image_emb + text_emb) / 2.0
                    
                    # Convert to numpy and L2 normalize
                    multimodal_emb = multimodal_emb.cpu().numpy()
                    multimodal_emb = multimodal_emb / np.linalg.norm(
                        multimodal_emb, axis=1, keepdims=True
                    )
                    
                    # Fill in the full batch (using zeros for failed images)
                    batch_emb = np.zeros((len(batch_images), 512), dtype=np.float32)
                    valid_idx = [
                        j for j, (img, _) in enumerate(zip(batch_images, batch_texts))
                        if img is not None
                    ]
                    batch_emb[valid_idx] = multimodal_emb
                
                embeddings.append(batch_emb)
        
        result = np.vstack(embeddings)
        assert result.shape == (len(images), 512)
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
