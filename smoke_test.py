"""Smoke test: end-to-end pipeline on N=100 VSR examples (CPU, no caching)."""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
np.random.seed(42)
torch.manual_seed(42)

N = 100

# --- 1. Load VSR ---
print("=" * 60)
print("1. Loading VSR dataset (first 100 examples)...")
from datasets import load_dataset
vsr = load_dataset("cambridgeltl/vsr_random", split="train")
vsr_subset = vsr.select(range(N))

texts = [ex["caption"] for ex in vsr_subset]
relation_types = np.array([ex["relation"] for ex in vsr_subset])
binary_labels = np.array([int(ex["label"]) for ex in vsr_subset])

print(f"  texts: {len(texts)}")
print(f"  unique relations: {len(np.unique(relation_types))}")
print(f"  label balance: {binary_labels.sum()} True, {(1-binary_labels).sum()} False")
print(f"  sample: '{texts[0]}'")

# --- 2. SBERT embeddings ---
print("\n2. SBERT embeddings...")
from src.embedders import SBERTEmbedder
sbert = SBERTEmbedder()
sbert_emb = sbert.embed_text(texts, batch_size=64)
norms = np.linalg.norm(sbert_emb, axis=1)
print(f"  shape: {sbert_emb.shape}")
print(f"  norms: min={norms.min():.4f} max={norms.max():.4f} (should be ~1.0)")
assert sbert_emb.shape == (N, 768), f"Expected (100, 768), got {sbert_emb.shape}"
assert np.allclose(norms, 1.0, atol=1e-5), "SBERT embeddings not L2-normalized"

# --- 3. CLIP text embeddings ---
print("\n3. CLIP text embeddings...")
from src.embedders import CLIPTextEmbedder
clip = CLIPTextEmbedder()
clip_emb = clip.embed_text(texts, batch_size=32)
norms = np.linalg.norm(clip_emb, axis=1)
print(f"  shape: {clip_emb.shape}")
print(f"  norms: min={norms.min():.4f} max={norms.max():.4f} (should be ~1.0)")
assert clip_emb.shape == (N, 512), f"Expected (100, 512), got {clip_emb.shape}"
assert np.allclose(norms, 1.0, atol=1e-5), "CLIP embeddings not L2-normalized"

# --- 4. Probing ---
print("\n4. Probing classifiers...")
from src.probing import train_probe, probe_by_relation_type, model_comparison_table

# Binary probe (True/False label) on each model
sbert_binary = train_probe(sbert_emb, binary_labels, n_splits=3, C=1.0)
print(f"  SBERT binary probe  — accuracy: {sbert_binary['accuracy_mean']:.3f} ± {sbert_binary['accuracy_std']:.3f}")
print(f"                        F1 macro:  {sbert_binary['f1_macro_mean']:.3f} ± {sbert_binary['f1_macro_std']:.3f}")

clip_binary = train_probe(clip_emb, binary_labels, n_splits=3, C=1.0)
print(f"  CLIP binary probe   — accuracy: {clip_binary['accuracy_mean']:.3f} ± {clip_binary['accuracy_std']:.3f}")
print(f"                        F1 macro:  {clip_binary['f1_macro_mean']:.3f} ± {clip_binary['f1_macro_std']:.3f}")

# Per-relation probe (N=100 is tiny so min_samples=5 to get anything)
print("\n  Per-relation probing (SBERT, top 5 by F1)...")
sbert_per_rel = probe_by_relation_type(
    sbert_emb, relation_types, binary_labels, n_splits=3, C=1.0, min_samples=5
)
print(sbert_per_rel[["relation", "n_samples", "f1_macro_mean"]].head(5).to_string(index=False))

clip_per_rel = probe_by_relation_type(
    clip_emb, relation_types, binary_labels, n_splits=3, C=1.0, min_samples=5
)
comparison = model_comparison_table({"sbert": sbert_per_rel, "clip_text": clip_per_rel})
print("\n  Model comparison (top 5 relations)...")
print(comparison.head(5).to_string(index=False))

# --- Done ---
print("\n" + "=" * 60)
print("SMOKE TEST PASSED — pipeline is end-to-end functional.")
print("Ready for full extraction on Colab.")
print("=" * 60)
