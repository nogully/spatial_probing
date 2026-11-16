"""Smoke test: end-to-end pipeline on N=100 VSR examples (CPU, no caching)."""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
np.random.seed(42)
torch.manual_seed(42)

N = 100

# --- 0. Probe logic on synthetic data (no models, no downloads) ---
# Run only this section with: python smoke_test.py --probe-only
print("=" * 60)
print("0. Synthetic check of probe_truth_by_relation...")
from src.probing import probe_truth_by_relation, probe_relation_identity

rng = np.random.default_rng(42)
syn_counts = {"signal": 200, "random": 300, "tiny": 10}
syn_rel = np.concatenate([np.full(n, r) for r, n in syn_counts.items()])
syn_emb = rng.normal(size=(len(syn_rel), 16)).astype(np.float32)
syn_y = rng.integers(0, 2, size=len(syn_rel))
# "signal": label is linearly decodable from dimension 0; "random": labels are noise
sig = syn_rel == "signal"
syn_y[sig] = (syn_emb[sig, 0] > 0).astype(int)

syn_truth = probe_truth_by_relation(syn_emb, syn_rel, syn_y, n_splits=5, C=1.0, min_samples=50)
syn_truth = syn_truth.set_index("relation")
print(syn_truth[["n_samples", "n_true", "n_false", "f1_macro_mean",
                 "baseline_stratified_f1_macro", "baseline_most_frequent_f1_macro"]].to_string())

assert "tiny" not in syn_truth.index, "relation below min_samples should be skipped"
for r in ("signal", "random"):
    row = syn_truth.loc[r]
    assert row["n_samples"] == syn_counts[r], (
        f"{r}: n_samples={row['n_samples']} but relation has {syn_counts[r]} examples"
    )
    assert row["n_true"] + row["n_false"] == syn_counts[r]
    assert row["n_true"] == int(syn_y[syn_rel == r].sum())
assert syn_truth.loc["signal", "f1_macro_mean"] > 0.9, "decodable labels should score high"
gap = abs(syn_truth.loc["random", "f1_macro_mean"] - syn_truth.loc["random", "baseline_stratified_f1_macro"])
assert gap < 0.15, f"random labels should score near the stratified baseline (gap={gap:.3f})"

# The identity probe is one-vs-rest over all rows; its n_samples is still the relation count
syn_ident = probe_relation_identity(syn_emb, syn_rel, syn_y, n_splits=5, min_samples=50).set_index("relation")
assert syn_ident.loc["random", "n_samples"] == syn_counts["random"]
assert {"n_true_caption", "n_false_caption"} <= set(syn_ident.columns)
# Grouped CV: single-key groups, and joint groups as connected components
from src.probing import make_groups
assert list(make_groups(np.array(["a", "b", "a", "c"]))) == [0, 1, 0, 2]
joint = make_groups(np.array(["a", "a", "b", "c", "d"]), np.array([1, 2, 2, 3, 4]))
assert joint[0] == joint[1] == joint[2], "rows linked by caption OR image must share a group"
assert len({joint[0], joint[3], joint[4]}) == 3

syn_caption = np.array([f"cap{i // 3}" for i in range(len(syn_rel))])   # groups of 3
syn_image = np.array([f"img{i // 2}" for i in range(len(syn_rel))])     # groups of 2
syn_grouped = probe_truth_by_relation(
    syn_emb, syn_rel, syn_y, n_splits=5, min_samples=50, groups=syn_caption
).set_index("relation")
assert (syn_grouped["cv"] == "stratified_group").all()
assert syn_grouped.loc["random", "n_samples"] == syn_counts["random"]
assert syn_grouped.loc["signal", "n_groups"] == 67 and syn_grouped.loc["signal", "max_group_size"] == 3
syn_joint = probe_truth_by_relation(
    syn_emb, syn_rel, syn_y, n_splits=5, min_samples=50, groups=[syn_caption, syn_image]
).set_index("relation")
assert syn_joint.loc["random", "max_group_size"] == 6   # lcm(3, 2) chains
assert syn_joint.loc["signal", "f1_macro_mean"] > 0.9
print("  Synthetic probe checks passed.")

if "--probe-only" in sys.argv:
    sys.exit(0)

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
from src.probing import train_probe, model_comparison_table

# Pooled True/False probe (all relations together) on each model
sbert_binary = train_probe(sbert_emb, binary_labels, n_splits=3, C=1.0)
print(f"  SBERT binary probe  — accuracy: {sbert_binary['accuracy_mean']:.3f} ± {sbert_binary['accuracy_std']:.3f}")
print(f"                        F1 macro:  {sbert_binary['f1_macro_mean']:.3f} ± {sbert_binary['f1_macro_std']:.3f}")

clip_binary = train_probe(clip_emb, binary_labels, n_splits=3, C=1.0)
print(f"  CLIP binary probe   — accuracy: {clip_binary['accuracy_mean']:.3f} ± {clip_binary['accuracy_std']:.3f}")
print(f"                        F1 macro:  {clip_binary['f1_macro_mean']:.3f} ± {clip_binary['f1_macro_std']:.3f}")

# Within-relation True/False probe (N=100 is tiny so min_samples=8 to get anything)
print("\n  Within-relation truth probe (SBERT)...")
sbert_per_rel = probe_truth_by_relation(
    sbert_emb, relation_types, binary_labels, n_splits=3, C=1.0, min_samples=8
)
print(sbert_per_rel[["relation", "n_samples", "f1_macro_mean"]].head(5).to_string(index=False))

clip_per_rel = probe_truth_by_relation(
    clip_emb, relation_types, binary_labels, n_splits=3, C=1.0, min_samples=8
)
comparison = model_comparison_table({"sbert": sbert_per_rel, "clip_text": clip_per_rel})
print("\n  Model comparison (truth probe)...")
print(comparison.head(5).to_string(index=False))

# Relation-identity probe (one-vs-rest over all examples)
print("\n  Relation-identity probe (SBERT, top 5 by F1)...")
sbert_ident = probe_relation_identity(
    sbert_emb, relation_types, binary_labels, n_splits=3, C=1.0, min_samples=5
)
clip_ident = probe_relation_identity(
    clip_emb, relation_types, binary_labels, n_splits=3, C=1.0, min_samples=5
)
print(model_comparison_table({"sbert": sbert_ident, "clip_text": clip_ident}).head(5).to_string(index=False))

# --- Done ---
print("\n" + "=" * 60)
print("SMOKE TEST PASSED — pipeline is end-to-end functional.")
print("Ready for full extraction on Colab.")
print("=" * 60)
