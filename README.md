# Spatial Reasoning Probing Study

**Does visual training help language models understand space?**

One known weakness of language models is spatial reasoning. Having learned from text alone, they have no perceptual grounding for concepts like *above*, *left of*, or *inside* — they've only ever seen these words in context, never experienced what they refer to.

CLIP changes the training regime: instead of predicting masked tokens, it learns to align image-caption pairs. The hypothesis here is that this contrastive visual training leaves a residue in the text embedding space — making spatial relations more linearly decodable, even when no image is present at inference time.

This study tests that hypothesis using **probing classifiers**: lightweight logistic regression models trained on frozen embeddings to ask whether spatial relation labels are linearly decodable from each representation.

> *N.B. I am disclosing that I used AI (Claude Code) to assist with coding in this project, as encouraged in the course. Research design, experimental decisions, and interpretation of results are my own.*

---

## Research Question

> Does visual contrastive training (CLIP) produce text embeddings that encode spatial relations better than purely distributional training (SBERT)? And which specific relation types benefit — or remain resistant?

---

## Main Deliverable

[Gully_Probing_Study.ipynb](Gully_Probing_Study.ipynb)

## Results Summary

Within-relation truth probe: for each relation with at least 50 training examples, a logistic regression probe (C=1.0) is trained on that relation's examples to predict VSR's True/False caption label. Macro F1, 5-fold CV, 36 relations.

"Stratified" is plain `StratifiedKFold`. "Grouped" is `StratifiedGroupKFold`, grouped by caption (SBERT, CLIP Text), by image (CLIP Image), or jointly by caption and image (CLIP Concat).

| Condition | Stratified: mean | median | min | max | Grouped: mean | median | min | max |
|---|---|---|---|---|---|---|---|---|
| SBERT (`all-mpnet-base-v2`) | 0.370 | 0.384 | 0.251 | 0.474 | 0.463 | 0.463 | 0.365 | 0.537 |
| CLIP Text encoder | 0.370 | 0.368 | 0.233 | 0.502 | 0.445 | 0.446 | 0.342 | 0.532 |
| CLIP Image encoder | 0.506 | 0.521 | 0.361 | 0.687 | 0.503 | 0.474 | 0.344 | 0.730 |
| CLIP Concat (image + text, 1024d) | 0.455 | 0.422 | 0.315 | 0.689 | 0.488 | 0.506 | 0.349 | 0.680 |
| Dummy baseline: `most_frequent` | 0.350 | 0.347 | 0.334 | 0.412 | 0.350 | | | |
| Dummy baseline: `stratified` | 0.494 | 0.498 | 0.371 | 0.593 | 0.484–0.488 | | | |

Grouped-fold baselines vary slightly by condition because the folds differ; the range of means is shown.

SBERT vs CLIP Text, relations won / tied / lost by SBERT (tie = difference < 1e-9): stratified folds 15 / 1 / 20; grouped folds 28 / 1 / 7.

Full tables: [results/truth_probe/](results/truth_probe/). Additional diagnostics (interaction features, bag-of-words baseline, regularization sweep): [results/diagnostics/](results/diagnostics/).

### Main Results Heatmap

![Main Results Heatmap](results/figures/05_main_heatmap.png)

### Correction

The first version of this project probed relation identity (a one-vs-rest "is this caption about relation X?" classifier over all 7,680 examples) rather than caption truth within each relation. Those results are kept in [results/relation_identity/](results/relation_identity/), with the corresponding heatmap at [results/figures/05_relation_identity_heatmap.png](results/figures/05_relation_identity_heatmap.png).


---

## Models

| Model | Type | Training Objective | Visual Signal |
|---|---|---|---|
| `sentence-transformers/all-mpnet-base-v2` | SBERT (MPNet-based) | Contrastive, over 1 billion sentence pairs | None |
| `openai/clip-vit-base-patch32` (text encoder) | CLIP Text | Contrastive image-text | Indirect |
| `openai/clip-vit-base-patch32` (image encoder) | CLIP Image | Contrastive image-text | Direct |
| CLIP text + image (concat, 1024d) | CLIP Concat | Contrastive image-text | Both |

---

## Dataset

**VSR — Visual Spatial Reasoning** (Liu et al., 2022)
- ~7,680 training examples (image, caption, relation type, True/False label)
- 64 relation types; 36 retained after filtering relations with fewer than 50 examples
- Images are COCO images fetched at runtime; only filenames stored in the HuggingFace dataset

---

## Reproducing the Results

### Environment

```bash
git clone https://github.com/nogully/spatial_probing.git
cd spatial_probing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Embedding Extraction (Google Colab — GPU required)

You may need:
- A **GitHub Personal Access Token** (to clone the repo in Colab - should be able to do without tho)
- A **HuggingFace token** (optional - to suppress rate-limit warnings when downloading models)

Open `notebooks/02_embedding_extraction.ipynb`, run the Colab setup cell (mounts Drive, clones repo, installs deps), then run all cells. Embeddings are cached to Google Drive as `.npy` files.

Sync the following files to `results/embeddings/` locally before running probing:
- `sbert_vsr_train.npy`
- `clip_text_vsr_train.npy`
- `clip_image_vsr_train.npy`
- `clip_concat_vsr_train.npy`

### Probing and Visualization (local)

Run notebooks 03, 04 + 05 locally:

```
notebooks/03_probing_experiments.ipynb   # trains probes, saves CSVs to results/truth_probe/
notebooks/04_diagnostics.ipynb           # extra analyses, saves CSVs to results/diagnostics/
notebooks/05_visualization.ipynb         # all graphics (I like data viz)
```

---

## Project Structure

```
spatial_probing/
├── README.md
├── Gully_Probing_Study.ipynb    # main deliverable (loads pre-computed results)
├── Gully_Probing_Study.html
├── requirements.txt
├── smoke_test.py
├── .env.example                 # copy to .env (gitignored) and set CACHE_DIR
│
├── notebooks/
│   ├── 02_embedding_extraction.ipynb    # run on Colab
│   ├── 03_probing_experiments.ipynb
│   ├── 04_diagnostics.ipynb
│   └── 05_visualization.ipynb
│
├── src/
│   ├── datasets.py              # VSR loader
│   ├── embedders.py             # SBERT, CLIP text, image, concat embedders
│   └── probing.py               # logistic regression probes + CV
│
└── results/
    ├── embeddings/              # cached .npy files — gitignored
    ├── truth_probe/             # within-relation True/False probe results
    ├── relation_identity/       # earlier one-vs-rest relation-identity results
    ├── diagnostics/             # extra analyses
    └── figures/                 # output plots
```

---

## References

- Radford et al. (2021) — CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- Reimers & Gurevych (2019) — SBERT: [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- Liu et al. (2022) — VSR: [Visual Spatial Reasoning](https://arxiv.org/abs/2205.00363)


---

## Author

Nora Gully — University of Colorado Boulder, CSCI 4622 Machine Learning, Spring 2026
