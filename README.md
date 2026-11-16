# Spatial Reasoning Probing Study

**Does visual training help language models understand space?**

One known weakness of language models is spatial reasoning. Having learned from text alone, they have no perceptual grounding for concepts like *above*, *left of*, or *inside* — they've only ever seen these words in context, never experienced what they refer to.

CLIP changes what gets paired in contrastive training: where SBERT learns to align sentences with other sentences, CLIP learns to align captions with images. The hypothesis here is that this image–text contrastive training leaves a residue in the text embedding space — making spatial relations more linearly decodable, even when no image is present at inference time.

This study set out to test that hypothesis using **probing classifiers**: lightweight logistic regression models trained on frozen embeddings to ask whether a caption’s True/False label is linearly decodable from each representation.

> *N.B. I am disclosing that I used AI (Claude Code) to assist with coding in this project, as encouraged in the course.*

---

## Initial Research Question

> Are spatial relations more linearly decodable from text embeddings produced by image–text contrastive training in a multimodal model (CLIP) than from those produced by text–text contrastive training in a text-only model (SBERT)? 

### Correction

The first version of this project probed relation identity (a one-vs-rest "is this caption about relation X?" classifier over all 7,680 examples) rather than caption truth **within each relation**. Those (retired) results are kept in [results/relation_identity/](results/relation_identity/), with the corresponding heatmap at [results/figures/05_relation_identity_heatmap.png](results/figures/05_relation_identity_heatmap.png).

### What the corrected probe shows

Caption-only probes score 0.487 (SBERT) and 0.489 (CLIP Text) against 0.482 for shuffled labels. In VSR, many captions appear with both a True image and a False image, so the caption alone carries no detectable information about its label. The two text encoders can’t be compared this way.

The CLIP image embedding does score above chance: 0.543 against 0.492 for shuffled labels, with six of 36 relations remaining significant after correction for multiple comparisons. The image probe never sees the caption, so this is signal about the image, not proof of spatial understanding.

---

## Results Summary

Within-relation truth probe: for each relation with at least 50 training examples, a probe is trained on that relation's examples to predict VSR's True/False caption label.

- Probe: `StandardScaler` + logistic regression (C=1.0)
- Cross-validation: grouped 5-fold (`StratifiedGroupKFold`), grouped by caption (SBERT, CLIP Text), by image (CLIP Image), or jointly by caption and image (CLIP Concat)
- Metric: macro F1
- Chance reference: label-permutation test, 1,000 shuffles within each relation; p-values are one-sided, and the smallest possible value is 0.001

All 36 relations (n ≥ 50):

| Condition | Mean | Median | Min | Max | Permutation null mean (95% range) | p for the mean | Relations with p < 0.05 (raw / Holm) |
|---|---|---|---|---|---|---|---|
| SBERT (`all-mpnet-base-v2`) | 0.487 | 0.489 | 0.386 | 0.592 | 0.482 (0.463–0.500) | 0.288 | 0 / 0 |
| CLIP Text encoder | 0.489 | 0.481 | 0.410 | 0.589 | 0.482 (0.463–0.501) | 0.234 | 0 / 0 |
| CLIP Image encoder | 0.543 | 0.530 | 0.374 | 0.701 | 0.492 (0.472–0.511) | ≤ 0.001 | 12 / 6 |
| CLIP Concat (image + text, 1024d) | 0.553 | 0.551 | 0.323 | 0.709 | 0.486 (0.468–0.506) | ≤ 0.001 | 13 / 6 |

The 19 relations with n ≥ 150:

| Condition | Mean | Median | Min | Max | Permutation null mean (95% range) | p for the mean | Relations with p < 0.05 (raw / Holm) |
|---|---|---|---|---|---|---|---|
| SBERT (`all-mpnet-base-v2`) | 0.493 | 0.493 | 0.446 | 0.518 | 0.488 (0.471–0.505) | 0.304 | 0 / 0 |
| CLIP Text encoder | 0.495 | 0.506 | 0.446 | 0.545 | 0.488 (0.471–0.506) | 0.215 | 0 / 0 |
| CLIP Image encoder | 0.573 | 0.550 | 0.465 | 0.693 | 0.497 (0.479–0.515) | ≤ 0.001 | 10 / 7 |
| CLIP Concat (image + text, 1024d) | 0.579 | 0.586 | 0.466 | 0.709 | 0.493 (0.475–0.511) | ≤ 0.001 | 10 / 8 |

### Wins / Ties / Losses

Per-relation macro F1 compared for each pair of conditions at the main setting. A tie is a difference smaller than 1e-9.

| a | b | All 36 relations: a wins / ties / a loses | mean difference (a − b) | 19 relations with n ≥ 150: a wins / ties / a loses | mean difference (a − b) |
|---|---|---|---|---|---|
| SBERT | CLIP Text | 17 / 0 / 19 | -0.002 | 8 / 0 / 11 | -0.002 |
| SBERT | CLIP Image | 11 / 0 / 25 | -0.056 | 3 / 0 / 16 | -0.080 |
| SBERT | CLIP Concat | 7 / 0 / 29 | -0.066 | 1 / 0 / 18 | -0.086 |
| CLIP Text | CLIP Image | 10 / 0 / 26 | -0.054 | 4 / 0 / 15 | -0.078 |
| CLIP Text | CLIP Concat | 6 / 0 / 30 | -0.064 | 2 / 0 / 17 | -0.084 |
| CLIP Image | CLIP Concat | 13 / 0 / 23 | -0.010 | 8 / 0 / 11 | -0.006 |

With plain stratified folds instead of grouped folds, mean macro F1 over the 36 relations is 0.308 / 0.304 / 0.539 / 0.478 (SBERT / CLIP Text / CLIP Image / CLIP Concat). The text conditions fall below chance there because captions repeat within a relation: 34% of test rows have their caption in the training fold only with the opposite True/False label (paired with a different image), so a probe that sees only the caption is pushed toward the wrong answer on those rows.

Full tables: [results/truth_probe/](results/truth_probe/). Other settings and checks (regularization sweep, repeated captions across stratified folds, interaction features, bag-of-words baseline): [results/diagnostics/](results/diagnostics/).

### Main Results Heatmap

![Main Results Heatmap](results/figures/05_main_heatmap.png)

*Left: observed macro F1 at the main setting. A star marks a permutation p < 0.05 for that cell; the stars are one-sided and not corrected for multiple comparisons across the 144 cells. Right: mean macro F1 of the permutation null (1,000 label shuffles) for the same cell.*

### Per-Relation Results

Macro F1 at the main setting, sorted by number of training examples (n). `*` marks permutation p < 0.05 (one-sided, 1,000 label shuffles, not corrected for multiple comparisons). Exact p-values, permutation null means, and the stratified-fold scores are in [results/truth_probe/comparison.csv](results/truth_probe/comparison.csv).

| Relation | n | SBERT | CLIP Text | CLIP Image | CLIP Concat |
|---|---|---|---|---|---|
| touching | 884 | 0.505 | 0.514 | 0.605* | 0.603* |
| in front of | 528 | 0.509 | 0.507 | 0.514 | 0.527 |
| behind | 498 | 0.516 | 0.506 | 0.538 | 0.517 |
| under | 415 | 0.474 | 0.481 | 0.550* | 0.539 |
| on | 408 | 0.503 | 0.510 | 0.569* | 0.590* |
| on top of | 349 | 0.500 | 0.510 | 0.623* | 0.684* |
| at the right side of | 335 | 0.493 | 0.517 | 0.504 | 0.545 |
| at the left side of | 273 | 0.509 | 0.545 | 0.504 | 0.519 |
| beneath | 250 | 0.488 | 0.467 | 0.614* | 0.606* |
| contains | 242 | 0.446 | 0.446 | 0.680* | 0.586* |
| above | 237 | 0.504 | 0.539 | 0.530 | 0.625* |
| facing | 219 | 0.467 | 0.480 | 0.674* | 0.620* |
| next to | 216 | 0.490 | 0.535 | 0.531 | 0.524 |
| below | 189 | 0.487 | 0.459 | 0.484 | 0.512 |
| in | 186 | 0.487 | 0.507 | 0.693* | 0.655* |
| inside | 177 | 0.485 | 0.485 | 0.673* | 0.709* |
| far away from | 161 | 0.518 | 0.480 | 0.594* | 0.609* |
| left of | 155 | 0.483 | 0.463 | 0.465 | 0.466 |
| at the edge of | 151 | 0.501 | 0.449 | 0.540 | 0.557 |
| beside | 136 | 0.445 | 0.460 | 0.497 | 0.481 |
| facing away from | 121 | 0.433 | 0.410 | 0.424 | 0.437 |
| far from | 111 | 0.570 | 0.534 | 0.526 | 0.544 |
| away from | 105 | 0.466 | 0.517 | 0.640* | 0.640* |
| part of | 85 | 0.487 | 0.471 | 0.701* | 0.650* |
| near | 79 | 0.592 | 0.579 | 0.450 | 0.445 |
| across from | 76 | 0.540 | 0.463 | 0.570 | 0.601 |
| right of | 75 | 0.502 | 0.458 | 0.486 | 0.507 |
| close to | 67 | 0.444 | 0.414 | 0.391 | 0.439 |
| at the back of | 65 | 0.527 | 0.589 | 0.481 | 0.516 |
| surrounding | 65 | 0.510 | 0.554 | 0.562 | 0.615* |
| in the middle of | 59 | 0.479 | 0.503 | 0.404 | 0.466 |
| parallel to | 59 | 0.386 | 0.432 | 0.374 | 0.323 |
| off | 54 | 0.425 | 0.478 | 0.522 | 0.583 |
| over | 54 | 0.495 | 0.468 | 0.512 | 0.598 |
| perpendicular to | 52 | 0.424 | 0.440 | 0.524 | 0.583 |
| adjacent to | 51 | 0.438 | 0.425 | 0.582 | 0.479 |


---

## Models

| Model | Type | Training Objective | Visual Signal |
|---|---|---|---|
| `sentence-transformers/all-mpnet-base-v2` | SBERT (MPNet-based) | Contrastive, over 1 billion sentence pairs | None |
| `openai/clip-vit-base-patch32` (text encoder) | CLIP Text | Contrastive image-text | Indirect |
| `openai/clip-vit-base-patch32` (image encoder) | CLIP Image | Contrastive image-text | Direct |
| CLIP text + image (concat, 1024d) | CLIP Concat | Contrastive image-text | Both |

CLIP Text is the text encoder's pooled output *before* CLIP's projection layer; the text half of CLIP Concat is the *projected* text embedding (the shared image–text space), so the two are different vectors for the same caption.

---

## Dataset

**VSR — Visual Spatial Reasoning** (Liu et al., 2022)
- 7,680 training examples (image, caption, relation type, True/False label)
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
- A **HuggingFace token** (optional - to suppress rate-limit warnings when downloading models)

Open `notebooks/02_embedding_extraction.ipynb`, run the Colab setup cell (mounts Drive, clones repo, installs deps), then run all cells. Embeddings are cached to Google Drive as `.npy` files.

Sync the following files to `results/embeddings/` locally before running probing:
- `sbert_vsr_train.npy`
- `clip_text_vsr_train.npy`
- `clip_image_vsr_train.npy`
- `clip_concat_vsr_train.npy`

### Probing and Visualization (local)

Run the permutation test, then notebooks 03, 04 + 05 locally:

```
python permutation_test.py --standardize --C 1.0 --cv grouped --models sbert clip_text clip_image clip_concat
notebooks/03_probing_experiments.ipynb   # trains probes, saves CSVs to results/truth_probe/
notebooks/04_diagnostics.ipynb           # extra analyses, saves CSVs to results/diagnostics/
notebooks/05_visualization.ipynb         # all graphics (I like data viz)
```

---

## Project Structure

```
spatial_probing/
├── README.md
├── requirements.txt
├── smoke_test.py
├── permutation_test.py       # label-permutation test for the truth probe
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

Nora Gully — University of Colorado Boulder, CSCI 4622 Machine Learning, Spring 2026, revised Nov 2026
