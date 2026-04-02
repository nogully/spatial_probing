# Spatial Reasoning Probing Study – Study 1

Investigating whether visually grounded embeddings (CLIP) encode spatial relations better than purely distributional embeddings (BERT/SBERT) using probing classifiers.

## Core Research Question

Does visual contrastive training (CLIP) produce text embeddings that encode spatial relations better than purely distributional training (BERT/SBERT)? And which specific spatial relation types benefit — or remain resistant?

## Quick Start

```bash
pip install -r requirements.txt
```

Development runs locally in VS Code. GPU-bound embedding extraction runs on Google Colab.

## Project Structure

```
spatial_probing/
├── CLAUDE.md              # detailed implementation guide (not committed)
├── README.md              # this file
├── requirements.txt       # dependencies
├── .env                   # gitignored (local config)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_extraction.ipynb
│   ├── 03_probing_experiments.ipynb
│   ├── 04_rsa_analysis.ipynb
│   └── 05_visualization.ipynb
│
├── src/
│   ├── __init__.py
│   ├── datasets.py        # dataset loading + preprocessing
│   ├── embedders.py       # model loading + embedding extraction
│   ├── probing.py         # probe training + evaluation
│   └── analysis.py        # RSA, ranking tasks, visualization helpers
│
└── results/
    ├── embeddings/        # cached .npy files (gitignored)
    ├── probes/            # saved probe models (gitignored)
    └── figures/           # output plots
```

## Models Under Study

1. **Sentence-BERT** (`sentence-transformers/all-mpnet-base-v2`) — baseline distributional model
2. **CLIP Text** (`openai/clip-vit-base-patch32` text encoder) — contrastive training, text-only
3. **CLIP Multimodal** (text + image) — contrastive training, joint embedding

## Datasets

- **VSR** (Visual Spatial Reasoning) — primary dataset, 10k examples with images
- **SpartQA** — text-only multi-hop spatial inference
- **StepGame** — structured complexity scaling

## Build Order

1. `src/datasets.py` — VSR loader
2. `src/embedders.py` — SBERT and CLIP text embedders
3. `notebooks/01_data_exploration.ipynb`
4. `notebooks/02_embedding_extraction.ipynb`
5. `src/probing.py` — logistic regression probe
6. `notebooks/03_probing_experiments.ipynb`
7. Add CLIP multimodal embedder
8. `src/analysis.py` — RSA
9. `notebooks/04_rsa_analysis.ipynb`
10. `notebooks/05_visualization.ipynb`

## Key References

- Radford et al. (2021) — CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- Liu et al. (2022) — VSR: [Visual Spatial Reasoning](https://arxiv.org/abs/2205.00363)
- Belinkov (2022) — Probing Classifiers survey: [Promises, Shortcomings, and Advances](https://direct.mit.edu/coli/article/48/1/207/107571)

## Authors

Nora Gully
