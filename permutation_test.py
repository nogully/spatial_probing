"""Label-permutation test for the within-relation truth probe.

For each condition and CV scheme, shuffles the True/False labels within each
relation many times, reruns the full cross-validated probe each time, and
compares the observed macro F1 to that null distribution. See
`src.probing.permutation_test_truth_probe` for the exact procedure.

Usage:
    python permutation_test.py                       # image + concat, C=1.0, 1000 permutations
    python permutation_test.py --standardize --C 1.0 --tag scaled
    python permutation_test.py --models sbert clip_text clip_image clip_concat

Outputs (in --out-dir, default results/truth_probe/; <tag> is empty by default):
    permutation_test<tag>.csv           one row per condition x cv x relation
    permutation_test<tag>_summary.csv   one row per condition x cv x relation subset
    permutation_null<tag>/<model>_<cv>.csv   null macro F1, permutations x relations
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.probing import permutation_test_truth_probe  # noqa: E402

GROUP_DESC = {
    "sbert": "caption",
    "clip_text": "caption",
    "clip_image": "image",
    "clip_concat": "caption+image (joint)",
}


def holm_adjust(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def summarize(per_rel: pd.DataFrame, null: np.ndarray, subset: np.ndarray, label: str) -> dict:
    """Across-relation summary for the relations selected by boolean `subset`."""
    obs = per_rel.loc[subset, "observed_f1"].to_numpy()
    null_mean = null[:, subset].mean(axis=1)  # one joint draw per permutation
    n_perm = len(null_mean)
    p_up = per_rel.loc[subset, "p_upper"].to_numpy()
    p_lo = per_rel.loc[subset, "p_lower"].to_numpy()
    return {
        "relations": label,
        "n_relations": int(subset.sum()),
        "observed_mean_f1": obs.mean(),
        "null_mean_f1": null_mean.mean(),
        "null_std_of_mean": null_mean.std(),
        "null_p2.5_of_mean": np.percentile(null_mean, 2.5),
        "null_p97.5_of_mean": np.percentile(null_mean, 97.5),
        "p_upper_of_mean": (1 + (null_mean >= obs.mean() - 1e-12).sum()) / (1 + n_perm),
        "p_lower_of_mean": (1 + (null_mean <= obs.mean() + 1e-12).sum()) / (1 + n_perm),
        "n_relations_p_upper_lt_0.05": int((p_up < 0.05).sum()),
        "n_relations_p_upper_lt_0.05_holm": int((holm_adjust(p_up) < 0.05).sum()),
        "n_relations_p_lower_lt_0.05": int((p_lo < 0.05).sum()),
        "n_relations_p_lower_lt_0.05_holm": int((holm_adjust(p_lo) < 0.05).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["clip_image", "clip_concat"])
    parser.add_argument("--cv", nargs="+", default=["stratified", "grouped"], choices=["stratified", "grouped"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--large-n", type=int, default=150, help="threshold for the 'n >= large-n' summary subset")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--tag", default="", help="suffix for output names, e.g. '_scaled'")
    parser.add_argument("--out-dir", default="results/truth_probe")
    args = parser.parse_args()

    from datasets import load_dataset
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    cache_dir = Path(os.getenv("CACHE_DIR", "results/embeddings"))
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    out_dir = PROJECT_ROOT / args.out_dir
    null_dir = out_dir / f"permutation_null{args.tag}"
    null_dir.mkdir(parents=True, exist_ok=True)

    train = load_dataset("cambridgeltl/vsr_random")["train"]
    relation_types = np.array(train["relation"])
    binary_labels = np.array([int(v) for v in train["label"]])
    captions = np.array(train["caption"])
    image_files = np.array(train["image"])
    group_keys = {
        "sbert": captions,
        "clip_text": captions,
        "clip_image": image_files,
        "clip_concat": [captions, image_files],
    }

    detail, summary = [], []
    for model in args.models:
        embeddings = np.load(cache_dir / f"{model}_vsr_train.npy")
        for cv_name in args.cv:
            start = time.time()
            per_rel, null = permutation_test_truth_probe(
                embeddings, relation_types, binary_labels,
                n_permutations=args.n_permutations, n_splits=5, C=args.C,
                min_samples=args.min_samples,
                groups=group_keys[model] if cv_name == "grouped" else None,
                standardize=args.standardize, n_jobs=args.n_jobs, seed=42,
            )
            per_rel["p_upper_holm"] = holm_adjust(per_rel["p_upper"].to_numpy())
            meta = {
                "condition": model,
                "cv": cv_name,
                "grouping": GROUP_DESC[model] if cv_name == "grouped" else "none",
                "C": args.C,
                "standardize": args.standardize,
                "n_permutations": args.n_permutations,
            }
            detail.append(per_rel.assign(**meta)[list(meta) + list(per_rel.columns)])

            everything = np.ones(len(per_rel), dtype=bool)
            large = (per_rel["n_samples"] >= args.large_n).to_numpy()
            summary.append({**meta, **summarize(per_rel, null, everything, f"all (n >= {args.min_samples})")})
            summary.append({**meta, **summarize(per_rel, null, large, f"n >= {args.large_n}")})

            pd.DataFrame(null, columns=per_rel["relation"]).round(5).to_csv(
                null_dir / f"{model}_{cv_name}.csv", index=False
            )
            print(f"{model:12s} {cv_name:10s} done in {time.time() - start:6.1f}s", flush=True)

            # Write after every run so partial results survive an interruption
            pd.concat(detail, ignore_index=True).to_csv(out_dir / f"permutation_test{args.tag}.csv", index=False)
            pd.DataFrame(summary).to_csv(out_dir / f"permutation_test{args.tag}_summary.csv", index=False)

    pd.set_option("display.width", 250)
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
