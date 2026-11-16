"""Probing classifiers for spatial relation representations."""

import warnings
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


class LogisticRegressionProbe:
    """
    Logistic regression probe for testing linear decodability of properties in embeddings.
    
    Uses stratified k-fold cross-validation to estimate generalization performance.
    Intentionally simple (no kernel tricks) to ensure positive results are due to
    the representation, not the probe complexity.
    
    Reference: Belinkov (2022) "Probing Classifiers: Promises, Shortcomings, and Advances"
    https://direct.mit.edu/coli/article/48/1/207/107571
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        """
        Initialize logistic regression probe.

        Args:
            C: Inverse of regularization strength (higher = weaker regularization).
               Use high C to avoid over-regularizing and destroying representational signal.
            max_iter: Maximum iterations for solver
            random_state: Random seed
        """
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.clf = None
        self.cv_results = None

    def train_with_cv(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_splits: int = 5,
    ) -> Dict:
        """
        Train probe with stratified k-fold cross-validation.

        Args:
            embeddings: (N, dim) array of embeddings
            labels: (N,) array of labels (binary 0/1 or multiclass)
            n_splits: Number of CV folds

        Returns:
            Dict with 'accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall'
            per fold, plus mean and std across folds.
        """
        # Define scoring metrics
        scoring = {
            "accuracy": make_scorer(accuracy_score),
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
            "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
            "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
            "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        }

        # Stratified k-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Cross-validation
        cv_results = cross_validate(
            LogisticRegression(
                C=self.C, max_iter=self.max_iter, random_state=self.random_state, solver="lbfgs"
            ),
            embeddings,
            labels,
            cv=skf,
            scoring=scoring,
            return_train_score=False,
        )

        # Format results
        results = {}
        for metric in scoring.keys():
            key = f"test_{metric}"
            scores = cv_results[key]
            results[f"{metric}_mean"] = scores.mean()
            results[f"{metric}_std"] = scores.std()
            results[f"{metric}_scores"] = scores

        return results


def train_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
) -> Dict:
    """
    Train a logistic regression probe on frozen embeddings.

    Args:
        embeddings: (N, dim) array of L2-normalized embeddings
        labels: (N,) array of labels (binary 0/1 or multiclass integers)
        n_splits: Number of CV folds
        C: Regularization hyperparameter (higher = weaker regularization)

    Returns:
        Dictionary with results:
            - 'accuracy_mean', 'accuracy_std'
            - 'f1_macro_mean', 'f1_macro_std'
            - 'f1_weighted_mean', 'f1_weighted_std'
            - 'f1_macro_scores': array of per-fold F1 scores
    """
    probe = LogisticRegressionProbe(C=C, max_iter=1000, random_state=42)
    results = probe.train_with_cv(embeddings, labels, n_splits=n_splits)
    return results


def probe_relation_identity(
    embeddings: np.ndarray,
    relation_labels: np.ndarray,
    binary_labels: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
    min_samples: int = 20,
) -> pd.DataFrame:
    """
    One-vs-rest relation-identity probe, fit on ALL examples.

    For each relation X, the target is relation membership:
        y = 1 if the example's relation is X, 0 for every other example
    and the probe is cross-validated on all N rows. This measures whether the
    relation a caption is about is linearly decodable. It does NOT predict
    VSR's True/False caption label; for that, use `probe_truth_by_relation`.

    (This function was previously named `probe_by_relation_type`.)

    Args:
        embeddings: (N, dim) array of embeddings
        relation_labels: (N,) array of relation type strings
        binary_labels: (N,) array of VSR True/False labels. Used only for the
            descriptive counts `n_true_caption` / `n_false_caption`; never
            used as a classification target here.
        n_splits: Number of CV folds
        C: Regularization hyperparameter
        min_samples: Skip relations with fewer than this many examples

    Returns:
        DataFrame with columns:
            - 'relation': relation type
            - 'n_samples': number of examples of this relation (the positives)
            - 'n_true_caption', 'n_false_caption': VSR label counts within
              the relation (not the classifier's positives/negatives)
            - 'accuracy_mean', 'accuracy_std'
            - 'f1_macro_mean', 'f1_macro_std', 'f1_weighted_mean'
            - 'precision_mean', 'recall_mean'
    """
    unique_relations = sorted(np.unique(relation_labels))

    results_list = []

    for relation in unique_relations:
        # One-vs-rest target over all examples: is this example about `relation`?
        is_this_relation = (relation_labels == relation).astype(int)

        # Skip if too few samples
        if np.sum(is_this_relation) < min_samples:
            continue

        probe_results = train_probe(embeddings, is_this_relation, n_splits=n_splits, C=C)

        result_row = {
            "relation": relation,
            "n_samples": np.sum(is_this_relation),
            "n_true_caption": np.sum((is_this_relation == 1) & (binary_labels == 1)),
            "n_false_caption": np.sum((is_this_relation == 1) & (binary_labels == 0)),
            "accuracy_mean": probe_results["accuracy_mean"],
            "accuracy_std": probe_results["accuracy_std"],
            "f1_macro_mean": probe_results["f1_macro_mean"],
            "f1_macro_std": probe_results["f1_macro_std"],
            "f1_weighted_mean": probe_results["f1_weighted_mean"],
            "precision_mean": probe_results["precision_macro_mean"],
            "recall_mean": probe_results["recall_macro_mean"],
        }
        results_list.append(result_row)

    return pd.DataFrame(results_list).sort_values("f1_macro_mean", ascending=False)


BASELINE_STRATEGIES = ("most_frequent", "stratified")

GroupKeys = Union[np.ndarray, Sequence[np.ndarray]]


def make_groups(*keys: np.ndarray) -> np.ndarray:
    """
    Build integer CV group ids from one or more per-example keys.

    With one key (e.g. captions), examples sharing that key share a group.
    With several keys (e.g. captions and image filenames), groups are the
    connected components of the "shares any key" graph: two examples land in
    the same group if they share a caption OR an image, transitively. This is
    the joint grouping used for the image+text condition.

    Args:
        *keys: one or more (N,) arrays of hashable values

    Returns:
        (N,) integer array of group ids
    """
    if len(keys) == 0:
        raise ValueError("make_groups needs at least one key array")
    n = len(keys[0])
    codes = []
    for key in keys:
        if len(key) != n:
            raise ValueError("all key arrays must have the same length")
        codes.append(pd.factorize(np.asarray(key))[0])
    if len(codes) == 1:
        return codes[0]

    # Bipartite graph: example nodes 0..n-1, then one node per distinct key value
    rows, cols, offset = [], [], n
    for code in codes:
        rows.append(np.arange(n))
        cols.append(offset + code)
        offset += code.max() + 1
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(offset, offset))
    return connected_components(graph, directed=False)[1][:n]


def probe_truth_by_relation(
    embeddings: np.ndarray,
    relation_labels: np.ndarray,
    binary_labels: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
    min_samples: int = 50,
    groups: Optional[GroupKeys] = None,
    standardize: bool = False,
) -> pd.DataFrame:
    """
    Within-relation caption-truth probe.

    For each relation with at least `min_samples` examples, restrict to the
    examples of that relation and predict VSR's True/False label:
        mask = relation_labels == relation
        X = embeddings[mask], y = binary_labels[mask]

    Probe: LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
    random_state=42).

    Cross-validation:
      - groups=None: StratifiedKFold(n_splits, shuffle=True, random_state=42).
        The same caption or image can appear in both train and test folds.
      - groups given: StratifiedGroupKFold(n_splits, shuffle=True,
        random_state=42), so no group is split across train and test.
        `groups` is either one (N,) key array (e.g. captions for text
        conditions, image filenames for image conditions) or a sequence of
        key arrays (e.g. [captions, images]) for joint grouping. Group ids
        are built per relation with `make_groups`.

    Two DummyClassifier baselines are scored on the same folds:
    `most_frequent` (always predict the training fold's majority class) and
    `stratified` (random predictions drawn from the training fold's class
    distribution, random_state=42). These define chance for this task. Note
    that the majority-class baseline's macro F1 is below 0.5 (about 0.33 for
    balanced classes), because the never-predicted class contributes F1 = 0.

    Args:
        embeddings: (N, dim) array of embeddings
        relation_labels: (N,) array of relation type strings
        binary_labels: (N,) array of VSR True/False labels (1/0) — the target
        n_splits: Number of CV folds
        C: Regularization hyperparameter
        min_samples: Skip relations with fewer than this many examples
        groups: Optional CV grouping key(s), see above
        standardize: If True, fit a StandardScaler on each training fold and
            apply it before the probe (diagnostic only; default False)

    Returns:
        DataFrame, one row per relation, sorted by n_samples (descending):
            - 'relation', 'n_samples', 'n_true', 'n_false'
            - 'cv': "stratified" or "stratified_group"
            - 'n_groups', 'max_group_size' (n_samples and 1 when ungrouped)
            - 'accuracy_mean', 'accuracy_std' (std across folds)
            - 'f1_macro_mean', 'f1_macro_std' (std across folds)
            - 'precision_mean', 'recall_mean' (macro-averaged)
            - 'baseline_most_frequent_f1_macro', 'baseline_most_frequent_accuracy'
            - 'baseline_stratified_f1_macro', 'baseline_stratified_accuracy'
    """
    binary_labels = np.asarray(binary_labels).astype(int)
    relation_labels = np.asarray(relation_labels)

    if groups is None:
        group_keys = None
    elif isinstance(groups, np.ndarray) and groups.ndim == 1:
        group_keys = [groups]
    else:
        group_keys = [np.asarray(g) for g in groups]

    results_list = []

    for relation in sorted(np.unique(relation_labels)):
        mask = relation_labels == relation
        n_samples = int(mask.sum())
        if n_samples < min_samples:
            continue

        X = embeddings[mask]
        y = binary_labels[mask]
        n_true = int((y == 1).sum())
        n_false = int((y == 0).sum())

        if min(n_true, n_false) < n_splits:
            warnings.warn(
                f"Skipping relation {relation!r}: minority class has "
                f"{min(n_true, n_false)} examples, fewer than n_splits={n_splits}."
            )
            continue

        if group_keys is None:
            rel_groups = None
            n_groups, max_group_size = n_samples, 1
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            rel_groups = make_groups(*[k[mask] for k in group_keys])
            group_sizes = np.bincount(rel_groups)
            n_groups, max_group_size = len(group_sizes), int(group_sizes.max())
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_scores: Dict[str, list] = {}

        for train_idx, test_idx in cv.split(X, y, rel_groups):
            y_test = y[test_idx]
            if rel_groups is not None:
                assert not set(rel_groups[train_idx]) & set(rel_groups[test_idx])
            if len(np.unique(y[train_idx])) < 2:
                raise ValueError(
                    f"Relation {relation!r}: a training fold contains a single class."
                )

            clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
            if standardize:
                clf = make_pipeline(StandardScaler(), clf)
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[test_idx])

            fold = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            }

            for strategy in BASELINE_STRATEGIES:
                dummy = DummyClassifier(strategy=strategy, random_state=42)
                dummy.fit(X[train_idx], y[train_idx])
                y_dummy = dummy.predict(X[test_idx])
                fold[f"baseline_{strategy}_accuracy"] = accuracy_score(y_test, y_dummy)
                fold[f"baseline_{strategy}_f1_macro"] = f1_score(
                    y_test, y_dummy, average="macro", zero_division=0
                )

            for key, value in fold.items():
                fold_scores.setdefault(key, []).append(value)

        scores = {k: np.asarray(v) for k, v in fold_scores.items()}
        row = {
            "relation": relation,
            "n_samples": n_samples,
            "n_true": n_true,
            "n_false": n_false,
            "cv": "stratified" if group_keys is None else "stratified_group",
            "n_groups": n_groups,
            "max_group_size": max_group_size,
            "accuracy_mean": scores["accuracy"].mean(),
            "accuracy_std": scores["accuracy"].std(),
            "f1_macro_mean": scores["f1_macro"].mean(),
            "f1_macro_std": scores["f1_macro"].std(),
            "precision_mean": scores["precision"].mean(),
            "recall_mean": scores["recall"].mean(),
        }
        for strategy in BASELINE_STRATEGIES:
            for metric in ("f1_macro", "accuracy"):
                key = f"baseline_{strategy}_{metric}"
                row[key] = scores[key].mean()
        results_list.append(row)

    return (
        pd.DataFrame(results_list)
        .sort_values(["n_samples", "relation"], ascending=[False, True])
        .reset_index(drop=True)
    )


def model_comparison_table(
    results_dict: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine per-relation results from multiple models into a single comparison table.

    Works with the output of either `probe_relation_identity` or
    `probe_truth_by_relation`. Only requires the columns 'relation',
    'n_samples' and 'f1_macro_mean'. If the inputs carry the dummy-baseline
    macro-F1 columns (from `probe_truth_by_relation`), they are appended once;
    they depend only on the labels and folds, not on the embeddings.

    Args:
        results_dict: Dict mapping model name -> per-relation results DataFrame

    Returns:
        DataFrame with columns [relation, n_samples, <model>_f1 ..., and, when
        available, baseline_most_frequent_f1, baseline_stratified_f1], sorted
        by n_samples (descending), then relation.
    """
    combined = None
    for model_name, results_df in results_dict.items():
        model_results = results_df[["relation", "n_samples", "f1_macro_mean"]].rename(
            columns={"f1_macro_mean": f"{model_name}_f1"}
        )
        if combined is None:
            combined = model_results
        else:
            combined = combined.merge(model_results, on=["relation", "n_samples"], how="outer")

    baseline_cols = [f"baseline_{s}_f1_macro" for s in BASELINE_STRATEGIES]
    first_df = next(iter(results_dict.values()))
    if all(c in first_df.columns for c in baseline_cols):
        baselines = first_df[["relation"] + baseline_cols].rename(
            columns={c: c.replace("_f1_macro", "_f1") for c in baseline_cols}
        )
        combined = combined.merge(baselines, on="relation", how="left")

    return combined.sort_values(
        ["n_samples", "relation"], ascending=[False, True]
    ).reset_index(drop=True)
