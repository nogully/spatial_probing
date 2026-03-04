"""Probing classifiers for spatial relation representations."""

from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.preprocessing import LabelEncoder


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


def probe_by_relation_type(
    embeddings: np.ndarray,
    relation_labels: np.ndarray,
    binary_labels: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
    min_samples: int = 20,
) -> pd.DataFrame:
    """
    Train a separate binary probe for each relation type.

    For each spatial relation type (e.g., 'above', 'left'), train a binary classifier:
        Label = 1 if this relation type, 0 otherwise

    This tests whether the relation type is linearly decodable from embeddings.

    Args:
        embeddings: (N, dim) array of embeddings
        relation_labels: (N,) array of relation type strings
        binary_labels: (N,) array of True/False (for sampling stats only)
        n_splits: Number of CV folds
        C: Regularization hyperparameter
        min_samples: Skip relations with fewer than this many examples

    Returns:
        DataFrame with columns:
            - 'relation': relation type
            - 'n_samples': number of examples
            - 'accuracy_mean', 'accuracy_std'
            - 'f1_macro_mean', 'f1_macro_std'
            - 'precision_mean', 'recall_mean'
    """
    # Get unique relations
    unique_relations = np.unique(relation_labels)
    unique_relations = sorted([r for r in unique_relations])

    results_list = []

    for relation in unique_relations:
        # Create binary labels for this relation
        is_this_relation = (relation_labels == relation).astype(int)

        # Skip if too few samples
        if np.sum(is_this_relation) < min_samples:
            continue

        # Train probe
        probe_results = train_probe(embeddings, is_this_relation, n_splits=n_splits, C=C)

        # Add to results
        result_row = {
            "relation": relation,
            "n_samples": np.sum(is_this_relation),
            "n_positive": np.sum((is_this_relation == 1) & (binary_labels == 1)),
            "n_negative": np.sum((is_this_relation == 1) & (binary_labels == 0)),
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


def probe_weight_analysis(
    embeddings: np.ndarray,
    labels: np.ndarray,
    dim_names: Optional[List[str]] = None,
    top_k: int = 10,
) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """
    Analyze probe weight to find most diagnostic dimensions.

    Train a logistic regression probe without CV and extract learned weights.
    Returns the top-k dimensions by weight magnitude.

    Args:
        embeddings: (N, dim) array of embeddings
        labels: (N,) array of binary labels
        dim_names: Optional list of dimension names (e.g., ["D0", "D1", ...])
        top_k: Number of top dimensions to return

    Returns:
        Tuple of (coef array, list of (name/index, weight) tuples for top-k dims)
    """
    # Train on full data (not CV) for weight extraction
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(embeddings, labels)

    # Get weights (shape: (1, dim) for binary, (n_classes, dim) for multiclass)
    coef = clf.coef_
    if coef.shape[0] == 1:
        # Binary classification
        weights = coef[0]
    else:
        # Multiclass: use magnitude across all classes
        weights = np.abs(coef).max(axis=0)

    # Get top-k by magnitude
    top_indices = np.argsort(np.abs(weights))[-top_k:][::-1]

    if dim_names is None:
        dim_names = [f"D{i}" for i in range(len(weights))]

    top_dims = [(dim_names[i], weights[i]) for i in top_indices]

    return coef, top_dims


def model_comparison_table(
    results_dict: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine per-relation results from multiple models into a single comparison table.

    Args:
        results_dict: Dict mapping model name -> DataFrame from probe_by_relation_type

    Returns:
        DataFrame with columns [relation, model1_f1, model2_f1, ...] sorted by mean difference
    """
    # Merge all model results on relation
    combined = None
    for model_name, results_df in results_dict.items():
        model_results = results_df[["relation", "f1_macro_mean"]].copy()
        model_results = model_results.rename(columns={"f1_macro_mean": f"{model_name}_f1"})

        if combined is None:
            combined = model_results
        else:
            combined = combined.merge(model_results, on="relation", how="outer")

    # Sort by mean F1 across models
    model_columns = [f"{m}_f1" for m in results_dict.keys()]
    combined["mean_f1"] = combined[model_columns].mean(axis=1)
    combined = combined.sort_values("mean_f1", ascending=False)

    return combined.drop(columns=["mean_f1"])
