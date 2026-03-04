"""Representational analysis: RSA, RDM, and related methods."""

from typing import Tuple, Optional
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import pandas as pd


def compute_rdm(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Compute Representational Dissimilarity Matrix (RDM).

    The RDM encodes pairwise dissimilarities between examples in the embedding space.
    For L2-normalized embeddings, cosine dissimilarity = 1 - cosine_similarity.

    Args:
        embeddings: (N, dim) array of L2-normalized embeddings
        metric: Distance metric (default 'cosine')

    Returns:
        (N, N) symmetric RDM where RDM[i,j] = 1 - cosine_similarity(emb[i], emb[j])

    Reference: Kriegeskorte et al. (2008) "Representational Similarity Analysis"
    https://doi.org/10.3389/neuro.06.004.2008
    """
    # For L2-normalized embeddings, cosine similarity = dot product
    # Cosine dissimilarity = 1 - cosine_similarity
    similarities = embeddings @ embeddings.T  # (N, N)
    rdm = 1 - similarities
    return rdm


def rsa_correlation(rdm1: np.ndarray, rdm2: np.ndarray, method: str = "spearman") -> float:
    """
    Compute correlation between two RDMs using only upper triangle.

    Compares the structure of representational spaces without requiring labels.
    High correlation indicates models structure examples similarly.

    Args:
        rdm1: (N, N) RDM from model 1
        rdm2: (N, N) RDM from model 2
        method: 'spearman' (recommended) or 'pearson'

    Returns:
        Correlation coefficient (float in [-1, 1])

    Reference: Kriegeskorte et al. (2008), adopted for NLP by Abdou et al. (2019)
    """
    assert rdm1.shape == rdm2.shape, "RDMs must have same shape"

    # Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(rdm1.shape[0], k=1)
    rdm1_flat = rdm1[triu_indices]
    rdm2_flat = rdm2[triu_indices]

    if method == "spearman":
        corr, _ = spearmanr(rdm1_flat, rdm2_flat)
    elif method == "pearson":
        corr = np.corrcoef(rdm1_flat, rdm2_flat)[0, 1]
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(corr)


def balanced_subsample(
    embeddings: np.ndarray,
    relation_labels: np.ndarray,
    n_per_relation: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample embeddings balanced across relation types.

    Useful for RSA to avoid bias toward frequent relation types.

    Args:
        embeddings: (N, dim) array
        relation_labels: (N,) array of relation type strings
        n_per_relation: Number of examples to sample per relation type
        random_state: Random seed

    Returns:
        (sampled_embeddings, sampled_indices, sampled_relations)
    """
    np.random.seed(random_state)

    sampled_indices = []
    unique_relations = np.unique(relation_labels)

    for relation in unique_relations:
        indices = np.where(relation_labels == relation)[0]
        # Sample up to n_per_relation examples
        n_sample = min(len(indices), n_per_relation)
        sampled = np.random.choice(indices, size=n_sample, replace=False)
        sampled_indices.extend(sampled)

    sampled_indices = np.array(sampled_indices)
    sampled_embeddings = embeddings[sampled_indices]
    sampled_relations = relation_labels[sampled_indices]

    return sampled_embeddings, sampled_indices, sampled_relations


def odd_one_out_ranking(
    embeddings: np.ndarray,
    relation_labels: np.ndarray,
    n_trials: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Odd-one-out ranking task: measure how well model can identify related items.

    For each trial, construct a triplet (A, B, foil) where:
    - A and B share the same relation type
    - foil has a different relation type
    - Measure: does the model rank A and B as more similar than A and foil?

    Args:
        embeddings: (N, dim) array of L2-normalized embeddings
        relation_labels: (N,) array of relation type strings
        n_trials: Number of triplet trials
        random_state: Random seed

    Returns:
        Dict with:
            - 'accuracy': proportion of correct rankings
            - 'n_trials': number of trials
            - 'correct': number of correct trials
    """
    np.random.seed(random_state)

    unique_relations = np.unique(relation_labels)
    unique_relations = unique_relations[
        [len(np.where(relation_labels == r)[0]) >= 2 for r in unique_relations]
    ]  # keep relations with >= 2 examples

    correct = 0

    for _ in range(n_trials):
        # Pick a relation with >= 2 examples
        relation = np.random.choice(unique_relations)
        indices = np.where(relation_labels == relation)[0]

        if len(indices) < 2:
            continue

        # Pick two examples from same relation
        a_idx, b_idx = np.random.choice(indices, size=2, replace=False)

        # Pick a foil from a different relation
        other_relations = unique_relations[unique_relations != relation]
        foil_relation = np.random.choice(other_relations)
        foil_indices = np.where(relation_labels == foil_relation)[0]
        foil_idx = np.random.choice(foil_indices)

        # Compute similarities (cosine, so higher = more similar)
        sim_a_b = np.dot(embeddings[a_idx], embeddings[b_idx])
        sim_a_foil = np.dot(embeddings[a_idx], embeddings[foil_idx])

        # Check if A is more similar to B than to foil
        if sim_a_b > sim_a_foil:
            correct += 1

    accuracy = correct / n_trials

    return {
        "accuracy": accuracy,
        "n_trials": n_trials,
        "correct": correct,
    }


def rsa_analysis_pairwise(
    embeddings_dict: dict, relation_labels: np.ndarray, subsample_n: int = 500
) -> Tuple[pd.DataFrame, dict]:
    """
    Full RSA analysis: compute RDM correlations between all model pairs.

    Args:
        embeddings_dict: Dict mapping model name -> (N, dim) embeddings
        relation_labels: (N,) array of relation types for balanced sampling
        subsample_n: Size of balanced subsample for RDM computation

    Returns:
        (correlation_df, rdm_dict) where:
            - correlation_df: (n_models, n_models) DataFrame of pairwise correlations
            - rdm_dict: Dict mapping model name -> RDM
    """
    model_names = list(embeddings_dict.keys())

    # Subsample for computational efficiency
    sampled_embeddings = {}
    for model_name, embeddings in embeddings_dict.items():
        subsampled, _, _ = balanced_subsample(
            embeddings, relation_labels, n_per_relation=subsample_n // len(np.unique(relation_labels))
        )
        sampled_embeddings[model_name] = subsampled

    # Compute RDMs
    rdm_dict = {}
    for model_name, embeddings in sampled_embeddings.items():
        rdm_dict[model_name] = compute_rdm(embeddings)

    # Compute correlations
    n_models = len(model_names)
    corr_matrix = np.zeros((n_models, n_models))

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                corr = rsa_correlation(rdm_dict[model1], rdm_dict[model2])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

    corr_df = pd.DataFrame(corr_matrix, index=model_names, columns=model_names)

    return corr_df, rdm_dict


def probe_weight_top_dimensions(
    model_coefs: dict, embedding_dims: dict, top_k: int = 10
) -> pd.DataFrame:
    """
    Summarize most diagnostic dimensions across models.

    Args:
        model_coefs: Dict mapping model name -> (n_relations, dim) coefficient array
        embedding_dims: Dict mapping model name -> embedding dimension
        top_k: Number of top dimensions to show per model

    Returns:
        DataFrame with model, rank, dimension, weight columns
    """
    rows = []
    for model_name, coefs in model_coefs.items():
        # Compute weight magnitude across all relations
        weight_magnitudes = np.abs(coefs).max(axis=0)
        top_indices = np.argsort(weight_magnitudes)[-top_k:][::-1]

        for rank, dim_idx in enumerate(top_indices, 1):
            rows.append(
                {
                    "model": model_name,
                    "rank": rank,
                    "dimension": f"D{dim_idx}",
                    "weight_magnitude": weight_magnitudes[dim_idx],
                }
            )

    return pd.DataFrame(rows)
