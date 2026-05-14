"""
Evaluation metrics for Listwise Preference Optimization.

Implements ranking quality metrics including Kendall's Tau,
NDCG, and monotonicity measures.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import kendalltau


def compute_kendall_tau(predicted_ranks: np.ndarray, true_ranks: np.ndarray) -> float:
    """
    Compute Kendall's Tau rank correlation coefficient.

    Args:
        predicted_ranks: Array of predicted rankings [N, R].
        true_ranks: Array of ground-truth rankings [N, R].

    Returns:
        Average Kendall's Tau across all samples.
    """
    taus = []
    for pred, true in zip(predicted_ranks, true_ranks):
        tau, _ = kendalltau(pred, true)
        if not np.isnan(tau):
            taus.append(tau)
    return np.mean(taus) if taus else 0.0


def compute_ndcg(predicted_scores: np.ndarray, k: int = 4) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@k).

    Assumes the ground-truth relevance is [R, R-1, ..., 1] (position 0 is most relevant).

    Args:
        predicted_scores: Array of predicted scores [N, R].
        k: Number of top positions to consider.

    Returns:
        Average NDCG@k across all samples.
    """
    N, R = predicted_scores.shape
    k = min(k, R)

    # Ground-truth relevance: position 0 has highest relevance
    true_relevance = np.arange(R, 0, -1)  # [R, R-1, ..., 1]

    ndcg_scores = []
    for i in range(N):
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(-predicted_scores[i])

        # DCG
        dcg = 0.0
        for j in range(k):
            rel = true_relevance[sorted_indices[j]]
            dcg += (2**rel - 1) / np.log2(j + 2)

        # Ideal DCG (perfect ranking)
        ideal_relevance = np.sort(true_relevance)[::-1]
        idcg = 0.0
        for j in range(k):
            rel = ideal_relevance[j]
            idcg += (2**rel - 1) / np.log2(j + 2)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)


def compute_monotonicity(rewards: np.ndarray) -> float:
    """
    Compute the fraction of samples with strictly monotonically decreasing rewards.

    A sample is monotonic if: reward[0] > reward[1] > ... > reward[R-1]

    Args:
        rewards: Array of reward scores [N, R], sorted by true ranking.

    Returns:
        Fraction of samples with perfect monotonicity.
    """
    N, R = rewards.shape
    monotonic_count = 0

    for i in range(N):
        is_monotonic = all(rewards[i, j] > rewards[i, j + 1] for j in range(R - 1))
        if is_monotonic:
            monotonic_count += 1

    return monotonic_count / N


def compute_pairwise_accuracy(rewards: np.ndarray) -> float:
    """
    Compute pairwise ranking accuracy.

    For each pair (i, j) where i < j (i.e., i should have higher reward),
    check if reward[i] > reward[j].

    Args:
        rewards: Array of reward scores [N, R], sorted by true ranking.

    Returns:
        Fraction of correctly ordered pairs.
    """
    N, R = rewards.shape
    correct = 0
    total = 0

    for sample in range(N):
        for i in range(R):
            for j in range(i + 1, R):
                total += 1
                if rewards[sample, i] > rewards[sample, j]:
                    correct += 1

    return correct / total if total > 0 else 0.0


def evaluate_ranking(
    rewards: np.ndarray,
    n_rank: int = 4,
) -> Dict[str, float]:
    """
    Compute all ranking evaluation metrics.

    Args:
        rewards: Array of shape [N, R] with predicted reward scores.
                 Columns are ordered by true ranking (col 0 = best).
        n_rank: Number of ranked candidates.

    Returns:
        Dictionary with all evaluation metrics.
    """
    N, R = rewards.shape
    assert R == n_rank, f"Expected {n_rank} candidates, got {R}"

    # Predicted ranks: argsort descending scores
    predicted_ranks = np.argsort(-rewards, axis=1).argsort(axis=1)
    true_ranks = np.tile(np.arange(R), (N, 1))

    metrics = {
        "kendall_tau": compute_kendall_tau(predicted_ranks, true_ranks),
        "ndcg@4": compute_ndcg(rewards, k=4),
        "monotonicity": compute_monotonicity(rewards),
        "pairwise_accuracy": compute_pairwise_accuracy(rewards),
    }

    # Top-1 accuracy: is the highest-reward candidate actually rank 0?
    top1_predictions = np.argmax(rewards, axis=1)
    metrics["top1_accuracy"] = np.mean(top1_predictions == 0)

    return metrics


def print_evaluation_report(metrics: Dict[str, float]) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("  LPO Evaluation Report")
    print("=" * 60)
    print(f"  Kendall's Tau:       {metrics['kendall_tau']:.4f}")
    print(f"  NDCG@4:             {metrics['ndcg@4']:.4f}")
    print(f"  Monotonicity:       {metrics['monotonicity']:.4f} ({metrics['monotonicity']*100:.1f}%)")
    print(f"  Pairwise Accuracy:  {metrics['pairwise_accuracy']:.4f}")
    print(f"  Top-1 Accuracy:     {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.1f}%)")
    print("=" * 60 + "\n")
