"""
Plackett-Luce Ranking Loss for Listwise Preference Optimization (LPO)

This module implements the Plackett-Luce loss function for training vision-language
models with ranked preference data. The loss encourages the model to assign higher
probabilities to higher-ranked candidates.

Formula (with candidates sorted by true ranking y_1 ≻ y_2 ≻ ... ≻ y_R):
    L_LPO = (1/(R-1)) * sum_{i=0}^{R-2} [log(sum_{j=i}^{R-1} exp(r_j)) - r_i]

Key implementation details:
- Uses torch.logcumsumexp for numerical stability
- Fully vectorized, no Python loops over R
- Supports batched computation
"""

import torch
from typing import Dict


def plackett_luce_loss(rewards: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Compute the Plackett-Luce ranking loss.

    The loss encourages the model to assign higher rewards to higher-ranked candidates.
    Candidates are assumed to be sorted by true ranking (rewards[:, 0] is the best).

    Args:
        rewards: Tensor of shape [B, R] where B is batch size and R is number of candidates.
                 Already sorted by true ranking (rewards[:, 0] corresponds to best candidate).
        reduction: "mean" | "sum" | "none"
                   - "mean": return scalar mean loss over batch
                   - "sum": return scalar sum loss over batch
                   - "none": return [B] loss vector

    Returns:
        Loss tensor. Scalar if reduction != "none", else shape [B].

    Example:
        >>> rewards = torch.tensor([[10.0, 5.0, 0.0, -5.0]])  # Perfect ranking
        >>> loss = plackett_luce_loss(rewards)
        >>> # Loss should be small since rewards match ranking
    """
    B, R = rewards.shape

    if R < 2:
        raise ValueError(f"Need at least 2 candidates for ranking loss, got R={R}")

    # Compute suffix logsumexp: for each position i, compute log(sum_{j=i}^{R-1} exp(r_j))
    # Using logcumsumexp on flipped tensor, then flip back
    suffix_logsumexp = torch.logcumsumexp(rewards.flip(-1), dim=-1).flip(-1)  # [B, R]

    # The loss for each position i (0 to R-2) is: suffix_logsumexp[i] - rewards[i]
    # We only sum over positions 0 to R-2 (the last position R-1 has no candidates below it)
    per_position_loss = suffix_logsumexp[:, :-1] - rewards[:, :-1]  # [B, R-1]

    # Average over positions
    loss_per_sample = per_position_loss.mean(dim=-1)  # [B]

    if reduction == "mean":
        return loss_per_sample.mean()
    elif reduction == "sum":
        return loss_per_sample.sum()
    elif reduction == "none":
        return loss_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Use 'mean', 'sum', or 'none'.")


def compute_pl_rewards(
    policy_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    beta: float
) -> torch.Tensor:
    """
    Compute rewards for Plackett-Luce loss from policy and reference log-probabilities.

    reward = beta * (logp_policy - logp_ref)

    This follows the standard RLHF reward formulation where we measure how much
    the policy deviates from the reference model.

    Args:
        policy_logp: Log-probabilities from the policy model. Any shape.
        ref_logp: Log-probabilities from the reference model. Same shape as policy_logp.
        beta: Temperature/scaling parameter. Higher beta = stronger preference signal.

    Returns:
        Rewards tensor with same shape as inputs.
    """
    return beta * (policy_logp - ref_logp)


def compute_lpo_metrics(rewards: torch.Tensor) -> Dict[str, float]:
    """
    Compute monitoring metrics for LPO training.

    These metrics help track whether the model is learning the correct ranking:
    - mono_ratio: Fraction of samples where top candidate has higher reward than bottom
    - mean_margin: Average difference between top and bottom rewards
    - reward_top1: Mean reward of top-ranked candidates
    - reward_bottom1: Mean reward of bottom-ranked candidates
    - reward_std: Standard deviation of all rewards

    Args:
        rewards: Tensor of shape [B, R] with rewards for each candidate.
                 Assumed sorted by true ranking ([:, 0] is best).

    Returns:
        Dictionary with metric names and values.
    """
    B, R = rewards.shape

    # Monotonicity ratio: how often is top reward > bottom reward
    top_rewards = rewards[:, 0]      # [B]
    bottom_rewards = rewards[:, -1]  # [B]
    mono_ratio = (top_rewards > bottom_rewards).float().mean().item()

    # Mean margin between top and bottom
    mean_margin = (top_rewards - bottom_rewards).mean().item()

    # Individual reward statistics
    reward_top1 = top_rewards.mean().item()
    reward_bottom1 = bottom_rewards.mean().item()
    reward_std = rewards.std().item()

    return {
        "lpo/mono_ratio": mono_ratio,
        "lpo/mean_margin": mean_margin,
        "lpo/reward_top1": reward_top1,
        "lpo/reward_bottom1": reward_bottom1,
        "lpo/reward_std": reward_std,
    }
