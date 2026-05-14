"""
LPO Trainer: Listwise Preference Optimization Trainer.

Extends HuggingFace's Trainer to support listwise preference optimization
using the Plackett-Luce ranking loss.
"""

import logging
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

from .lpo_loss import plackett_luce_loss, compute_pl_rewards, compute_lpo_metrics

logger = logging.getLogger(__name__)


class LPOTrainer(Trainer):
    """
    Trainer for Listwise Preference Optimization (LPO).

    This trainer computes the Plackett-Luce loss over ranked candidate responses.
    It requires a reference model to compute implicit rewards as:
        reward = beta * (log_pi_policy - log_pi_ref)

    Args:
        ref_model: The frozen reference model (SFT checkpoint).
        beta: Temperature parameter for reward computation.
        n_rank: Number of ranked candidates per sample.
        **kwargs: Additional arguments passed to HuggingFace Trainer.
    """

    def __init__(
        self,
        ref_model: Optional[nn.Module] = None,
        beta: float = 0.1,
        n_rank: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.n_rank = n_rank

        # Freeze reference model
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    def _get_sequence_logp(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sequence log-probabilities.

        Args:
            model: The language model.
            input_ids: [B*R, L] input token IDs.
            attention_mask: [B*R, L] attention mask.
            labels: [B*R, L] labels with IGNORE_INDEX for non-target tokens.

        Returns:
            Tensor of shape [B*R] with average log-probability per sequence.
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [B*R, L, V]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Compute per-token log-probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)  # [B*R, L-1, V]

        # Gather log-probs for target tokens
        # Replace IGNORE_INDEX with 0 for gathering (will be masked out)
        gather_labels = shift_labels.clone()
        ignore_mask = (shift_labels == -100)
        gather_labels[ignore_mask] = 0

        per_token_logp = log_probs.gather(
            dim=-1, index=gather_labels.unsqueeze(-1)
        ).squeeze(-1)  # [B*R, L-1]

        # Zero out ignored positions
        per_token_logp[ignore_mask] = 0.0

        # Average over valid tokens per sequence
        valid_token_count = (~ignore_mask).sum(dim=-1).clamp(min=1)  # [B*R]
        sequence_logp = per_token_logp.sum(dim=-1) / valid_token_count  # [B*R]

        return sequence_logp

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute the LPO (Plackett-Luce) loss.

        Steps:
        1. Forward pass through policy model to get log-probs for all B*R sequences.
        2. Forward pass through reference model (no grad) to get ref log-probs.
        3. Compute rewards = beta * (policy_logp - ref_logp).
        4. Reshape rewards to [B, R] and compute Plackett-Luce loss.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        n_rank = inputs.get("n_rank", self.n_rank)

        # Policy model forward
        policy_logp = self._get_sequence_logp(model, input_ids, attention_mask, labels)

        # Reference model forward (no gradient)
        with torch.no_grad():
            if self.ref_model is not None:
                ref_logp = self._get_sequence_logp(
                    self.ref_model, input_ids, attention_mask, labels
                )
            else:
                # If no ref model, use zeros (equivalent to beta * policy_logp)
                ref_logp = torch.zeros_like(policy_logp)

        # Compute rewards
        rewards = compute_pl_rewards(policy_logp, ref_logp, self.beta)  # [B*R]

        # Reshape to [B, R]
        batch_size = input_ids.shape[0] // n_rank
        rewards_matrix = rewards.view(batch_size, n_rank)  # [B, R]

        # Compute Plackett-Luce loss
        loss = plackett_luce_loss(rewards_matrix, reduction="mean")

        # Log metrics
        if self.state.global_step % self.args.logging_steps == 0:
            metrics = compute_lpo_metrics(rewards_matrix.detach())
            for key, value in metrics.items():
                self.log({key: value})

        if return_outputs:
            return loss, {"rewards": rewards_matrix.detach()}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction step for evaluation."""
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss = self.compute_loss(model, inputs)

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, None, None)
