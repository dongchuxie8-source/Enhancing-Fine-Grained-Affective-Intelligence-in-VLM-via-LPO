"""
Evaluation script for LPO-trained models.

Usage:
    python evaluate.py --model_path outputs/lpo --data_path data/sample_data.json
"""

import os
import json
import argparse
import logging

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.lpo_dataset import LPODataset, LPODataCollator
from src.lpo_loss import compute_pl_rewards
from src.evaluate import evaluate_ranking, print_evaluation_report

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LPO model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained LPO model")
    parser.add_argument("--base_model_path", type=str, default="liuhaotian/llava-v1.5-7b",
                        help="Path to base model (for LoRA)")
    parser.add_argument("--ref_model_path", type=str, default=None,
                        help="Path to reference model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to evaluation data JSON")
    parser.add_argument("--image_folder", type=str, default=None,
                        help="Root folder for images")
    parser.add_argument("--n_rank", type=int, default=4,
                        help="Number of ranked candidates")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Temperature parameter")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Evaluation batch size")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save evaluation results JSON")
    return parser.parse_args()


@torch.no_grad()
def compute_rewards_for_dataset(model, ref_model, dataloader, beta, device):
    """Compute rewards for all samples in the dataset."""
    all_rewards = []

    model.eval()
    if ref_model is not None:
        ref_model.eval()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        n_rank = batch["n_rank"]

        # Policy forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        gather_labels = shift_labels.clone()
        ignore_mask = (shift_labels == -100)
        gather_labels[ignore_mask] = 0

        per_token_logp = log_probs.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
        per_token_logp[ignore_mask] = 0.0
        valid_count = (~ignore_mask).sum(dim=-1).clamp(min=1)
        policy_logp = per_token_logp.sum(dim=-1) / valid_count

        # Reference forward
        if ref_model is not None:
            ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits
            ref_shift_logits = ref_logits[:, :-1, :].contiguous()
            ref_log_probs = torch.nn.functional.log_softmax(ref_shift_logits, dim=-1)
            ref_per_token_logp = ref_log_probs.gather(dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
            ref_per_token_logp[ignore_mask] = 0.0
            ref_logp = ref_per_token_logp.sum(dim=-1) / valid_count
        else:
            ref_logp = torch.zeros_like(policy_logp)

        rewards = compute_pl_rewards(policy_logp, ref_logp, beta)
        rewards_matrix = rewards.view(-1, n_rank)
        all_rewards.append(rewards_matrix.cpu().numpy())

    return np.concatenate(all_rewards, axis=0)


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    # Try loading as LoRA model first, fall back to full model
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, args.model_path)
        logger.info("Loaded LoRA model")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        logger.info("Loaded full model")

    # Reference model
    ref_model = None
    if args.ref_model_path:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

    device = next(model.parameters()).device

    # Load dataset
    logger.info(f"Loading evaluation data from {args.data_path}")
    dataset = LPODataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        image_folder=args.image_folder,
        n_rank=args.n_rank,
    )

    collator = LPODataCollator(tokenizer=tokenizer, n_rank=args.n_rank)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False
    )

    # Compute rewards
    logger.info("Computing rewards...")
    rewards = compute_rewards_for_dataset(model, ref_model, dataloader, args.beta, device)

    # Evaluate
    logger.info(f"Evaluating {rewards.shape[0]} samples...")
    metrics = evaluate_ranking(rewards, n_rank=args.n_rank)
    print_evaluation_report(metrics)

    # Save results
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
