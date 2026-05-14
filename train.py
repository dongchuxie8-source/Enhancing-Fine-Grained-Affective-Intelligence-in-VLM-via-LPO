"""
Main training script for Listwise Preference Optimization (LPO).

Usage:
    python train.py --config config/lpo_config.yaml
    python train.py --data_path data/train_lpo.json --output_dir outputs/lpo
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from copy import deepcopy

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.lpo_dataset import LPODataset, LPODataCollator
from src.lpo_trainer import LPOTrainer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LPO Training for Emotion VLM")

    # Model
    parser.add_argument("--model_name_or_path", type=str, default="liuhaotian/llava-v1.5-7b",
                        help="Path to pretrained model or model identifier")
    parser.add_argument("--ref_model_path", type=str, default=None,
                        help="Path to reference model (SFT checkpoint). If None, uses base model.")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data JSON file")
    parser.add_argument("--image_folder", type=str, default=None,
                        help="Root folder for images")
    parser.add_argument("--n_rank", type=int, default=4,
                        help="Number of ranked candidates per sample")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")

    # LPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Temperature parameter for reward computation")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "v_proj"],
                        help="Target modules for LoRA")

    # Training
    parser.add_argument("--output_dir", type=str, default="outputs/lpo",
                        help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Checkpoint save frequency")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 mixed precision")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Misc
    parser.add_argument("--report_to", type=str, default="wandb",
                        help="Reporting backend (wandb, tensorboard, none)")

    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_model_and_tokenizer(args):
    """Load the base model and tokenizer."""
    logger.info(f"Loading model from {args.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
    )

    return model, tokenizer


def apply_lora(model, args):
    """Apply LoRA adapters to the model."""
    logger.info(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def main():
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("  Listwise Preference Optimization (LPO) Training")
    logger.info("=" * 60)
    logger.info(f"  Beta: {args.beta}")
    logger.info(f"  N_rank: {args.n_rank}")
    logger.info(f"  LoRA rank: {args.lora_r}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # Create reference model (frozen copy before LoRA)
    logger.info("Creating reference model...")
    if args.ref_model_path:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            device_map="auto",
        )
    else:
        ref_model = deepcopy(model)

    # Apply LoRA to policy model
    model = apply_lora(model, args)

    # Load dataset
    logger.info(f"Loading data from {args.data_path}")
    train_dataset = LPODataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        image_folder=args.image_folder,
        n_rank=args.n_rank,
        max_length=args.max_length,
    )

    data_collator = LPODataCollator(tokenizer=tokenizer, n_rank=args.n_rank)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        report_to=args.report_to,
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=4,
    )

    # Initialize LPO Trainer
    trainer = LPOTrainer(
        ref_model=ref_model,
        beta=args.beta,
        n_rank=args.n_rank,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting LPO training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
