#!/bin/bash
# Stage 1: Supervised Fine-Tuning (SFT)
# This produces the reference model for LPO training.

python train.py \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --data_path data/train_lpo.json \
    --image_folder data/images \
    --output_dir outputs/sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lora_r 8 \
    --lora_alpha 16 \
    --bf16 \
    --logging_steps 10 \
    --save_steps 200 \
    --beta 0.0 \
    --report_to wandb \
    --seed 42
