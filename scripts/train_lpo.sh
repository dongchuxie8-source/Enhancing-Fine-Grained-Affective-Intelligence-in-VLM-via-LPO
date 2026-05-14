#!/bin/bash
# Stage 2: Listwise Preference Optimization (LPO)
# Requires a reference model from Stage 1 (SFT).

python train.py \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --ref_model_path outputs/sft \
    --data_path data/train_lpo.json \
    --image_folder data/images \
    --output_dir outputs/lpo \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --beta 0.1 \
    --n_rank 4 \
    --bf16 \
    --logging_steps 10 \
    --save_steps 100 \
    --report_to wandb \
    --seed 42
