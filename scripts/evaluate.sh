#!/bin/bash
# Evaluate the LPO-trained model on the test set.

python evaluate.py \
    --model_path outputs/lpo \
    --base_model_path liuhaotian/llava-v1.5-7b \
    --ref_model_path outputs/sft \
    --data_path data/test_lpo.json \
    --image_folder data/images \
    --n_rank 4 \
    --beta 0.1 \
    --batch_size 8 \
    --output_file outputs/eval_results.json
