#!/bin/bash

# Training config
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 
export DIFFUSERS_OFFLINE=1 
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_P2P_DISABLE=1
BASE_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
DATASET="data/CosmicManHQ-1.0/label"
RUN_NAME="cosmicman_sdxl"
OUT_DIR="output"

accelerate launch --main_process_port 27255  --num_processes 1 --config_file  ./configs/accelerate_config.yaml \
    train_sdxl.py \
    --pretrained_model_name_or_path=$BASE_MODEL \
    --dataset=$DATASET \
    --output_dir=$OUT_DIR/$RUN_NAME \
    --use_ema \
    --seed=41 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=100000 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 --learning_rate=1e-5 \
    --bucket_side_min=768 \
    --bucket_side_max=1536 \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing \
    --noise_offset=0.05 \
    --ucg=0.1 \
    --mixed_precision=bf16 \
    --hola_loss_weight=1.0 \
    --attn_map_size=16 \
    --dropout_rate_of_parsing=0.1 \
    --accelerators_save_steps=2000 \
    --checkpointing_save_steps=1000 \
    --checkpoints_total_limit=3 \
