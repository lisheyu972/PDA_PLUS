#!/bin/bash

set -e

GENERATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GENERATION_ROOT"

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
export HF_ENDPOINT=https://hf-mirror.com

MODEL_NAME="${MODEL_NAME:-flux.2-klein-base-4b}"
OUTPUT_DIR="${OUTPUT_DIR:-$GENERATION_ROOT/checkpoints/flux2-lora-insertion-seg}"
ISAID_DATA_ROOT="${ISAID_DATA_ROOT:-$GENERATION_ROOT/data/ISAID_DOTA_processed}"
SAMARS_DATA_ROOT="${SAMARS_DATA_ROOT:-$GENERATION_ROOT/data/samars}"
GPUS="${GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29504}"

export KLEIN_4B_BASE_MODEL_PATH="${KLEIN_4B_BASE_MODEL_PATH:-$GENERATION_ROOT/models/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
export AE_MODEL_PATH="${AE_MODEL_PATH:-$GENERATION_ROOT/models/FLUX.2-ae/ae.safetensors}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p "$OUTPUT_DIR"

if [ "$GPUS" -gt 1 ]; then
    PYTHONPATH="$GENERATION_ROOT/src" torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" \
        "$GENERATION_ROOT/scripts/gen_train/train_lora_insertion_seg.py" \
        --model_name "$MODEL_NAME" \
        --data_roots "$ISAID_DATA_ROOT" "$SAMARS_DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 8 \
        --num_epochs 20 \
        --lr 1e-4 \
        --lora_rank 16 \
        --lora_alpha 32 \
        --save_every 2000 \
        --log_every 10 \
        --num_workers 4 \
        --val_ratio 0.05 \
        --cfg_guidance 4.0 \
        --mask_loss_weight 10 \
        --wandb_project flux2-lora-insertion \
        --wandb_run_name insertion-seg
else
    PYTHONPATH="$GENERATION_ROOT/src" python "$GENERATION_ROOT/scripts/gen_train/train_lora_insertion_seg.py" \
        --model_name "$MODEL_NAME" \
        --data_roots "$ISAID_DATA_ROOT" "$SAMARS_DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 4 \
        --num_epochs 20 \
        --lr 1e-4 \
        --lora_rank 16 \
        --lora_alpha 32 \
        --save_every 2000 \
        --log_every 10 \
        --num_workers 4 \
        --val_ratio 0.05 \
        --cfg_guidance 4.0 \
        --mask_loss_weight 5.0 \
        --no_wandb
fi
