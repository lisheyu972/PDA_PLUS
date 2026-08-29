#!/bin/bash

set -e

GENERATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$GENERATION_ROOT"

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY
export HF_ENDPOINT=https://hf-mirror.com

export KLEIN_4B_BASE_MODEL_PATH="${KLEIN_4B_BASE_MODEL_PATH:-$GENERATION_ROOT/models/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
export AE_MODEL_PATH="${AE_MODEL_PATH:-$GENERATION_ROOT/models/FLUX.2-ae/ae.safetensors}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$GENERATION_ROOT/checkpoints/flux2-lora-insertion-seg/lora_epoch20}"
DATA_ROOT="${DATA_ROOT:-$GENERATION_ROOT/data/ISAID_DOTA_processed_val}"
OUTPUT_DIR="${OUTPUT_DIR:-$GENERATION_ROOT/output/inference_insertion_seg_mata_latent}"
NUM_STEPS=50

MATA_SCALE=0.005
MATA_ITERS=1
MATA_START=0.7
MATA_END=0.9
MATA_STRIDE=1
NUM_SCALES=3
PATCH_GRID_SIZES="1,2,4"
MIN_PATCH_PIXELS=4
ALPHA_SEM=0.0
SINKHORN_BLUR=0.05
SINKHORN_ITERS=50
LOCAL_DILATION=24
OBJ_EROSION=0
ROLLBACK_THRESHOLD=1.5
SEMANTIC_MASK_DIR=""
LOG_EVERY=0

CMD="PYTHONPATH=$GENERATION_ROOT/src python $GENERATION_ROOT/scripts/gen_infer/inference_insertion_mata_latent.py \
    --model_name flux.2-klein-base-4b \
    --checkpoint_dir $CHECKPOINT_DIR \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --num_steps $NUM_STEPS \
    --bg_folder Background_Erased_WithSeg \
    --crop_folder Crops \
    --mata_scale $MATA_SCALE \
    --mata_iters $MATA_ITERS \
    --mata_start $MATA_START \
    --mata_end $MATA_END \
    --mata_stride $MATA_STRIDE \
    --num_scales $NUM_SCALES \
    --patch_grid_sizes $PATCH_GRID_SIZES \
    --min_patch_pixels $MIN_PATCH_PIXELS \
    --alpha_sem $ALPHA_SEM \
    --sinkhorn_blur $SINKHORN_BLUR \
    --sinkhorn_iters $SINKHORN_ITERS \
    --local_dilation $LOCAL_DILATION \
    --obj_erosion $OBJ_EROSION \
    --rollback_threshold $ROLLBACK_THRESHOLD \
    --log_every $LOG_EVERY \
    --device cuda:0"

if [ -n "$SEMANTIC_MASK_DIR" ]; then
    CMD="$CMD --semantic_mask_dir $SEMANTIC_MASK_DIR"
fi

eval "$CMD"
