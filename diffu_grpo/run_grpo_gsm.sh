#!/bin/bash
set -euo pipefail

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Work from the script's directory
cd "$(dirname "$0")"

# Create logs directory if it doesn't exist
mkdir -p logs

# Hyperparameters (override via env before running if needed, e.g. BLOCK_LENGTH=64)
TEMPERATURE=1.0
BLOCK_LENGTH=32
MAX_COMPLETION_LENGTH=256

NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -eq 4 ]; then
    NUM_GENERATIONS=64
    PER_DEVICE_TRAIN_BATCH_SIZE=16
    LOSS_CHUNK_DIVISOR=4
elif [ "$NUM_GPUS" -eq 8 ]; then
    NUM_GENERATIONS=128
    PER_DEVICE_TRAIN_BATCH_SIZE=16
    LOSS_CHUNK_DIVISOR=2
else
    echo "Error: expected 4 or 8 GPUs, got $NUM_GPUS" >&2
    exit 1
fi

LEARNING_RATE=1.25e-6
ADVANTAGE_MIN_CLIP=0.0
BETA=0.0

RUN_NAME="gsm8k_new_T${TEMPERATURE}_bl${BLOCK_LENGTH}_ng${NUM_GENERATIONS}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}_lr${LEARNING_RATE}_complen${MAX_COMPLETION_LENGTH}_clip${ADVANTAGE_MIN_CLIP}_beta${BETA}_chunk_${LOSS_CHUNK_DIVISOR}"
OUTPUT_DIR="./checkpoints/${RUN_NAME}"

: "${MODEL_PATH:?MODEL_PATH must be set (export MODEL_PATH=/path/to/checkpoint-or-hf-repo)}"

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "${NUM_GPUS}" \
    --main_process_port 12446 \
    diffu_grpo_train.py \
    --config sbatch_scripts/train.yaml \
    --model_path "${MODEL_PATH}" \
    --dataset "gsm8k" \
    --run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --temperature ${TEMPERATURE} \
    --num_iterations 1 \
    --max_steps 10000 \
    --beta ${BETA} \
    --normalize true \
    --scale 30.0 \
    --use_scheduler false \
    --max_prompt_length 256 \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --block_length ${BLOCK_LENGTH} \
    --learning_rate ${LEARNING_RATE} \
    --advantage_min_clip ${ADVANTAGE_MIN_CLIP} \
    --freeze_unmasking_head false \
    --num_generations ${NUM_GENERATIONS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --rollout_mode training \
    --scale_reward none \
    --loss_chunk_divisor ${LOSS_CHUNK_DIVISOR} \
    --prompt_mode "non-thinking"
