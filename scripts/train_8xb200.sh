#!/bin/bash
# train_8xb200.sh — Launch GR00T N1.6 post-training on 8x B200 GPUs
#
# Usage:
#   bash scripts/train_8xb200.sh [options]
#
# First run scripts/prepare.sh to download weights and validate data.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Defaults (override via env vars or CLI flags)
# ---------------------------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-8}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$REPO_ROOT/checkpoints/GR00T-N1.6-3B}"
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/demo_data/gr1.PickNPlace}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-GR1}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-$REPO_ROOT/scripts/gr1_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/gr00t_posttrain}"

MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
USE_WANDB="${USE_WANDB:-false}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-path)         DATASET_PATH="$2"; shift 2 ;;
        --embodiment-tag)       EMBODIMENT_TAG="$2"; shift 2 ;;
        --modality-config-path) MODALITY_CONFIG_PATH="$2"; shift 2 ;;
        --base-model-path)      BASE_MODEL_PATH="$2"; shift 2 ;;
        --output-dir)           OUTPUT_DIR="$2"; shift 2 ;;
        --max-steps)            MAX_STEPS="$2"; shift 2 ;;
        --global-batch-size)    GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
        --learning-rate)        LEARNING_RATE="$2"; shift 2 ;;
        --num-gpus)             NUM_GPUS="$2"; shift 2 ;;
        --use-wandb)            USE_WANDB="true"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo " GR00T N1.6 — 8x B200 Post-Training"
echo "============================================"
echo "  GPUs           : $NUM_GPUS"
echo "  Base model     : $BASE_MODEL_PATH"
echo "  Dataset        : $DATASET_PATH"
echo "  Embodiment     : $EMBODIMENT_TAG"
echo "  Output         : $OUTPUT_DIR"
echo "  Batch size     : $GLOBAL_BATCH_SIZE"
echo "  Max steps      : $MAX_STEPS"
echo "  Learning rate  : $LEARNING_RATE"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Set up uv environment if needed (no system changes — uv only)
# ---------------------------------------------------------------------------
export CUDA_HOME="${CUDA_HOME:-/opt/pytorch/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

if [ ! -d "$REPO_ROOT/.venv" ]; then
    echo ""
    echo "[Setup] Installing dependencies via uv (no system changes)..."
    uv sync
    uv pip install -e .
fi
source "$REPO_ROOT/.venv/bin/activate"

# Verify torch sees all GPUs
VISIBLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo ""
echo "[Check] PyTorch sees $VISIBLE_GPUS GPU(s)"
if [ "$VISIBLE_GPUS" -lt "$NUM_GPUS" ]; then
    echo "WARNING: requested $NUM_GPUS GPUs but only $VISIBLE_GPUS visible. Adjusting."
    NUM_GPUS="$VISIBLE_GPUS"
fi

# ---------------------------------------------------------------------------
# 2. Build torchrun launch command
# ---------------------------------------------------------------------------
LAUNCH_ARGS=(
    --base-model-path "$BASE_MODEL_PATH"
    --dataset-path "$DATASET_PATH"
    --embodiment-tag "$EMBODIMENT_TAG"
    --num-gpus "$NUM_GPUS"
    --output-dir "$OUTPUT_DIR"
    --max-steps "$MAX_STEPS"
    --save-steps "$SAVE_STEPS"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --warmup-ratio "$WARMUP_RATIO"
    --weight-decay "$WEIGHT_DECAY"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
)

# Add modality config if provided (required for NEW_EMBODIMENT)
if [ -n "$MODALITY_CONFIG_PATH" ]; then
    LAUNCH_ARGS+=(--modality-config-path "$MODALITY_CONFIG_PATH")
fi

# Enable wandb
if [ "$USE_WANDB" = "true" ]; then
    LAUNCH_ARGS+=(--use-wandb)
fi

# ---------------------------------------------------------------------------
# 3. Launch training
# ---------------------------------------------------------------------------
echo ""
echo "[Train] Launching torchrun with $NUM_GPUS GPUs (DeepSpeed ZeRO-2)..."
echo ""

set -x
exec torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    gr00t/experiment/launch_finetune.py \
    "${LAUNCH_ARGS[@]}"
