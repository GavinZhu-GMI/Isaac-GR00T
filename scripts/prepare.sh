#!/bin/bash
# prepare.sh — Download model weights + validate/prepare demo data for post-training
# Usage: bash scripts/prepare.sh [--dataset-path <path>] [--embodiment-tag <tag>]
#
# Defaults to the GR1 demo dataset shipped with the repo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODEL_REPO="nvidia/GR00T-N1.6-3B"
MODEL_LOCAL_DIR="$REPO_ROOT/checkpoints/GR00T-N1.6-3B"
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/demo_data/gr1.PickNPlace}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-GR1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --embodiment-tag) EMBODIMENT_TAG="$2"; shift 2 ;;
        --model-repo) MODEL_REPO="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo " GR00T N1.6 — Data & Weight Preparation"
echo "============================================"
echo "  Model repo   : $MODEL_REPO"
echo "  Local weights : $MODEL_LOCAL_DIR"
echo "  Dataset path  : $DATASET_PATH"
echo "  Embodiment    : $EMBODIMENT_TAG"
echo "============================================"

# ---------------------------------------------------------------------------
# 0. Pull Git LFS objects for dataset files
# ---------------------------------------------------------------------------
echo ""
echo "[Step 0] Ensuring Git LFS data is pulled..."
if command -v git-lfs &>/dev/null || git lfs version &>/dev/null 2>&1; then
    git lfs install --local
    git lfs pull
    echo "  Git LFS pull complete."
else
    echo "  WARNING: git-lfs not installed. Data files may be LFS pointers."
    echo "  Install with: sudo apt-get install -y git-lfs"
fi

# ---------------------------------------------------------------------------
# 1. Ensure uv venv is set up (no system changes — uv only)
# ---------------------------------------------------------------------------
export CUDA_HOME="${CUDA_HOME:-/opt/pytorch/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

if [ ! -d "$REPO_ROOT/.venv" ]; then
    echo ""
    echo "[Step 1] Setting up uv environment (uv sync + editable install)..."
    uv sync
    uv pip install -e .
fi
source "$REPO_ROOT/.venv/bin/activate"

# Ensure flash-attn is installed (required by the model)
if ! python -c "import flash_attn" 2>/dev/null; then
    echo ""
    echo "[Step 1b] Installing flash-attn..."
    CUDA_HOME="$CUDA_HOME" uv pip install "flash-attn==2.7.4.post1" --no-build-isolation
fi

# ---------------------------------------------------------------------------
# 1. Download model weights from HuggingFace
# ---------------------------------------------------------------------------
echo ""
echo "[Step 2] Downloading model weights: $MODEL_REPO"

if [ -d "$MODEL_LOCAL_DIR" ] && [ -f "$MODEL_LOCAL_DIR/config.json" ]; then
    echo "  Weights already present at $MODEL_LOCAL_DIR — skipping download."
else
    mkdir -p "$MODEL_LOCAL_DIR"
    # Use huggingface-cli if available, otherwise fall back to python
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_LOCAL_DIR"
    else
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', local_dir='$MODEL_LOCAL_DIR')
"
    fi
    echo "  Download complete."
fi

# ---------------------------------------------------------------------------
# 2. Validate dataset structure
# ---------------------------------------------------------------------------
echo ""
echo "[Step 3] Validating dataset at: $DATASET_PATH"

ERRORS=0

# Check required directories
for subdir in meta data; do
    if [ ! -d "$DATASET_PATH/$subdir" ]; then
        echo "  ERROR: missing directory $DATASET_PATH/$subdir"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check required meta files
for metafile in modality.json info.json episodes.jsonl tasks.jsonl; do
    if [ ! -f "$DATASET_PATH/meta/$metafile" ]; then
        echo "  ERROR: missing $DATASET_PATH/meta/$metafile"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check parquet files exist
PARQUET_COUNT=$(find "$DATASET_PATH/data" -name "*.parquet" 2>/dev/null | wc -l)
if [ "$PARQUET_COUNT" -eq 0 ]; then
    echo "  ERROR: no .parquet files found under $DATASET_PATH/data/"
    ERRORS=$((ERRORS + 1))
else
    echo "  Found $PARQUET_COUNT parquet file(s)."
fi

# Check video files exist
VIDEO_COUNT=$(find "$DATASET_PATH/videos" -name "*.mp4" 2>/dev/null | wc -l)
if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "  WARNING: no .mp4 video files found under $DATASET_PATH/videos/"
else
    echo "  Found $VIDEO_COUNT video file(s)."
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "  Dataset validation failed with $ERRORS error(s). Fix issues and re-run."
    exit 1
fi
echo "  Dataset structure OK."

# ---------------------------------------------------------------------------
# 3. Compute / refresh dataset statistics
# ---------------------------------------------------------------------------
echo ""
echo "[Step 4] Computing dataset statistics..."

if [ -f "$DATASET_PATH/meta/stats.json" ] && [ -f "$DATASET_PATH/meta/relative_stats.json" ]; then
    echo "  stats.json and relative_stats.json already exist."
    echo "  To recompute, delete them and re-run this script."
else
    python "$REPO_ROOT/gr00t/data/stats.py" "$DATASET_PATH" "$EMBODIMENT_TAG"
    echo "  Statistics computed."
fi

# ---------------------------------------------------------------------------
# 4. Quick sanity: load one batch through the data pipeline
# ---------------------------------------------------------------------------
echo ""
echo "[Step 5] Sanity check — loading one sample through the data pipeline..."

python -c "
import json, pathlib
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

ds_path = pathlib.Path('$DATASET_PATH')
tag = EmbodimentTag('$EMBODIMENT_TAG'.lower())

# 1. Check modality config (if pre-registered)
if tag.value in MODALITY_CONFIGS:
    modality_config = MODALITY_CONFIGS[tag.value]
    print(f'  Modality config keys: {list(modality_config.keys())}')
    # Cross-check against modality.json
    with open(ds_path / 'meta' / 'modality.json') as f:
        modality_json = json.load(f)
    for section in ['state', 'action', 'video']:
        if section in modality_config:
            cfg_keys = modality_config[section].modality_keys
            json_keys = list(modality_json.get(section, {}).keys())
            missing = [k for k in cfg_keys if k not in json_keys]
            if missing:
                print(f'  WARNING: {section} config keys {missing} not in modality.json (available: {json_keys})')
            else:
                print(f'  {section}: config keys {cfg_keys} OK')
else:
    print(f'  Tag \"{tag.value}\" has no pre-registered posttrain config (pretrain-only or NEW_EMBODIMENT).')
    print(f'  You will need --modality-config-path at training time.')
    with open(ds_path / 'meta' / 'modality.json') as f:
        modality_json = json.load(f)
    for section in ['state', 'action', 'video', 'annotation']:
        if section in modality_json:
            print(f'  modality.json {section}: {list(modality_json[section].keys())}')

# 2. Verify parquet is readable
import pandas as pd
parquet_files = sorted(ds_path.glob('data/**/*.parquet'))
df = pd.read_parquet(parquet_files[0])
print(f'  Parquet sample: {len(df)} rows, columns: {list(df.columns)[:6]}...')
print('  Data pipeline sanity check PASSED.')
"

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo " Preparation complete!"
echo "============================================"
echo "  Model weights : $MODEL_LOCAL_DIR"
echo "  Dataset        : $DATASET_PATH"
echo "  Stats          : $DATASET_PATH/meta/stats.json"
echo ""
echo "  Next: run training with:"
echo "    bash scripts/train_8xb200.sh \\"
echo "      --dataset-path $DATASET_PATH \\"
echo "      --embodiment-tag $EMBODIMENT_TAG"
echo "============================================"
