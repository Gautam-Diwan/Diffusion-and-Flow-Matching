#!/bin/bash
# =============================================================================
# Torch-Fidelity Evaluation Script
# =============================================================================
#
# Usage:
#   ./scripts/evaluate_with_fidelity.sh \
#       --checkpoint checkpoints/ddpm/ddpm_final.pt \
#       --method ddpm \
#       --dataset-path data/celeba \
#       --metrics kid
#
# =============================================================================

set -e

# Defaults
METHOD="ddpm"
CHECKPOINT="YOUR_PATH"
# torch-fidelity needs image files; HuggingFace caches Arrow. Use exported dir (see --dataset-path).
DATASET_PATH="data/celeba_images"
METRICS="kid"
NUM_SAMPLES=1000
BATCH_SIZE=256
NUM_STEPS=1000
GENERATED_DIR=""  # Will be set based on checkpoint location
CACHE_DIR=""     # Will be set based on checkpoint location
CELEBA_CACHE="data/celeba"  # HuggingFace cache used by export script

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --celeba-cache) CELEBA_CACHE="$2"; shift 2 ;;
        --metrics) METRICS="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    exit 1
fi

# Set output directories based on checkpoint location
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
GENERATED_DIR="${CHECKPOINT_DIR}/samples/generated"
CACHE_DIR="${CHECKPOINT_DIR}/samples/cache"

echo "=========================================="
echo "Torch-Fidelity Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Method: $METHOD"
echo "Dataset: $DATASET_PATH"
echo "Metrics: $METRICS"
echo "Num samples: $NUM_SAMPLES"
echo "Output: $GENERATED_DIR"
echo "=========================================="

# Step 1: Generate samples
echo ""
echo "[1/2] Generating samples..."
rm -rf "$GENERATED_DIR"

SAMPLE_CMD="python sample.py \
    --checkpoint $CHECKPOINT \
    --method $METHOD \
    --output_dir $GENERATED_DIR \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE"

[ -n "$NUM_STEPS" ] && SAMPLE_CMD="$SAMPLE_CMD --num_steps $NUM_STEPS"

eval $SAMPLE_CMD

# Step 1b: Ensure reference dataset is image dir (torch-fidelity needs .png/.jpg, not Arrow cache)
NUM_REF=0
if [ -d "$DATASET_PATH" ]; then
    NUM_REF=$(find "$DATASET_PATH" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
fi
if [ $((NUM_REF)) -eq 0 ]; then
    echo ""
    echo "[1b] Reference dir $DATASET_PATH missing or empty; exporting from HuggingFace cache..."
    python scripts/export_celeba_images.py --cache-dir "$CELEBA_CACHE" --output-dir "$DATASET_PATH" --split train
    if [ $? -ne 0 ]; then
        echo "Export failed. Create $DATASET_PATH with reference images or set --dataset-path to a dir of images."
        exit 1
    fi
fi

# Step 2: Run fidelity
echo ""
echo "[2/2] Computing metrics..."
mkdir -p "$CACHE_DIR"

FIDELITY_CMD="fidelity --gpu 0 --batch-size $BATCH_SIZE --cache-root $CACHE_DIR \
    --input1 $GENERATED_DIR --input2 $DATASET_PATH"

[[ "$METRICS" == *"fid"* ]] && FIDELITY_CMD="$FIDELITY_CMD --fid"
[[ "$METRICS" == *"kid"* ]] && FIDELITY_CMD="$FIDELITY_CMD --kid"
[[ "$METRICS" == *"is"* ]] && FIDELITY_CMD="$FIDELITY_CMD --isc"

eval $FIDELITY_CMD

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
