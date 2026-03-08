#!/bin/bash

# Evaluate trained VAE on validation/test sets
# Usage: ./eval_vae.sh [val|test] [--quick]

set -e

# Configuration
CHECKPOINT="outputs/vae/checkpoints/best.pt"
MODEL_CONFIG="configs/model/vae.yaml"
METADATA_DIR="data/metadata"
DEVICE="cuda"
BATCH_SIZE=16
NUM_WORKERS=4

# Visualization settings
NUM_VIS=20
NUM_WORST=10
NUM_BEST=5

# Parse arguments
SPLIT="${1:-val}"
QUICK_MODE=false

if [[ "$2" == "--quick" ]] || [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
fi

# Validate split
if [[ "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
    echo "Error: Split must be 'val' or 'test'"
    echo "Usage: $0 [val|test] [--quick]"
    exit 1
fi

# Set output directory
if $QUICK_MODE; then
    OUTPUT_DIR="outputs/vae/eval_${SPLIT}_quick"
    MAX_SAMPLES_FLAG="--max_samples 32"
    echo "Running in QUICK MODE (32 samples)"
else
    OUTPUT_DIR="outputs/vae/eval_${SPLIT}"
    MAX_SAMPLES_FLAG=""
    echo "Running FULL evaluation"
fi

# Print configuration
echo "==========================================="
echo "VAE RECONSTRUCTION EVALUATION"
echo "==========================================="
echo "Checkpoint:    $CHECKPOINT"
echo "Split:         $SPLIT"
echo "Output dir:    $OUTPUT_DIR"
echo "Device:        $DEVICE"
echo "Batch size:    $BATCH_SIZE"
echo "==========================================="
echo ""

# Check checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

# Check model config exists
if [ ! -f "$MODEL_CONFIG" ]; then
    echo "ERROR: Model config not found at $MODEL_CONFIG"
    exit 1
fi

# Run evaluation
echo "Starting evaluation..."
echo ""

python3 scripts/eval_vae.py \
    --checkpoint "$CHECKPOINT" \
    --model_config "$MODEL_CONFIG" \
    --metadata_dir "$METADATA_DIR" \
    --split "$SPLIT" \
    --source all \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --device "$DEVICE" \
    --num_visualizations $NUM_VIS \
    --num_worst_cases $NUM_WORST \
    --num_best_cases $NUM_BEST \
    --output_dir "$OUTPUT_DIR" \
    $MAX_SAMPLES_FLAG

# Print completion message
echo ""
echo "==========================================="
echo "EVALUATION COMPLETE"
echo "==========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View summary:"
echo "  cat $OUTPUT_DIR/summary.txt"
echo ""
echo "View visualizations:"
echo "  ls $OUTPUT_DIR/visualizations/"
echo "  ls $OUTPUT_DIR/worst_cases/"
echo "  ls $OUTPUT_DIR/best_cases/"
echo ""
