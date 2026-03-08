#!/bin/bash
# ==============================================================================
# Latent Diffusion Baseline Training Script
# 
# This script launches full latent diffusion baseline training (NO CLIP) in tmux.
# ==============================================================================

set -euo pipefail

# ==============================================================================
# Configuration Variables
# ==============================================================================

SESSION_NAME="latent_diffusion_baseline"
PROJECT_ROOT="/workspace/ece285"
TRAIN_CONFIG="configs/train/diffusion_train.yaml"
VAE_CONFIG="configs/model/vae.yaml"
DIFFUSION_CONFIG="configs/model/diffusion.yaml"
VAE_CHECKPOINT="outputs/vae/checkpoints/best.pt"
DEVICE="cuda"
RUN_NAME="latent_diffusion_baseline_full"
OUTPUT_DIR="outputs/diffusion_baseline"
EPOCHS="500"
EVAL_EVERY="10"
SAVE_EVERY="10"
LOG_DIR="outputs/diffusion_baseline/logs"
LOG_FILE="${LOG_DIR}/train_${RUN_NAME}.log"
RESUME_CKPT=""
EXTRA_ARGS=""

# ==============================================================================
# Safety Checks
# ==============================================================================

echo "======================================================================"
echo "Latent Diffusion Baseline Training Launcher"
echo "======================================================================"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "ERROR: tmux is not installed"
    echo "Install with: apt-get install tmux"
    exit 1
fi

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT" || exit 1
echo "Working directory: $(pwd)"
echo ""

# Check if config files exist
for config_file in "$TRAIN_CONFIG" "$VAE_CONFIG" "$DIFFUSION_CONFIG"; do
    if [ ! -f "$config_file" ]; then
        echo "ERROR: Config file not found: $config_file"
        exit 1
    fi
done
echo "✓ All config files found"

# Check if VAE checkpoint exists
if [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "ERROR: VAE checkpoint not found: $VAE_CHECKPOINT"
    echo "Please train the VAE first (Step 7)"
    exit 1
fi
echo "✓ VAE checkpoint found: $VAE_CHECKPOINT"

# Create output and log directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
echo "✓ Output directory: $OUTPUT_DIR"
echo "✓ Log directory: $LOG_DIR"
echo ""

# ==============================================================================
# Check for Existing Session
# ==============================================================================

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "======================================================================"
    echo "WARNING: tmux session '$SESSION_NAME' already exists!"
    echo "======================================================================"
    echo ""
    echo "The training session is already running."
    echo ""
    echo "To attach to the existing session:"
    echo "  tmux attach -t $SESSION_NAME"
    echo ""
    echo "To view the logs:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "To kill the existing session and start fresh:"
    echo "  tmux kill-session -t $SESSION_NAME"
    echo "  Then run this script again"
    echo ""
    exit 0
fi

# ==============================================================================
# Build Training Command
# ==============================================================================

TRAIN_CMD="python3 scripts/train_latent_diffusion.py"
TRAIN_CMD="$TRAIN_CMD --train_config $TRAIN_CONFIG"
TRAIN_CMD="$TRAIN_CMD --vae_config $VAE_CONFIG"
TRAIN_CMD="$TRAIN_CMD --diffusion_config $DIFFUSION_CONFIG"
TRAIN_CMD="$TRAIN_CMD --vae_checkpoint $VAE_CHECKPOINT"
TRAIN_CMD="$TRAIN_CMD --device $DEVICE"
TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
TRAIN_CMD="$TRAIN_CMD --eval_every_n_epochs $EVAL_EVERY"
TRAIN_CMD="$TRAIN_CMD --save_every_n_epochs $SAVE_EVERY"
TRAIN_CMD="$TRAIN_CMD --run_name $RUN_NAME"
TRAIN_CMD="$TRAIN_CMD --output_dir $OUTPUT_DIR"

# Add resume checkpoint if specified
if [ -n "$RESUME_CKPT" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME_CKPT"
    echo "Resuming from: $RESUME_CKPT"
fi

# Add extra arguments if specified
if [ -n "$EXTRA_ARGS" ]; then
    TRAIN_CMD="$TRAIN_CMD $EXTRA_ARGS"
fi

# ==============================================================================
# Launch Training in tmux
# ==============================================================================

echo "======================================================================"
echo "Launching Training Session"
echo "======================================================================"
echo ""
echo "Session name: $SESSION_NAME"
echo "Run name: $RUN_NAME"
echo "Epochs: $EPOCHS"
echo "Eval every: $EVAL_EVERY epochs"
echo "Save every: $SAVE_EVERY epochs"
echo "Log file: $LOG_FILE"
echo ""
echo "Training command:"
echo "$TRAIN_CMD"
echo ""

# Create tmux session
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT"

# Send environment setup and training command
tmux send-keys -t "$SESSION_NAME" "export WANDB_MODE=online" C-m
tmux send-keys -t "$SESSION_NAME" "cd $PROJECT_ROOT" C-m
tmux send-keys -t "$SESSION_NAME" "$TRAIN_CMD 2>&1 | tee $LOG_FILE" C-m

echo "======================================================================"
echo "Training launched successfully!"
echo "======================================================================"
echo ""
echo "To attach to the training session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach from the session (without stopping training):"
echo "  Press: Ctrl+B, then D"
echo ""
echo "To view logs in real-time:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check training progress without attaching:"
echo "  tail -n 50 $LOG_FILE"
echo ""
echo "To kill the training session:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo "WandB dashboard:"
echo "  Check https://wandb.ai for online metrics"
echo ""
echo "======================================================================"
