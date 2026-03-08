#!/bin/bash

################################################################################
# Full VAE Training Script with Tmux Session Management
# 
# This script launches full-dataset VAE training inside a tmux session
# for persistence across disconnects.
################################################################################

set -euo pipefail

# ==============================================================================
# EDITABLE CONFIGURATION
# ==============================================================================

# Tmux session name
SESSION_NAME="vae_train"

# Project paths
PROJECT_ROOT="/workspace/ece285"
TRAIN_CONFIG="configs/train/vae_train.yaml"
MODEL_CONFIG="configs/model/vae.yaml"

# Training parameters
DEVICE="cuda"
RUN_NAME="vae_full_train"
OUTPUT_DIR="outputs/vae"
EPOCHS="50"

# Logging
LOG_DIR="outputs/vae/logs"
LOG_FILE="${LOG_DIR}/train_vae_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Optional: Resume from checkpoint (leave empty for fresh training)
RESUME_CKPT=""  # e.g., "outputs/vae/checkpoints/latest.pt"

# Optional: Extra arguments to pass to training script
EXTRA_ARGS=""

# ==============================================================================
# ENVIRONMENT VERIFICATION
# ==============================================================================

echo "========================================================================"
echo "VAE FULL TRAINING LAUNCHER"
echo "========================================================================"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ ERROR: tmux is not installed"
    echo "Please install tmux first: apt-get install tmux"
    exit 1
fi

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: python3 is not found"
    exit 1
fi

# Change to project root
echo "📁 Changing to project root: ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}" || {
    echo "❌ ERROR: Could not cd to ${PROJECT_ROOT}"
    exit 1
}

# Verify config files exist
if [[ ! -f "${TRAIN_CONFIG}" ]]; then
    echo "❌ ERROR: Training config not found: ${TRAIN_CONFIG}"
    exit 1
fi

if [[ ! -f "${MODEL_CONFIG}" ]]; then
    echo "❌ ERROR: Model config not found: ${MODEL_CONFIG}"
    exit 1
fi

# Check if wandb is available
if ! python3 -c "import wandb" 2>/dev/null; then
    echo "❌ ERROR: wandb is not installed"
    echo "Please install: pip install wandb"
    exit 1
fi

echo "✓ Environment checks passed"
echo ""

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"
echo "✓ Created output directories"
echo ""

# ==============================================================================
# WANDB CONFIGURATION
# ==============================================================================

echo "🔧 Configuring WandB..."
export WANDB_MODE=online
echo "✓ WandB mode: online"
echo ""
echo "⚠️  REMINDER: Make sure you are logged into WandB:"
echo "   If not logged in, run: wandb login"
echo ""

# ==============================================================================
# TMUX SESSION CHECK
# ==============================================================================

echo "🔍 Checking for existing tmux session: ${SESSION_NAME}"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo ""
    echo "⚠️  WARNING: Tmux session '${SESSION_NAME}' already exists!"
    echo ""
    echo "To attach to the existing session, run:"
    echo "    tmux attach -t ${SESSION_NAME}"
    echo ""
    echo "To kill the existing session first, run:"
    echo "    tmux kill-session -t ${SESSION_NAME}"
    echo ""
    echo "Exiting to avoid duplicate training runs."
    exit 1
fi

echo "✓ Session does not exist, proceeding..."
echo ""

# ==============================================================================
# BUILD TRAINING COMMAND
# ==============================================================================

TRAIN_CMD="python3 scripts/train_vae.py"
TRAIN_CMD="${TRAIN_CMD} --train_config ${TRAIN_CONFIG}"
TRAIN_CMD="${TRAIN_CMD} --model_config ${MODEL_CONFIG}"
TRAIN_CMD="${TRAIN_CMD} --device ${DEVICE}"
TRAIN_CMD="${TRAIN_CMD} --epochs ${EPOCHS}"
TRAIN_CMD="${TRAIN_CMD} --run_name ${RUN_NAME}"
TRAIN_CMD="${TRAIN_CMD} --output_dir ${OUTPUT_DIR}"

# Add resume checkpoint if specified
if [[ -n "${RESUME_CKPT}" ]]; then
    if [[ ! -f "${RESUME_CKPT}" ]]; then
        echo "❌ ERROR: Resume checkpoint not found: ${RESUME_CKPT}"
        exit 1
    fi
    TRAIN_CMD="${TRAIN_CMD} --resume ${RESUME_CKPT}"
    echo "📌 Resuming from checkpoint: ${RESUME_CKPT}"
fi

# Add extra arguments if specified
if [[ -n "${EXTRA_ARGS}" ]]; then
    TRAIN_CMD="${TRAIN_CMD} ${EXTRA_ARGS}"
fi

# Add logging
TRAIN_CMD="${TRAIN_CMD} 2>&1 | tee ${LOG_FILE}"

# ==============================================================================
# LAUNCH TRAINING IN TMUX
# ==============================================================================

echo "========================================================================"
echo "LAUNCHING TRAINING"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Session name: ${SESSION_NAME}"
echo "  Run name: ${RUN_NAME}"
echo "  Device: ${DEVICE}"
echo "  Epochs: ${EPOCHS}"
echo "  Log file: ${LOG_FILE}"
echo ""
echo "Training command:"
echo "  ${TRAIN_CMD}"
echo ""
echo "========================================================================"
echo ""

# Create tmux session and run training
tmux new-session -d -s "${SESSION_NAME}" -c "${PROJECT_ROOT}" bash -c "
    set -euo pipefail
    
    # Banner
    echo '========================================================================'
    echo 'VAE TRAINING SESSION'
    echo 'Session: ${SESSION_NAME}'
    echo 'Started: \$(date)'
    echo '========================================================================'
    echo ''
    
    # Environment setup
    cd '${PROJECT_ROOT}'
    export WANDB_MODE=online
    
    # Run training
    echo '🚀 Starting training...'
    echo ''
    ${TRAIN_CMD}
    
    EXIT_CODE=\$?
    
    # Completion message
    echo ''
    echo '========================================================================'
    echo 'TRAINING COMPLETED'
    echo 'Finished: \$(date)'
    echo 'Exit code: '\${EXIT_CODE}
    echo '========================================================================'
    echo ''
    
    if [ \${EXIT_CODE} -eq 0 ]; then
        echo '✓ Training completed successfully!'
    else
        echo '✗ Training failed with exit code '\${EXIT_CODE}
    fi
    
    echo ''
    echo 'Outputs saved to:'
    echo '  Checkpoints: ${OUTPUT_DIR}/checkpoints/'
    echo '  Reconstructions: ${OUTPUT_DIR}/reconstructions/'
    echo '  Log file: ${LOG_FILE}'
    echo ''
    echo 'Press Ctrl+D or type \"exit\" to close this session'
    echo ''
    
    # Keep session alive
    exec bash
"

# ==============================================================================
# SUCCESS MESSAGE
# ==============================================================================

echo "✓ Training session launched successfully!"
echo ""
echo "========================================================================"
echo "SESSION MANAGEMENT"
echo "========================================================================"
echo ""
echo "Session name: ${SESSION_NAME}"
echo ""
echo "To attach to the training session:"
echo "    tmux attach -t ${SESSION_NAME}"
echo ""
echo "To detach from session (without stopping training):"
echo "    Press: Ctrl+B, then D"
echo ""
echo "To monitor the log file in real-time:"
echo "    tail -f ${LOG_FILE}"
echo ""
echo "To list all tmux sessions:"
echo "    tmux ls"
echo ""
echo "To kill the session:"
echo "    tmux kill-session -t ${SESSION_NAME}"
echo ""
echo "========================================================================"
echo ""
echo "🚀 Training is now running in the background!"
echo "   Attach to the session to monitor progress."
echo ""
