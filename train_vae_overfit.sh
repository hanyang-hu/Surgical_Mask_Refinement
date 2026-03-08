#!/bin/bash

# VAE Overfit Training Script
# Trains VAE on 16 samples for 500 epochs to verify overfitting capability

echo "========================================================================"
echo "VAE OVERFIT TRAINING - 500 EPOCHS"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

cd /workspace/ece285

python3 scripts/train_vae.py \
    --train_config configs/train/vae_train.yaml \
    --model_config configs/model/vae.yaml \
    --device cuda \
    --overfit_small \
    --overfit_num_samples 16 \
    --epochs 500 \
    --run_name vae_overfit_500

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "TRAINING FINISHED"
echo "========================================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""
echo "Outputs:"
echo "  Checkpoints: /workspace/ece285/outputs/vae/checkpoints/"
echo "  Reconstructions: /workspace/ece285/outputs/vae/reconstructions/"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo ""
    echo "Check WandB for training curves:"
    echo "  https://wandb.ai/changwei-uc-san-diego/surgical-mask-vae"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
