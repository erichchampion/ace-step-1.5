#!/usr/bin/env bash
# Run the conditioning pipeline comparison tests:
#   1. Export Python reference tensors
#   2. Run Swift comparison tests against them
#
# Usage:
#   ./scripts/run_conditioning_comparison.sh
#
# Prerequisites:
#   - Python venv at .venv with transformers, torch, mlx installed
#   - checkpoints/Qwen3-Embedding-0.6B and checkpoints/acestep-v15-turbo downloaded

set -euo pipefail
cd "$(dirname "$0")/.."

STAGES_DIR="/tmp/conditioning_stages"
ENCODER_WEIGHTS="$(pwd)/checkpoints/acestep-v15-turbo/encoder.safetensors"
EMBED_TOKENS="$(pwd)/checkpoints/Qwen3-Embedding-0.6B/model.safetensors"

# If encoder.safetensors doesn't exist, try model.safetensors (contains encoder.* keys)
if [ ! -f "$ENCODER_WEIGHTS" ]; then
    ENCODER_WEIGHTS="$(pwd)/checkpoints/acestep-v15-turbo/model.safetensors"
fi

echo "========================================"
echo "Step 1: Export Python reference tensors"
echo "========================================"
PYTHONPATH=. .venv/bin/python3 scripts/export_conditioning_stages.py --output-dir "$STAGES_DIR"

echo ""
echo "========================================"
echo "Step 2: Run Swift comparison tests"
echo "========================================"
export CONDITIONING_STAGES_DIR="$STAGES_DIR"
export ENCODER_WEIGHTS_PATH="$ENCODER_WEIGHTS"
export EMBED_TOKENS_PATH="$EMBED_TOKENS"

swift test --filter ConditioningComparisonTests 2>&1 | tail -60

echo ""
echo "Done!"
