#!/usr/bin/env zsh

# Batch-quantize all checkpoints, stopping on first failure.
#
# Usage: ./scripts/batch_quantize.sh

set -euo pipefail

# Resolve the project root relative to this script's location
SCRIPT_DIR="${0:a:h}"
PROJECT_ROOT="${SCRIPT_DIR}/.."

cd "$PROJECT_ROOT" || { echo "ERROR: Could not cd to project root"; exit 1 }

pwd

# Activate the Python venv
source .venv/bin/activate || { echo "ERROR: Failed to activate venv"; exit 1 }

# Find checkpoint directories (depth 1 only)
checkpoints_dir="./checkpoints"

if [[ ! -d "$checkpoints_dir" ]]; then
  echo "ERROR: Checkpoints directory not found: $checkpoints_dir"
  exit 1
fi

# Collect checkpoint dirs into an array
checkpoint_dirs=("${checkpoints_dir}"/*(N/))

if [[ ${#checkpoint_dirs[@]} -eq 0 ]]; then
  echo "No checkpoint directories found under $checkpoints_dir"
  exit 1
fi

echo "Found ${#checkpoint_dirs[@]} checkpoint(s) to quantize"
echo "---"

failed_models=()

for checkpoint_path in "${checkpoint_dirs[@]}"; do
  checkpoint_name="${checkpoint_path:t}"  # last path node (basename)
  log_file="debug_${checkpoint_name}.txt"

  echo "Quantizing: $checkpoint_name"
  echo "  Log: $log_file"

  # Per-model sparsity: DiT tolerates aggressive pruning, VAE is sensitive to artifacts
  case "$checkpoint_name" in
    acestep*)  sparse_val=0.3 ;;   # DiT transformer — largest model, most prunable
    vae*)      sparse_val=0.1 ;;   # VAE encoder/decoder — audio fidelity sensitive
    *)         sparse_val=0.2 ;;   # Text encoder, LLM — moderate pruning
  esac
  echo "  Sparsity: ${sparse_val}"

  python ./scripts/quantize_checkpoints.py "$checkpoint_name" --grouped --sparse "$sparse_val" --ios18 2>&1 | tee "$log_file"
  exit_code=${pipestatus[1]}

  if [[ $exit_code -ne 0 ]]; then
    echo ""
    echo "=========================================="
    echo "ERROR: quantize_checkpoints.py failed for '$checkpoint_name' (exit code $exit_code)"
    echo "=========================================="
    echo ""
    echo "Last 20 lines of log:"
    tail -20 "$log_file"
    echo ""
    failed_models+=("$checkpoint_name")

    # Stop on first failure
    echo "Stopping batch processing due to error."
    break
  fi

  echo "---"
done

echo ""
echo "=========================================="
if [[ ${#failed_models[@]} -gt 0 ]]; then
  echo "BATCH RESULT: FAILED"
  echo "Failed model(s): ${failed_models[*]}"
  echo "=========================================="
  exit 1
else
  echo "BATCH RESULT: All ${#checkpoint_dirs[@]} checkpoint(s) quantized successfully. ✅"
  echo "=========================================="
  exit 0
fi
