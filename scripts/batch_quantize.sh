#!/usr/bin/env zsh

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

for checkpoint_path in "${checkpoint_dirs[@]}"; do
  checkpoint_name="${checkpoint_path:t}"  # last path node (basename)
  log_file="debug_${checkpoint_name}.txt"

  echo "Quantizing: $checkpoint_name"
  echo "  Log: $log_file"

  python ./scripts/quantize_checkpoints.py "$checkpoint_name" 2>&1 | tee "$log_file"

  if [[ ${pipestatus[1]} -ne 0 ]]; then
    echo "WARNING: quantize_checkpoints.py exited with an error for $checkpoint_name"
  fi

  echo "---"
done

echo "Done."
