#!/bin/zsh

# Upload MLX safetensors models (audio_tokenizer, audio_detokenizer) to HuggingFace Hub.
# Each subdirectory becomes a separate model repo under the ewchampion namespace.
#
# Usage: ./scripts/hf_mlx_upload.sh [export_dir]
#   export_dir: path to the export directory (default: mlx_tokenizer_export)
#
# Expected directory structure:
#   mlx_tokenizer_export/
#     audio_tokenizer/
#       audio_tokenizer.safetensors
#       tokenizer_config.json
#     audio_detokenizer/
#       audio_detokenizer.safetensors
#       tokenizer_config.json

set -euo pipefail

SCRIPT_DIR="${0:a:h}"
PROJECT_ROOT="${SCRIPT_DIR}/.."
EXPORT_DIR="${1:-${PROJECT_ROOT}/mlx_tokenizer_export}"

if [[ ! -d "$EXPORT_DIR" ]]; then
  echo "ERROR: Export directory not found: $EXPORT_DIR"
  echo "Run export_mlx_tokenizer.py first."
  exit 1
fi

# Increase timeout for large file uploads
export HF_HUB_HTTP_TIMEOUT=300

# HuggingFace namespace
HF_USER="ewchampion"

cd "$EXPORT_DIR"

# Iterate over each subdirectory
for model_dir in */; do
    # Remove trailing slash
    model_name="${model_dir%/}"
    
    # Skip if not a directory
    [[ -d "$model_name" ]] || continue
    
    # Check for required files
    has_safetensors=false
    for f in "$model_name"/*.safetensors; do
        [[ -f "$f" ]] && has_safetensors=true && break
    done
    
    if [[ "$has_safetensors" == "false" ]]; then
        echo "Skipping $model_name (no safetensors files)"
        continue
    fi
    
    repo_id="${HF_USER}/acestep-${model_name}-mlx"
    
    echo ""
    echo "--- Uploading $model_name → $repo_id ---"
    
    # List files being uploaded
    echo "Files to upload:"
    find "$model_name" -type f | while read f; do
        size=$(stat -f%z "$f" 2>/dev/null || stat --printf="%s" "$f" 2>/dev/null)
        echo "  $(echo $size | numfmt --to=iec 2>/dev/null || echo "${size} bytes")  $f"
    done
    
    # Retry logic (up to 3 attempts)
    for attempt in {1..3}; do
        echo "Attempt $attempt..."
        if hf upload "$repo_id" "$model_name" . ; then
            echo "✅ Successfully uploaded $model_name"
            break
        else
            echo "❌ Error uploading $model_name (Attempt $attempt failed)"
            if [[ $attempt -lt 3 ]]; then
                echo "Retrying in 10 seconds..."
                sleep 10
            else
                echo "FAILED to upload $model_name after 3 attempts."
                exit 1
            fi
        fi
    done
done

echo ""
echo "All MLX models uploaded successfully!"
