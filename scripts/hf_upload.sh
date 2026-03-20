#!/bin/zsh

# Upload all .mlpackage directories to HuggingFace Hub.
# Each directory becomes a separate model repo under the ewchampion namespace.
#
# Usage: ./scripts/hf_upload.sh

set -euo pipefail

# Resolve the project root relative to this script's location
SCRIPT_DIR="${0:a:h}"
PROJECT_ROOT="${SCRIPT_DIR}/.."
PACKAGES_DIR="${PROJECT_ROOT}/quantized_checkpoints_coreml"

if [[ ! -d "$PACKAGES_DIR" ]]; then
  echo "ERROR: Packages directory not found: $PACKAGES_DIR"
  exit 1
fi

# Increase timeout for large file uploads (5 minutes)
export HF_HUB_HTTP_TIMEOUT=300

# HuggingFace namespace
HF_USER="ewchampion"

cd "$PACKAGES_DIR"

# Iterate over each *.mlpackage directory
for mlpackage in *.mlpackage; do
    # Skip if no matches found
    [[ -d "$mlpackage" ]] || continue
    
    repo_id="${HF_USER}/${mlpackage}"
    
    echo ""
    echo "--- Uploading $mlpackage → $repo_id ---"
    
    # List files being uploaded for verification
    echo "Files to upload:"
    find "$mlpackage" -type f | while read f; do
        size=$(stat -f%z "$f" 2>/dev/null || stat --printf="%s" "$f" 2>/dev/null)
        echo "  $(echo $size | numfmt --to=iec 2>/dev/null || echo "${size} bytes")  $f"
    done
    
    # Retry logic (up to 3 attempts)
    for attempt in {1..3}; do
        echo "Attempt $attempt..."
        if hf upload "$repo_id" "$mlpackage" . ; then
            echo "✅ Successfully uploaded $mlpackage"
            break
        else
            echo "❌ Error uploading $mlpackage (Attempt $attempt failed)"
            if [[ $attempt -lt 3 ]]; then
                echo "Retrying in 10 seconds..."
                sleep 10
            else
                echo "FAILED to upload $mlpackage after 3 attempts."
                exit 1
            fi
        fi
    done
done
