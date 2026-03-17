#!/usr/bin/env zsh

# ─── HuggingFace Model Repo Validator ───
#
# Checks all ewchampion/* model repos on HuggingFace for expected files.
# Uses the HuggingFace API tree endpoint to list repo contents recursively,
# then validates each repo has its required ancillary files.
#
# Usage: ./scripts/hf_validate.sh
#
# Exit codes:
#   0 = all repos valid
#   1 = one or more repos have missing or unexpected files

set -euo pipefail

HF_USER="ewchampion"
API_BASE="https://huggingface.co/api/models"

# ─── Define all repos and their expected files by model type ───

# DiT repos: need encoder.safetensors, null_condition_embedding.safetensors
DIT_REPOS=(
  "acestep-v15-turbo-coreml-4bit.mlpackage"
  "acestep-v15-base-coreml-4bit.mlpackage"
  "acestep-v15-sft-coreml-4bit.mlpackage"
  "acestep-v15-turbo-coreml-6bit.mlpackage"
  "acestep-v15-base-coreml-6bit.mlpackage"
  "acestep-v15-sft-coreml-6bit.mlpackage"
  "acestep-v15-turbo-coreml-8bit.mlpackage"
  "acestep-v15-base-coreml-8bit.mlpackage"
  "acestep-v15-sft-coreml-8bit.mlpackage"
  "acestep-v15-turbo-coreml-16bit.mlpackage"
  "acestep-v15-base-coreml-16bit.mlpackage"
  "acestep-v15-sft-coreml-16bit.mlpackage"
)
DIT_REQUIRED_FILES=(
  "Manifest.json"
  "encoder.safetensors"
  "null_condition_embedding.safetensors"
)

# VAE repos: need encoder.safetensors
VAE_REPOS=(
  "vae-coreml-4bit.mlpackage"
  "vae-coreml-6bit.mlpackage"
  "vae-coreml-8bit.mlpackage"
  "vae-coreml-16bit.mlpackage"
)
VAE_REQUIRED_FILES=(
  "Manifest.json"
  "encoder.safetensors"
)

# VAE Encoder repos: CoreML model only (no ancillary files)
VAE_ENCODER_REPOS=(
  "vae_encoder-coreml-4bit.mlpackage"
  "vae_encoder-coreml-6bit.mlpackage"
  "vae_encoder-coreml-8bit.mlpackage"
  "vae_encoder-coreml-16bit.mlpackage"
)
VAE_ENCODER_REQUIRED_FILES=(
  "Manifest.json"
)

# LM repos: need tokenizer.json, tokenizer_config.json
LM_REPOS=(
  "acestep-5Hz-lm-0.6B-coreml-4bit.mlpackage"
  "acestep-5Hz-lm-1.7B-coreml-4bit.mlpackage"
  "acestep-5Hz-lm-4B-coreml-4bit.mlpackage"
  "acestep-5Hz-lm-0.6B-coreml-6bit.mlpackage"
  "acestep-5Hz-lm-1.7B-coreml-6bit.mlpackage"
  "acestep-5Hz-lm-4B-coreml-6bit.mlpackage"
  "acestep-5Hz-lm-0.6B-coreml-8bit.mlpackage"
  "acestep-5Hz-lm-1.7B-coreml-8bit.mlpackage"
  "acestep-5Hz-lm-4B-coreml-8bit.mlpackage"
  "acestep-5Hz-lm-0.6B-coreml-16bit.mlpackage"
  "acestep-5Hz-lm-1.7B-coreml-16bit.mlpackage"
  "acestep-5Hz-lm-4B-coreml-16bit.mlpackage"
)
LM_REQUIRED_FILES=(
  "Manifest.json"
  "tokenizer.json"
  "tokenizer_config.json"
)

# Text Encoder repos: need tokenizer.json, tokenizer_config.json, embed_tokens.safetensors
TEXT_ENCODER_REPOS=(
  "Qwen3-Embedding-0.6B-coreml-4bit.mlpackage"
  "Qwen3-Embedding-0.6B-coreml-6bit.mlpackage"
  "Qwen3-Embedding-0.6B-coreml-8bit.mlpackage"
  "Qwen3-Embedding-0.6B-coreml-16bit.mlpackage"
)
TEXT_ENCODER_REQUIRED_FILES=(
  "Manifest.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "embed_tokens.safetensors"
)

# CoreML model data files (at least one must exist under Data/)
COREML_DATA_PATTERN="Data/"

# ─── Helper: fetch file list from HuggingFace tree API (recursive) ───

fetch_repo_files() {
  local repo="$1"
  local url="${API_BASE}/${HF_USER}/${repo}/tree/main"
  local response

  # Fetch tree (non-recursive endpoint; we check recursively by also fetching subdirs)
  response=$(curl -s -w "\n%{http_code}" "$url" 2>/dev/null)
  local http_code=$(echo "$response" | tail -1)
  local body=$(echo "$response" | sed '$d')

  if [[ "$http_code" != "200" ]]; then
    echo "HTTP_ERROR:$http_code"
    return
  fi

  # Parse top-level entries
  local files=()
  local dirs=()

  # Extract file paths and directory paths using python for reliable JSON parsing
  local parsed
  parsed=$(python3 -c "
import json, sys
try:
    items = json.loads(sys.stdin.read())
    for item in items:
        print(f\"{item['type']}|{item['path']}|{item.get('size', 0)}\")
except Exception as e:
    print(f'PARSE_ERROR:{e}', file=sys.stderr)
" <<< "$body" 2>/dev/null)

  # Print file entries
  echo "$parsed"

  # Recurse into directories
  while IFS='|' read -r entry_type entry_path entry_size; do
    if [[ "$entry_type" == "directory" ]]; then
      local sub_url="${API_BASE}/${HF_USER}/${repo}/tree/main/${entry_path}"
      local sub_response
      sub_response=$(curl -s "$sub_url" 2>/dev/null)
      python3 -c "
import json, sys
try:
    items = json.loads(sys.stdin.read())
    for item in items:
        print(f\"{item['type']}|{item['path']}|{item.get('size', 0)}\")
except:
    pass
" <<< "$sub_response" 2>/dev/null
    fi
  done <<< "$parsed"
}

# ─── Helper: format file size ───

format_size() {
  local size=$1
  if (( size >= 1073741824 )); then
    printf "%.1f GB" $(echo "scale=1; $size / 1073741824" | bc)
  elif (( size >= 1048576 )); then
    printf "%.1f MB" $(echo "scale=1; $size / 1048576" | bc)
  elif (( size >= 1024 )); then
    printf "%.1f KB" $(echo "scale=1; $size / 1024" | bc)
  else
    printf "%d B" "$size"
  fi
}

# ─── Main validation logic ───

total_repos=0
pass_count=0
fail_count=0
not_found_count=0
failed_repos=()

validate_repos() {
  local model_type="$1"
  shift
  local -a required_files=()
  local -a repos=()

  # Split args: first comes required files (terminated by --), then repos
  local in_files=true
  for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
      in_files=false
      continue
    fi
    if $in_files; then
      required_files+=("$arg")
    else
      repos+=("$arg")
    fi
  done

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  $model_type Models (${#repos[@]} repos)"
  echo "  Required files: ${required_files[*]}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for repo in "${repos[@]}"; do
    total_repos=$((total_repos + 1))
    echo ""
    echo "  📦 ${HF_USER}/${repo}"

    # Fetch all files recursively
    local file_listing=""
    file_listing="$(fetch_repo_files "$repo")"

    # Check for HTTP errors
    if echo "$file_listing" | grep -q "^HTTP_ERROR:" 2>/dev/null; then
      local code
      code="$(echo "$file_listing" | grep "HTTP_ERROR:" | head -1 | cut -d: -f2)"
      if [[ "$code" == "404" ]]; then
        echo "     ❌ REPO NOT FOUND (404)"
        not_found_count=$((not_found_count + 1))
        failed_repos+=("${repo} [NOT FOUND]")
      else
        echo "     ❌ HTTP ERROR: $code"
        fail_count=$((fail_count + 1))
        failed_repos+=("${repo} [HTTP $code]")
      fi
      continue
    fi

    # Collect just the file paths
    local -a actual_files=()
    local has_data_dir=false
    local has_model_file=false

    while IFS='|' read -r entry_type entry_path entry_size; do
      [[ -z "$entry_type" ]] && continue
      if [[ "$entry_type" == "file" ]]; then
        actual_files+=("$entry_path")
        local formatted_size
        formatted_size="$(format_size "$entry_size")"
        echo "     📄 ${entry_path} (${formatted_size})"
      elif [[ "$entry_type" == "directory" ]]; then
        if [[ "$entry_path" == "Data" || "$entry_path" == Data/* ]]; then
          has_data_dir=true
        fi
      fi
    done <<< "$file_listing"

    # Check for CoreML Data directory
    if ! $has_data_dir; then
      echo "     ⚠️  No Data/ directory found (CoreML model data missing)"
    fi

    # Check for model files under Data/
    for f in "${actual_files[@]}"; do
      if [[ "$f" == Data/* ]]; then
        has_model_file=true
        break
      fi
    done

    # Check required files
    local missing=()
    for req in "${required_files[@]}"; do
      local found=false
      for f in "${actual_files[@]}"; do
        if [[ "$f" == "$req" ]]; then
          found=true
          break
        fi
      done
      if ! $found; then
        missing+=("$req")
      fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
      for m in "${missing[@]}"; do
        echo "     ❌ MISSING: $m"
      done
      fail_count=$((fail_count + 1))
      failed_repos+=("${repo} [missing: ${missing[*]}]")
    elif ! $has_data_dir; then
      fail_count=$((fail_count + 1))
      failed_repos+=("${repo} [missing: Data/]")
    else
      echo "     ✅ PASS"
      pass_count=$((pass_count + 1))
    fi
  done
}

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        HuggingFace Model Repository Validator               ║"
echo "║        Namespace: ${HF_USER}                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Validate each model type
validate_repos "DiT" "${DIT_REQUIRED_FILES[@]}" "--" "${DIT_REPOS[@]}"
validate_repos "VAE Decoder" "${VAE_REQUIRED_FILES[@]}" "--" "${VAE_REPOS[@]}"
validate_repos "VAE Encoder" "${VAE_ENCODER_REQUIRED_FILES[@]}" "--" "${VAE_ENCODER_REPOS[@]}"
validate_repos "LM" "${LM_REQUIRED_FILES[@]}" "--" "${LM_REPOS[@]}"
validate_repos "Text Encoder" "${TEXT_ENCODER_REQUIRED_FILES[@]}" "--" "${TEXT_ENCODER_REPOS[@]}"

# ─── Final Report ───

echo ""
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     VALIDATION REPORT                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Total repos checked:  $total_repos"
echo "  ✅ Passed:            $pass_count"
echo "  ❌ Failed:            $fail_count"
echo "  🔍 Not found:         $not_found_count"
echo ""

if [[ ${#failed_repos[@]} -gt 0 ]]; then
  echo "  ──── Failed Repos ────"
  for fr in "${failed_repos[@]}"; do
    echo "    ❌ ${HF_USER}/${fr}"
  done
  echo ""
  echo "  ACTION REQUIRED: Upload missing files to the repos listed above."
  echo "  Use: hf upload <repo_id> <local_file> <remote_path>"
  echo ""
  exit 1
else
  echo "  All repos contain their required files. 🎉"
  echo ""
  exit 0
fi
