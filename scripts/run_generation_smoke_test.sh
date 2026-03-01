#!/usr/bin/env bash
#
# Generation smoke test: run Python and/or Swift generation, then validate
# that each output file contains a non-silent, variable waveform.
#
# Usage:
#   ./scripts/run_generation_smoke_test.sh
#   SMOKE_DURATION=5.0 ./scripts/run_generation_smoke_test.sh      # longer clip
#   SMOKE_STEPS=8 ./scripts/run_generation_smoke_test.sh            # more diffusion steps
#   SMOKE_EXTRA_TESTS=1 ./scripts/run_generation_smoke_test.sh      # run parameter variation tests
#
# Environment:
#   OUTPUT_DIR          Where to write outputs (default: ./generation_smoke_output).
#   ACESTEP_CONFIG_PATH Model config for Python (e.g. acestep-v15-turbo). Default: acestep-v15-turbo.
#                       If unset, defaults to acestep-v15-turbo and a valid checkpoint is sought under checkpoints/.
#   DIT_WEIGHTS_PATH    DiT checkpoint dir (model.safetensors). Set automatically when checkpoints/$ACESTEP_CONFIG_PATH exists.
#   VAE_WEIGHTS_PATH    VAE decoder safetensors path. Set automatically when checkpoints/vae/decoder.safetensors exists.
#   CONDITIONING_DIR    Dir with encoder_hidden_states.bin and context_latents.bin for Swift. Set by export_conditioning_for_swift.py.
#   SMOKE_DURATION      Duration in seconds for generated audio (default: 5.0).
#   SMOKE_STEPS         Number of diffusion steps (default: 8).
#   SMOKE_SEED          Random seed (default: 42).
#   SMOKE_EXTRA_TESTS   When set to 1, run additional parameter variation tests.
#
# Exit: 0 if every existing output file passes validation; non-zero otherwise.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/generation_smoke_output}"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Configurable generation parameters
SMOKE_DURATION="${SMOKE_DURATION:-5.0}"
SMOKE_STEPS="${SMOKE_STEPS:-8}"
SMOKE_SEED="${SMOKE_SEED:-42}"
SMOKE_EXTRA_TESTS="${SMOKE_EXTRA_TESTS:-0}"

echo "Generation params: duration=${SMOKE_DURATION}s, steps=${SMOKE_STEPS}, seed=${SMOKE_SEED}"

# Default config to acestep-v15-turbo; if no checkpoint found for it, try first available known config
ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-acestep-v15-turbo}"
CHECKPOINTS_DIR="$REPO_ROOT/checkpoints"
if [ ! -d "$CHECKPOINTS_DIR/$ACESTEP_CONFIG_PATH" ]; then
  for name in acestep-v15-turbo acestep-v15-base; do
    if [ -d "$CHECKPOINTS_DIR/$name" ]; then
      ACESTEP_CONFIG_PATH="$name"
      echo "Using checkpoint: $ACESTEP_CONFIG_PATH"
      break
    fi
  done
fi

# Python: generate python_out.wav when a valid checkpoint dir exists
if [ -d "$CHECKPOINTS_DIR/$ACESTEP_CONFIG_PATH" ]; then
  echo ""
  echo "=== Python Generation (config: $ACESTEP_CONFIG_PATH) ==="
  export ACESTEP_CONFIG_PATH
  if (cd "$REPO_ROOT" && python scripts/generate_audio.py --output-dir "$OUTPUT_DIR" --duration "$SMOKE_DURATION" --seed "$SMOKE_SEED"); then
    echo "Python generator: ok"
  else
    echo "Python generator: failed (continuing)"
  fi
else
  echo "Skipping Python (no checkpoint at checkpoints/$ACESTEP_CONFIG_PATH or other known config)"
fi

# Swift: generate swift_out.wav (test writes when OUTPUT_DIR is set).
# Use real DiT when checkpoint dir contains model.safetensors (same config as Python when available).
DIT_WEIGHTS_DIR="$CHECKPOINTS_DIR/$ACESTEP_CONFIG_PATH"
if [ -f "$DIT_WEIGHTS_DIR/model.safetensors" ]; then
  export DIT_WEIGHTS_PATH="$DIT_WEIGHTS_DIR"
  export TEST_RUNNER_DIT_WEIGHTS_PATH="$DIT_WEIGHTS_DIR"
  echo "Using real DiT: $DIT_WEIGHTS_PATH"
  
  # Preprocess .pt files to .safetensors for Swift compatibility
  echo "Preprocessing models for Swift..."
  if (cd "$REPO_ROOT" && python scripts/preprocess_models_for_swift.py --checkpoints-dir "$CHECKPOINTS_DIR" --model "$(basename "$DIT_WEIGHTS_DIR")" --verbose); then
    echo "Model preprocessing: ok"
  else
    echo "Model preprocessing: failed (some .pt files may not be converted)"
  fi
  if [ -f "$DIT_WEIGHTS_DIR/silence_latent.safetensors" ]; then
    echo "Using silence_latent.safetensors for source latents"
  fi
fi
# Precomputed conditioning for Swift (encoder + context_latents). Export every run when checkpoint exists so Swift uses fresh conditioning.
CONDITIONING_DIR="$OUTPUT_DIR/conditioning"
if [ -d "$CHECKPOINTS_DIR/$ACESTEP_CONFIG_PATH" ]; then
  echo "Exporting conditioning for Swift..."
  if (cd "$REPO_ROOT" && python scripts/export_conditioning_for_swift.py --output-dir "$CONDITIONING_DIR" 2>&1); then
    echo "Conditioning exported to $CONDITIONING_DIR"
  else
    echo "Conditioning export failed (Swift may use zeros and produce garbled audio)"
  fi
fi
if [ -f "$CONDITIONING_DIR/encoder_hidden_states.bin" ] && [ -f "$CONDITIONING_DIR/context_latents.bin" ]; then
  export CONDITIONING_DIR
  export TEST_RUNNER_CONDITIONING_DIR="$CONDITIONING_DIR"
  echo "Using precomputed conditioning: $CONDITIONING_DIR"
fi
# Use real VAE from checkpoints when decoder.safetensors exists (create with: python scripts/export_vae_decoder_mlx.py).
VAE_WEIGHTS="$CHECKPOINTS_DIR/vae/decoder.safetensors"
if [ -f "$VAE_WEIGHTS" ]; then
  export VAE_WEIGHTS_PATH="$VAE_WEIGHTS"
  export TEST_RUNNER_VAE_WEIGHTS_PATH="$VAE_WEIGHTS"
  echo "Using real VAE: $VAE_WEIGHTS_PATH"
else
  if [ -d "$CHECKPOINTS_DIR/vae" ]; then
    echo "Tip: run 'python scripts/export_vae_decoder_mlx.py' to create $VAE_WEIGHTS for real decoded audio"
  fi
fi

# Export generation params so Swift tests can use the same values
export SMOKE_DURATION
export SMOKE_STEPS
export SMOKE_SEED
export TEST_RUNNER_SMOKE_DURATION="$SMOKE_DURATION"
export TEST_RUNNER_SMOKE_STEPS="$SMOKE_STEPS"
export TEST_RUNNER_SMOKE_SEED="$SMOKE_SEED"

# Use xcodebuild test so Metal shaders (default.metallib) are built; "swift test" does not build them and triggers "Failed to load the default metallib".
# Xcode 15+ passes env vars to tests when prefixed with TEST_RUNNER_ (prefix stripped in the test process).
export OUTPUT_DIR
export TEST_RUNNER_OUTPUT_DIR="$OUTPUT_DIR"
SWIFT_OK=0
echo ""
echo "=== Swift Generation ==="
if command -v xcodebuild >/dev/null 2>&1; then
  echo "Running Swift generator (xcodebuild test for Metal)..."
  if (cd "$REPO_ROOT/AceStepSwift" && xcodebuild test -scheme AceStepSwift -destination 'platform=macOS' -only-testing:AceStepSwiftTests/GenerationSmokeTests/testGenerationWritesWaveformToOutputDir 2>&1); then
    SWIFT_OK=1
  fi
fi
if [ "$SWIFT_OK" -eq 0 ]; then
  echo "Running Swift generator (swift test fallback; Metal may be unavailable)..."
  if (cd "$REPO_ROOT/AceStepSwift" && swift test --filter GenerationSmokeTests 2>&1); then
    SWIFT_OK=1
  fi
fi
if [ "$SWIFT_OK" -eq 1 ]; then
  echo "Swift generator: ok"
else
  echo "Swift generator: failed (continuing)"
fi

# ---- Validation ----
echo ""
echo "=== Validation ==="

TO_VALIDATE=()
[ -f "$OUTPUT_DIR/python_out.wav" ] && TO_VALIDATE+=("$OUTPUT_DIR/python_out.wav")
[ -f "$OUTPUT_DIR/swift_out.wav" ] && TO_VALIDATE+=("$OUTPUT_DIR/swift_out.wav")

if [ ${#TO_VALIDATE[@]} -eq 0 ]; then
  echo "No output files to validate (python_out.wav, swift_out.wav missing)"
  exit 1
fi

echo "Validating waveform(s) with expected duration ~${SMOKE_DURATION}s..."
python "$REPO_ROOT/scripts/validate_audio.py" \
  --expected-duration "$SMOKE_DURATION" \
  --duration-tolerance 1.0 \
  --compare \
  "${TO_VALIDATE[@]}"
VALIDATION_RESULT=$?

# ---- Extra parameter variation tests (when enabled) ----
if [ "$SMOKE_EXTRA_TESTS" -eq 1 ] && [ -d "$CHECKPOINTS_DIR/$ACESTEP_CONFIG_PATH" ]; then
  echo ""
  echo "=== Extra Tests: Parameter Variations ==="
  
  # Test seed reproducibility (run same seed twice, compare)
  EXTRA_OUTPUT="$OUTPUT_DIR/extra_tests"
  mkdir -p "$EXTRA_OUTPUT"
  
  echo "Testing seed reproducibility (seed=$SMOKE_SEED)..."
  SEED_OK=true
  if (cd "$REPO_ROOT" && python scripts/generate_audio.py --output-dir "$EXTRA_OUTPUT" --duration 1.0 --seed "$SMOKE_SEED" --caption "seed test run 1" 2>&1); then
    mv "$EXTRA_OUTPUT/python_out.wav" "$EXTRA_OUTPUT/seed_run1.wav" 2>/dev/null || true
    if (cd "$REPO_ROOT" && python scripts/generate_audio.py --output-dir "$EXTRA_OUTPUT" --duration 1.0 --seed "$SMOKE_SEED" --caption "seed test run 1" 2>&1); then
      mv "$EXTRA_OUTPUT/python_out.wav" "$EXTRA_OUTPUT/seed_run2.wav" 2>/dev/null || true
      if [ -f "$EXTRA_OUTPUT/seed_run1.wav" ] && [ -f "$EXTRA_OUTPUT/seed_run2.wav" ]; then
        echo "Comparing seed reproducibility:"
        python "$REPO_ROOT/scripts/validate_audio.py" --compare "$EXTRA_OUTPUT/seed_run1.wav" "$EXTRA_OUTPUT/seed_run2.wav"
      fi
    fi
  fi
  
  # Test short duration (1s)
  echo ""
  echo "Testing short duration (1s)..."
  if (cd "$REPO_ROOT" && python scripts/generate_audio.py --output-dir "$EXTRA_OUTPUT" --duration 1.0 --seed "$SMOKE_SEED" --caption "short clip" 2>&1); then
    mv "$EXTRA_OUTPUT/python_out.wav" "$EXTRA_OUTPUT/short_1s.wav" 2>/dev/null || true
    if [ -f "$EXTRA_OUTPUT/short_1s.wav" ]; then
      python "$REPO_ROOT/scripts/validate_audio.py" --expected-duration 1.0 --duration-tolerance 0.5 "$EXTRA_OUTPUT/short_1s.wav"
    fi
  fi
  
  # Test longer duration (10s)
  echo ""
  echo "Testing longer duration (10s)..."
  if (cd "$REPO_ROOT" && python scripts/generate_audio.py --output-dir "$EXTRA_OUTPUT" --duration 10.0 --seed "$SMOKE_SEED" --caption "long clip" 2>&1); then
    mv "$EXTRA_OUTPUT/python_out.wav" "$EXTRA_OUTPUT/long_10s.wav" 2>/dev/null || true
    if [ -f "$EXTRA_OUTPUT/long_10s.wav" ]; then
      python "$REPO_ROOT/scripts/validate_audio.py" --expected-duration 10.0 --duration-tolerance 1.0 "$EXTRA_OUTPUT/long_10s.wav"
    fi
  fi
  
  echo ""
  echo "Extra tests complete."
fi

echo ""
if [ "$VALIDATION_RESULT" -eq 0 ]; then
  echo "=== All validations PASSED ==="
else
  echo "=== Some validations FAILED ==="
fi
exit $VALIDATION_RESULT
