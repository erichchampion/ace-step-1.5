# Scripts

This directory contains helper scripts for the ACE-Step repo, including the **generation smoke test** (Python + Swift generation and waveform validation).

## Generation smoke test: Python venv setup

The smoke test runs Python scripts from the **repository root**. Use a virtual environment so that `generate_audio.py` and `validate_audio.py` have access to the `acestep` package and its dependencies.

### 1. Create and activate a venv (repository root)

From the **repository root** (parent of `scripts/`):

```bash
# Create venv at repo root (recommended: .venv or venv)
python3 -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

With the venv activated and from the **repository root**:

```bash
pip install -r requirements.txt
```

This installs PyTorch and everything needed for `generate_audio.py` and `validate_audio.py` (e.g. `soundfile` or `scipy` for WAV validation). The smoke test runs Python from the repo root so the `acestep` package is found via `sys.path`; optionally install the tree in editable mode with `pip install -e .` so `acestep` is available from any working directory.

### 3. Run the smoke test

From the **repository root**, with the venv **activated**:

```bash
./scripts/run_generation_smoke_test.sh 2>&1 | tee debug.txt
```

The script will use the activated `python` (from the venv). It runs the Python generator (if a checkpoint exists under `checkpoints/`), the Swift generator, and validates any produced WAV files.

**Swift step and Metal:** The Swift step uses `xcodebuild test` (when available) so that MLX’s Metal shaders (`default.metallib`) are built and loaded. Plain `swift test` does not build these shaders and can fail with “Failed to load the default metallib”. If `xcodebuild` is not available, the script falls back to `swift test` (the Swift step may then fail in environments where Metal is not set up).

**Swift step and real DiT/VAE/conditioning:** For Swift output to match Python (clear note instead of garbled), the Swift step needs (1) loaded DiT weights, (2) real VAE decoder, and (3) real conditioning (encoder hidden states + context latents). The script sets `DIT_WEIGHTS_PATH` when `checkpoints/<config>/model.safetensors` exists, and `VAE_WEIGHTS_PATH` when `checkpoints/vae/decoder.safetensors` exists. Conditioning is exported once by `export_conditioning_for_swift.py` into `<OUTPUT_DIR>/conditioning`; the script runs that automatically when the conditioning dir is missing and a checkpoint exists, then sets `CONDITIONING_DIR` so the Swift test uses the same conditioning as Python.

#### Using the real VAE (recommended for a passing Swift smoke test)

1. **Have the PyTorch VAE in the checkpoints tree.**
   Ensure the HuggingFace-style VAE is at `checkpoints/vae` (same layout used by the Python pipeline and `scripts/prepare_vae_calibration_data.py`).

2. **Export the decoder once for Swift.**
   From the **repository root**, with the venv **activated**:
   ```bash
   python scripts/export_vae_decoder_mlx.py
   ```
   This reads `checkpoints/vae` and writes `checkpoints/vae/decoder.safetensors` in MLX layout for Swift’s `MLXVAEDecoder`. You only need to run this once (or after updating the VAE checkpoint).

3. **Run the smoke test.**
   When `checkpoints/vae/decoder.safetensors` exists, `run_generation_smoke_test.sh` sets `VAE_WEIGHTS_PATH` and the Swift test loads the real VAE:
   ```bash
   ./scripts/run_generation_smoke_test.sh
   ```

**Optional:** Custom paths:
```bash
python scripts/export_vae_decoder_mlx.py --vae-dir /path/to/vae --output /path/to/decoder.safetensors
```
Then set `VAE_WEIGHTS_PATH` to the output file when running the Swift step (e.g. `export VAE_WEIGHTS_PATH=/path/to/decoder.safetensors` before the script, or pass it into your test runner).

#### Precomputed conditioning (Swift parity with Python)

For Swift to produce the same “clear note” as Python, it must use the same DiT weights, VAE, **and** conditioning (encoder hidden states + context latents). The script exports conditioning once:

1. On first run (or when `$OUTPUT_DIR/conditioning/context_latents.bin` is missing), it runs `python scripts/export_conditioning_for_swift.py --output-dir "$OUTPUT_DIR/conditioning"`. That runs one Python generation and writes `encoder_hidden_states.bin` and `context_latents.bin` in a format Swift can load.
2. When those files exist, the script sets `CONDITIONING_DIR` (and `TEST_RUNNER_CONDITIONING_DIR`) so the Swift test loads them instead of zeros.

To export conditioning manually (e.g. for a different caption/lyrics/duration):

```bash
python scripts/export_conditioning_for_swift.py --output-dir ./generation_smoke_output/conditioning [--caption "..." --lyrics "..." --duration 1.0]
```

Then run the smoke test; Swift will use the precomputed conditioning when `CONDITIONING_DIR` points to that directory.

### Optional: venv in `scripts/`

If you prefer the venv to live inside `scripts/`:

```bash
cd scripts
python3 -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r ../requirements.txt
cd ..
./scripts/run_generation_smoke_test.sh
```

Ensure the venv is **activated** before running `run_generation_smoke_test.sh` so that `python` in the script points at the venv.

### Summary

| Step | Where | Command |
|------|--------|---------|
| Create venv | Repo root | `python3 -m venv .venv` |
| Activate | Repo root | `source .venv/bin/activate` (or Windows equivalent) |
| Install deps | Repo root | `pip install -r requirements.txt` |
| (Optional) Export VAE for Swift | Repo root | `python scripts/export_vae_decoder_mlx.py` |
| (Optional) Export conditioning for Swift | Repo root | `python scripts/export_conditioning_for_swift.py --output-dir <dir>` (or let the smoke script do it) |
| Run smoke test | Repo root | `./scripts/run_generation_smoke_test.sh` |

Without an activated venv (or with a venv that lacks the repo dependencies), the **Python** step of the smoke test will fail when it tries to import `acestep`. The **Swift** step and **validator** (stdlib-only for 16-bit WAV) can still run.
