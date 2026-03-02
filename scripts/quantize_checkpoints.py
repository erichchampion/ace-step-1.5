#!/usr/bin/env python3
"""
Script to quantize models in the checkpoints/ directory to 8-bit and 4-bit
using mlx_lm.convert. Parallel directories will be created with the suffix
-8bit and -4bit for easy upload to Hugging Face.
"""

import os
import subprocess
import sys
import json
from pathlib import Path

# Model types that mlx_lm currently does not support, to avoid ugly tracebacks
UNSUPPORTED_MODEL_TYPES = {"acestep", "qwen3"}

def main():
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        print(f"Error: Directory '{checkpoints_dir}' does not exist.")
        sys.exit(1)

    quantized_dir = Path("quantized_checkpoints")
    quantized_dir.mkdir(exist_ok=True)

    print("Scanning checkpoints directory for models...\n")
    
    # Iterate over all directories in checkpoints/
    for item in checkpoints_dir.iterdir():
        if not item.is_dir():
            continue
            
        # Skip directories that already represent quantized models
        if "bit" in item.name:
            continue

        print(f"--- Processing Model: {item.name} ---")

        # Check for config.json to see if we should gracefully skip
        config_path = item / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                model_type = config.get("model_type")
                if model_type in UNSUPPORTED_MODEL_TYPES:
                    print(f"  [Skip] model_type '{model_type}' is not currently supported by mlx_lm. Skipping.\n")
                    continue
            except Exception as e:
                print(f"  [Warning] Could not read config.json: {e}")
        else:
            print("  [Warning] No config.json found. Will attempt to quantize anyway, but this may fail.")

        for bits in [8, 4]:
            output_dir = quantized_dir / f"{item.name}-{bits}bit"
            
            if output_dir.exists():
                print(f"  [{bits}-bit] Output directory '{output_dir}' already exists. Skipping.")
                continue
                
            print(f"  [{bits}-bit] Quantizing '{item.name}'...")
            
            # Note: mlx_lm.convert requires models to be in a supported format (like HF transformers).
            # We wrap it in a try-except block so the script continues on error (e.g., for standard VAEs).
            # We also capture stderr to gracefully report issues rather than dumping a massive traceback.
            cmd = [
                "python", "-m", "mlx_lm.convert",
                "--hf-path", str(item),
                "-q",
                "--q-bits", str(bits),
                "--mlx-path", str(output_dir)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  [{bits}-bit] Successfully created '{output_dir}'.")
                else:
                    # Print a condensed error message rather than the full traceback on the main thread
                    # Extract just the last line or the ValueError section if possible to be cleaner
                    errors = result.stderr.strip().split("\n")
                    last_error = errors[-1] if errors else "Unknown Error"
                    print(f"  [{bits}-bit] Error quantizing '{item.name}'. ({last_error})")
            except Exception as e:
                print(f"  [{bits}-bit] Unexpected error quantizing '{item.name}': {e}")
        print("\n")

if __name__ == "__main__":
    main()
