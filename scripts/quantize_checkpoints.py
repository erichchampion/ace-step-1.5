#!/usr/bin/env python3
"""
Script to quantize models in the checkpoints/ directory to 8-bit and 4-bit
using mlx_lm.convert. Parallel directories will be created with the suffix
-8bit and -4bit for easy upload to Hugging Face.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    checkpoints_dir = Path("../checkpoints")
    if not checkpoints_dir.exists():
        print(f"Error: Directory '{checkpoints_dir}' does not exist.")
        sys.exit(1)

    print("Scanning checkpoints directory for models...\n")
    
    # Iterate over all directories in checkpoints/
    for item in checkpoints_dir.iterdir():
        if not item.is_dir():
            continue
            
        # Skip directories that already represent quantized models
        if "bit" in item.name:
            continue

        print(f"--- Processing Model: {item.name} ---")
        
        for bits in [8, 4]:
            output_dir = item.parent / f"{item.name}-{bits}bit"
            
            if output_dir.exists():
                print(f"  [{bits}-bit] Output directory '{output_dir}' already exists. Skipping.")
                continue
                
            print(f"  [{bits}-bit] Quantizing '{item.name}'...")
            
            # Note: mlx_lm.convert requires models to be in a supported format (like HF transformers).
            # We wrap it in a try-except block so the script continues on error (e.g., for standard VAEs).
            cmd = [
                "python", "-m", "mlx_lm.convert",
                "--hf-path", str(item),
                "-q",
                "--q-bits", str(bits),
                "--mlx-path", str(output_dir)
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"  [{bits}-bit] Successfully created '{output_dir}'.")
            except subprocess.CalledProcessError as e:
                print(f"  [{bits}-bit] Error quantizing '{item.name}'. (It may not be a supported LLM format: {e})")
        print("\n")

if __name__ == "__main__":
    main()
