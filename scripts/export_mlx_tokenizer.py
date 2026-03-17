#!/usr/bin/env python3
"""
Export AceStep audio tokenizer and detokenizer weights from a full checkpoint
into separate safetensors files for MLX loading in Swift.

Usage:
    python scripts/export_mlx_tokenizer.py --checkpoint checkpoints/acestep-v15-turbo --output mlx_tokenizer_export

Outputs (in --output directory):
    audio_tokenizer/
        audio_tokenizer.safetensors    (tokenizer.* weights)
        tokenizer_config.json          (FSQ + architecture config)
    audio_detokenizer/
        audio_detokenizer.safetensors  (detokenizer.* weights)
        tokenizer_config.json          (same config, included for convenience)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file


# Weight key prefixes in the full AceStep checkpoint
TOKENIZER_PREFIX = "tokenizer."
DETOKENIZER_PREFIX = "detokenizer."


def find_checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    """Find all safetensors weight files in a checkpoint directory."""
    files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not files:
        # Try model subdirectory
        files = sorted(checkpoint_dir.glob("model*.safetensors"))
    return files


def load_full_checkpoint(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load all weights from a checkpoint directory."""
    files = find_checkpoint_files(checkpoint_dir)
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}")

    all_weights = {}
    for f in files:
        print(f"  Loading {f.name}...")
        weights = load_file(str(f))
        all_weights.update(weights)
    print(f"  Total keys: {len(all_weights)}")
    return all_weights


def extract_weights(all_weights: dict, prefix: str) -> dict[str, torch.Tensor]:
    """Extract weights with a given prefix, stripping the prefix from keys."""
    extracted = {}
    for key, value in all_weights.items():
        if key.startswith(prefix):
            stripped_key = key[len(prefix):]
            extracted[stripped_key] = value
    return extracted


def build_config(checkpoint_dir: Path) -> dict:
    """Build tokenizer config from the checkpoint's config.json."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")

    with open(config_path) as f:
        full_config = json.load(f)

    return {
        "hidden_size": full_config["hidden_size"],
        "audio_acoustic_hidden_dim": full_config["audio_acoustic_hidden_dim"],
        "pool_window_size": full_config["pool_window_size"],
        "fsq_dim": full_config["fsq_dim"],
        "fsq_input_levels": full_config["fsq_input_levels"],
        "fsq_input_num_quantizers": full_config["fsq_input_num_quantizers"],
        "num_attention_pooler_hidden_layers": full_config["num_attention_pooler_hidden_layers"],
        "num_attention_heads": full_config["num_attention_heads"],
        "num_key_value_heads": full_config["num_key_value_heads"],
        "head_dim": full_config["head_dim"],
        "intermediate_size": full_config["intermediate_size"],
        "rms_norm_eps": full_config["rms_norm_eps"],
        "rope_theta": full_config["rope_theta"],
        "max_position_embeddings": full_config["max_position_embeddings"],
        "use_sliding_window": full_config.get("use_sliding_window", True),
        "sliding_window": full_config.get("sliding_window", 128),
    }


def convert_to_float32(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert all weights to float32 for MLX compatibility."""
    return {k: v.to(torch.float32).contiguous() for k, v in weights.items()}


def main():
    parser = argparse.ArgumentParser(description="Export AceStep tokenizer/detokenizer for MLX")
    parser.add_argument("--checkpoint", required=True, help="Path to full AceStep checkpoint directory")
    parser.add_argument("--output", default="mlx_tokenizer_export", help="Output directory")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    output_dir = Path(args.output)

    if not checkpoint_dir.is_dir():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    print(f"Loading checkpoint from {checkpoint_dir}...")
    all_weights = load_full_checkpoint(checkpoint_dir)

    # Extract tokenizer weights
    print(f"\nExtracting tokenizer weights (prefix: '{TOKENIZER_PREFIX}')...")
    tokenizer_weights = extract_weights(all_weights, TOKENIZER_PREFIX)
    if not tokenizer_weights:
        print("ERROR: No tokenizer weights found!")
        sys.exit(1)
    print(f"  Found {len(tokenizer_weights)} tokenizer parameters:")
    for key, value in sorted(tokenizer_weights.items()):
        print(f"    {key}: {list(value.shape)} ({value.dtype})")

    # Extract detokenizer weights
    print(f"\nExtracting detokenizer weights (prefix: '{DETOKENIZER_PREFIX}')...")
    detokenizer_weights = extract_weights(all_weights, DETOKENIZER_PREFIX)
    if not detokenizer_weights:
        print("ERROR: No detokenizer weights found!")
        sys.exit(1)
    print(f"  Found {len(detokenizer_weights)} detokenizer parameters:")
    for key, value in sorted(detokenizer_weights.items()):
        print(f"    {key}: {list(value.shape)} ({value.dtype})")

    # Build config
    print("\nBuilding config...")
    config = build_config(checkpoint_dir)
    print(f"  Config: {json.dumps(config, indent=2)}")

    # Convert weights to float32
    tokenizer_weights = convert_to_float32(tokenizer_weights)
    detokenizer_weights = convert_to_float32(detokenizer_weights)

    # Save tokenizer
    tokenizer_dir = output_dir / "audio_tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer_dir / "audio_tokenizer.safetensors"
    save_file(tokenizer_weights, str(tokenizer_path))
    tokenizer_size = tokenizer_path.stat().st_size
    print(f"\n✅ Saved tokenizer: {tokenizer_path} ({tokenizer_size / 1024 / 1024:.1f} MB)")

    config_path = tokenizer_dir / "tokenizer_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"   Saved config: {config_path}")

    # Save detokenizer
    detokenizer_dir = output_dir / "audio_detokenizer"
    detokenizer_dir.mkdir(parents=True, exist_ok=True)
    detokenizer_path = detokenizer_dir / "audio_detokenizer.safetensors"
    save_file(detokenizer_weights, str(detokenizer_path))
    detokenizer_size = detokenizer_path.stat().st_size
    print(f"\n✅ Saved detokenizer: {detokenizer_path} ({detokenizer_size / 1024 / 1024:.1f} MB)")

    config_path = detokenizer_dir / "tokenizer_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"   Saved config: {config_path}")

    total_size = (tokenizer_size + detokenizer_size) / 1024 / 1024
    print(f"\nTotal export size: {total_size:.1f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
