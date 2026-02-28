#!/usr/bin/env python3
"""
Export VAE decoder weights to a safetensors file for Swift MLXVAEDecoder.
Reads PyTorch VAE from checkpoints/vae, converts to MLX layout (decoder.* keys),
and writes decoder.safetensors so Swift can load it via loadWeights(from:).

Usage:
    python scripts/export_vae_decoder_mlx.py [--vae-dir checkpoints/vae] [--output checkpoints/vae/decoder.safetensors]

Exit: 0 on success; non-zero on missing VAE or write failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export VAE decoder to safetensors for Swift MLXVAEDecoder."
    )
    parser.add_argument(
        "--vae-dir",
        type=Path,
        default=None,
        help="Path to PyTorch VAE (default: <repo>/checkpoints/vae)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output safetensors path (default: <vae-dir>/decoder.safetensors)",
    )
    args = parser.parse_args()
    root = _project_root()
    vae_dir = args.vae_dir or root / "checkpoints" / "vae"
    vae_dir = vae_dir.resolve()
    if not vae_dir.is_dir():
        print(f"VAE directory not found: {vae_dir}", file=sys.stderr)
        return 1
    out_path = args.output or vae_dir / "decoder.safetensors"
    out_path = out_path.resolve()

    try:
        from diffusers.models import AutoencoderOobleck
    except ImportError as e:
        print(f"diffusers not available: {e}", file=sys.stderr)
        return 1
    try:
        from acestep.models.mlx.vae_convert import convert_vae_weights
    except ImportError as e:
        print(f"acestep.models.mlx.vae_convert not available: {e}", file=sys.stderr)
        return 1
    try:
        from safetensors.numpy import save_file
    except ImportError as e:
        print(f"safetensors not available: {e}", file=sys.stderr)
        return 1
    import numpy as np

    print(f"Loading VAE from {vae_dir}...")
    vae = AutoencoderOobleck.from_pretrained(str(vae_dir))
    weights = convert_vae_weights(vae)
    decoder_tensors = {}
    for key, mx_arr in weights:
        if not key.startswith("decoder."):
            continue
        arr = np.asarray(mx_arr)
        decoder_tensors[key] = arr.astype(np.float32)
    if not decoder_tensors:
        print("No decoder.* keys in converted weights", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(decoder_tensors, str(out_path))
    print(f"Wrote {len(decoder_tensors)} tensors to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
