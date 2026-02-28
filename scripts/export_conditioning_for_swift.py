#!/usr/bin/env python3
"""
Export DiT conditioning (encoder_hidden_states, context_latents) for the Swift smoke test.
Runs one Python generation with the same params as the smoke test, then writes
encoder_hidden_states.bin and context_latents.bin in the format Swift loadPrecomputedConditioning expects:
  - 3 x int32 (little-endian): shape (d0, d1, d2)
  - then float32 (little-endian) row-major array.

Usage:
  python scripts/export_conditioning_for_swift.py --output-dir <dir>

Output dir will contain encoder_hidden_states.bin, context_latents.bin, and (when MLX
is available) initial_noise.bin (same seed 42, so Swift can use identical initial noise).
Set CONDITIONING_DIR (or TEST_RUNNER_CONDITIONING_DIR) to this dir when running the smoke test.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

DEFAULT_CONFIG_ORDER = ("acestep-v15-turbo", "acestep-v15-base")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_config_path(config_path: str, root: Path) -> str:
    """Return config_path if checkpoints/<config_path> exists; else first in DEFAULT_CONFIG_ORDER."""
    if (root / "checkpoints" / config_path).exists():
        return config_path
    for name in DEFAULT_CONFIG_ORDER:
        if (root / "checkpoints" / name).exists():
            return name
    return config_path


def _write_bin(path: Path, tensor) -> None:
    """Write a 3D tensor to path: 3 x int32 LE (shape) then float32 LE (flat)."""
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().float().numpy()
    else:
        import numpy as np
        arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {arr.shape}")
    d0, d1, d2 = arr.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<iii", d0, d1, d2))
        f.write(arr.astype("<f4").tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export conditioning tensors for Swift smoke test."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write encoder_hidden_states.bin and context_latents.bin",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="Smoke test",
        help="Caption used for conditioning (default: same as smoke test)",
    )
    parser.add_argument(
        "--lyrics",
        type=str,
        default="[Instrumental]",
        help="Lyrics used for conditioning (default: same as smoke test)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration in seconds (default: 1.0, must match Swift smoke test)",
    )
    args = parser.parse_args()
    root = _project_root()
    out_dir = args.output_dir.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    config_path = _resolve_config_path(
        os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo"), root
    )

    try:
        from acestep.handler import AceStepHandler
        from acestep.inference import (
            GenerationConfig,
            GenerationParams,
            generate_music,
        )
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        return 1

    project_root_str = str(root)
    handler = AceStepHandler()
    msg, ok = handler.initialize_service(
        project_root=project_root_str,
        config_path=config_path,
        device="auto",
    )
    if not ok:
        print(f"Handler init failed: {msg}", file=sys.stderr)
        return 1

    # Match generate_audio.py exactly so Swift conditioning matches python_out.wav (same prompt + flags).
    params = GenerationParams(
        caption=args.caption,
        lyrics=args.lyrics,
        duration=args.duration,
        inference_steps=4,
        seed=42,
        shift=3.0,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=42)

    result = generate_music(
        dit_handler=handler,
        llm_handler=None,
        params=params,
        config=config,
        save_dir=None,
        progress=None,
    )

    # Support both GenerationResult (dataclass) and legacy dict return
    if hasattr(result, "success"):
        success = result.success
        error_msg = getattr(result, "error", None) or "unknown"
        extra = getattr(result, "extra_outputs", None) or {}
    else:
        success = result.get("success", False)
        error_msg = result.get("error", "unknown")
        extra = result.get("extra_outputs") or {}

    print(f"[export_conditioning] result type={type(result).__name__}, success={success}", file=sys.stderr)
    print(f"[export_conditioning] extra_outputs keys: {list(extra.keys())}", file=sys.stderr)

    if not success:
        print(f"Generation failed: {error_msg}", file=sys.stderr)
        return 1

    enc = extra.get("encoder_hidden_states")
    ctx = extra.get("context_latents")
    if enc is None or ctx is None:
        print(
            "extra_outputs missing encoder_hidden_states or context_latents",
            file=sys.stderr,
        )
        return 1

    _write_bin(out_dir / "encoder_hidden_states.bin", enc)
    _write_bin(out_dir / "context_latents.bin", ctx)

    # Export initial noise (same seed 42, shape [1, T, 64]) so Swift can use identical noise.
    t_len = int(ctx.shape[1])
    try:
        import numpy as np
        import mlx.core as mx
        key = mx.random.key(42)
        noise_mlx = mx.random.normal((1, t_len, 64), key=key)
        noise_np = np.asarray(noise_mlx, dtype=np.float32)
        _write_bin(out_dir / "initial_noise.bin", noise_np)
        print(f"  initial_noise.bin shape (1, {t_len}, 64)", file=sys.stderr)
    except Exception as e:
        print(f"  Skipping initial_noise.bin: {e}", file=sys.stderr)

    print(f"Wrote conditioning to {out_dir}")
    print(f"  encoder_hidden_states.bin shape {tuple(enc.shape)}")
    print(f"  context_latents.bin shape {tuple(ctx.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
