#!/usr/bin/env python3
"""
Generate one audio clip via ACE-Step Python pipeline and save to a fixed filename.
Used by the generation smoke test so the shell script can find python_out.wav.

Requires: checkpoint under project checkpoints (default: acestep-v15-turbo, or first of acestep-v15-turbo, acestep-v15-base that exists).
Usage:
    python scripts/generate_audio.py --output-dir /path/to/dir [--config-path acestep-v15-turbo] [--duration 1.0] [--seed 42]

Exit: 0 if generation succeeded and at least one audio was saved; non-zero otherwise.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _project_root() -> Path:
    """Repo root (parent of directory containing this script)."""
    return Path(__file__).resolve().parent.parent


# Preferred config names when resolving a valid checkpoint (first existing wins).
DEFAULT_CONFIG_ORDER = ("acestep-v15-turbo", "acestep-v15-base")


def _resolve_config_path(config_path: str, root: Path) -> str:
    """Return config_path if checkpoints/<config_path> exists; else first in DEFAULT_CONFIG_ORDER that exists."""
    if (root / "checkpoints" / config_path).exists():
        return config_path
    for name in DEFAULT_CONFIG_ORDER:
        if (root / "checkpoints" / name).exists():
            return name
    return config_path  # leave as-is so init_service can report the error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate one audio clip for smoke test; writes python_out.wav to --output-dir."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write python_out.wav",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo"),
        help="Model config name (e.g. acestep-v15-turbo). Default: ACESTEP_CONFIG_PATH or acestep-v15-turbo; script will use first valid checkpoint in checkpoints/ if this one is missing.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Target duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="Smoke test",
        help="Caption for generation (default: Smoke test)",
    )
    args = parser.parse_args()

    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Use acestep-v15-turbo by default; if that checkpoint is missing, use first valid one found
    config_path = _resolve_config_path(args.config_path, root)
    if config_path != args.config_path:
        print(f"Using checkpoint: {config_path} (requested '{args.config_path}' not found)", file=sys.stderr)

    from acestep.handler import AceStepHandler
    from acestep.inference import (
        GenerationConfig,
        GenerationParams,
        generate_music,
    )

    project_root_str = str(root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_dir = args.output_dir

    handler = AceStepHandler()
    msg, ok = handler.initialize_service(
        project_root=project_root_str,
        config_path=config_path,
        device="auto",
    )
    if not ok:
        print(f"Python generator: init failed: {msg}", file=sys.stderr)
        return 1

    params = GenerationParams(
        caption=args.caption,
        lyrics="[Instrumental]",
        duration=args.duration,
        inference_steps=4,
        seed=args.seed,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
    )
    config = GenerationConfig(
        batch_size=1,
        use_random_seed=False,
        seeds=args.seed,
        audio_format="wav",
    )

    result = generate_music(
        dit_handler=handler,
        llm_handler=None,
        params=params,
        config=config,
        save_dir=str(save_dir),
        progress=None,
    )

    if not result.success:
        print(f"Python generator: {result.error or 'unknown error'}", file=sys.stderr)
        return 1
    if not result.audios:
        print("Python generator: no audios in result", file=sys.stderr)
        return 1

    first = result.audios[0]
    src = first.get("path") or ""
    if not src or not Path(src).exists():
        print("Python generator: first audio path missing or file not found", file=sys.stderr)
        return 1

    dest = save_dir / "python_out.wav"
    shutil.copy2(src, dest)
    print(f"Python generator: wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
