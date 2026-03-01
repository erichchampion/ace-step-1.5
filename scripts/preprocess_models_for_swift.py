#!/usr/bin/env python3
"""
Preprocess model checkpoints for Swift compatibility.

Converts .pt files (pickle format) to .safetensors format so they can be loaded
by Swift MLX. This is required because Swift cannot load .pt files.

Usage:
    python scripts/preprocess_models_for_swift.py --checkpoints-dir ./checkpoints
    python scripts/preprocess_models_for_swift.py --checkpoints-dir ./checkpoints --model acestep-v15-turbo
    python scripts/preprocess_models_for_swift.py --checkpoints-dir ./checkpoints --dry-run
    python scripts/preprocess_models_for_swift.py --checkpoints-dir ./checkpoints --force
"""

import argparse
import os
import sys
from pathlib import Path


def find_pt_files(
    checkpoints_dir: Path, model_name: str | None = None
) -> list[tuple[Path, Path]]:
    """
    Find all .pt files in model directories.

    Args:
        checkpoints_dir: Root directory containing model folders
        model_name: Optional specific model to process

    Returns:
        List of (pt_file, model_dir) tuples
    """
    results = []

    if model_name:
        model_dir = checkpoints_dir / model_name
        if model_dir.exists():
            for pt_file in model_dir.rglob("*.pt"):
                results.append((pt_file, model_dir))
    else:
        for model_dir in checkpoints_dir.iterdir():
            if model_dir.is_dir():
                for pt_file in model_dir.rglob("*.pt"):
                    results.append((pt_file, model_dir))

    return results


def convert_pt_to_safetensors(
    pt_file: Path,
    model_dir: Path,
    force: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Convert a .pt file to .safetensors format.

    Args:
        pt_file: Path to the .pt file
        model_dir: Directory containing the model
        force: Overwrite existing .safetensors files
        verbose: Print detailed progress

    Returns:
        True if conversion succeeded, False otherwise
    """
    safetensors_file = pt_file.with_suffix(".safetensors")

    if safetensors_file.exists() and not force:
        if verbose:
            print(
                f"  Skipping {pt_file.name}: .safetensors already exists (use --force to overwrite)"
            )
        return True

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"  ERROR: Required package not available: {e}")
        return False

    try:
        if verbose:
            print(f"  Loading {pt_file}...")
        data = torch.load(pt_file, weights_only=True)

        if verbose:
            if hasattr(data, "shape"):
                print(f"  Loaded tensor shape: {data.shape}")
            elif isinstance(data, dict):
                print(f"  Loaded dict with keys: {list(data.keys())[:5]}...")

        if verbose:
            print(f"  Saving to {safetensors_file}...")

        if isinstance(data, dict):
            save_file(data, safetensors_file)
        else:
            save_file({"data": data}, safetensors_file)

        if verbose:
            print(f"  Verifying {safetensors_file}...")
        _ = torch.load(safetensors_file, weights_only=True)

        print(f"  Converted: {pt_file.name} -> {safetensors_file.name}")
        return True

    except Exception as e:
        print(f"  ERROR: Failed to convert {pt_file}: {e}")
        return False


def preprocess_models(
    checkpoints_dir: Path,
    model_name: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Preprocess all .pt files in the checkpoints directory.

    Args:
        checkpoints_dir: Root directory containing model folders
        model_name: Optional specific model to process
        dry_run: Show what would be converted without doing it
        force: Overwrite existing .safetensors files
        verbose: Print detailed progress

    Returns:
        True if all conversions succeeded, False otherwise
    """
    if not checkpoints_dir.exists():
        print(f"ERROR: Checkpoints directory not found: {checkpoints_dir}")
        return False

    pt_files = find_pt_files(checkpoints_dir, model_name)

    if not pt_files:
        print(
            f"No .pt files found in {checkpoints_dir}"
            + (f" for model {model_name}" if model_name else "")
        )
        return True

    print(
        f"Found {len(pt_files)} .pt file(s)"
        + (f" in {model_name}" if model_name else "")
    )

    if dry_run:
        print("\nDry run - would convert:")
        for pt_file, model_dir in pt_files:
            safetensors_file = pt_file.with_suffix(".safetensors")
            existing = " (exists)" if safetensors_file.exists() else ""
            print(
                f"  {pt_file.relative_to(checkpoints_dir)} -> {safetensors_file.name}{existing}"
            )
        return True

    success = True
    for pt_file, model_dir in pt_files:
        rel_path = pt_file.relative_to(checkpoints_dir)
        print(f"Processing: {rel_path}")
        if not convert_pt_to_safetensors(
            pt_file, model_dir, force=force, verbose=verbose
        ):
            success = False

    return success


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert .pt files to .safetensors for Swift compatibility"
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        required=True,
        help="Root directory containing model folders (e.g., ./checkpoints)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Process specific model only (e.g., acestep-v15-turbo)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without doing it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .safetensors files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    success = preprocess_models(
        checkpoints_dir=args.checkpoints_dir,
        model_name=args.model,
        dry_run=args.dry_run,
        force=args.force,
        verbose=args.verbose,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
