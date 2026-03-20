import os
import argparse
from pathlib import Path
from safetensors.torch import load_file, save_file

def main():
    parser = argparse.ArgumentParser(description="Extract ConditionEncoder weights from a full model.")
    parser.add_argument("--input", type=str, required=True, help="Path to input model.safetensors")
    parser.add_argument("--output", type=str, required=True, help="Path to save encoder.safetensors")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    try:
        tensors = load_file(args.input)
    except Exception as e:
        print(f"Failed to load input file: {e}")
        return

    encoder_tensors = {}
    null_tensors = {}
    for key, tensor in tensors.items():
        if key.startswith("encoder."):
            encoder_tensors[key] = tensor
        if "null_condition_emb" in key:
            null_tensors[key] = tensor

    if not encoder_tensors:
        print("No keys starting with 'encoder.' found in model.safetensors.")
    else:
        print(f"Extracted {len(encoder_tensors)} encoder tensors.")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(encoder_tensors, out_path)
        print(f"Saved to {args.output}")

    if null_tensors:
        print(f"Extracted {len(null_tensors)} null conditioning tensors.")
        null_path = Path(args.output).parent / "null_condition_embedding.safetensors"
        save_file(null_tensors, null_path)
        print(f"Saved to {null_path}")
    else:
        print("No null_condition_emb keys found.")

if __name__ == "__main__":
    main()
