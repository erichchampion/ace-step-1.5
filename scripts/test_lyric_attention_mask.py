#!/usr/bin/env python3
"""
Test if lyric_attention_mask=None affects the text encoder output.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add acestep to path
sys.path.insert(0, str(Path(__file__).parent / "acestep"))

from transformers import AutoModel, AutoTokenizer


def load_text_encoder(checkpoint_dir: str, device: str = "cpu"):
    """Load text encoder and tokenizer."""
    text_encoder_path = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
    model = AutoModel.from_pretrained(text_encoder_path)
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    return model, tokenizer


def test_lyric_attention_mask(model, tokenizer, text: str):
    """Test if lyric_attention_mask parameter affects output."""
    inputs = tokenizer(
        text,
        padding="longest",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids.tolist()}")

    # Test 1: Without lyric_attention_mask
    print("\n=== Test 1: Without lyric_attention_mask ===")
    with torch.inference_mode():
        outputs1 = model(input_ids)
        hidden1 = outputs1.last_hidden_state
    print(f"Output shape: {hidden1.shape}")
    print(f"Output mean: {hidden1.mean().item():.6f}")
    print(f"Output std: {hidden1.std().item():.6f}")

    # Test 2: With lyric_attention_mask=None
    print("\n=== Test 2: With lyric_attention_mask=None ===")
    with torch.inference_mode():
        try:
            outputs2 = model(input_ids, lyric_attention_mask=None)
            hidden2 = outputs2.last_hidden_state
            print(f"Output shape: {hidden2.shape}")
            print(f"Output mean: {hidden2.mean().item():.6f}")
            print(f"Output std: {hidden2.std().item():.6f}")

            # Compare
            diff = (hidden1 - hidden2).abs().max().item()
            print(f"\nMax difference: {diff:.10f}")
            if diff < 1e-5:
                print("=> Outputs are IDENTICAL (lyric_attention_mask=None is ignored)")
            else:
                print("=> Outputs are DIFFERENT!")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    checkpoint_dir = "./checkpoints"
    text = """# Instruction
Fill the audio semantic mask based on the given conditions:

# Caption
A beautiful piano melody

# Metas
language=en
instrumental=true
bpm=120
key=C
timesig=4/4
duration=30<|endoftext|>"""

    print("Loading text encoder...")
    model, tokenizer = load_text_encoder(checkpoint_dir)

    test_lyric_attention_mask(model, tokenizer, text)
