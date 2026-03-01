#!/usr/bin/env python3
"""
Export reference encoder_hidden_states for debugging Swift vs Python differences.
Run from ace-step-1.5 directory.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add acestep to path
sys.path.insert(0, str(Path(__file__).parent / "acestep"))

from acestep.constants import DEFAULT_DIT_INSTRUCTION, SFT_GEN_PROMPT


def format_instruction(instruction: str) -> str:
    """Ensure instruction ends with a colon."""
    if not instruction.endswith(":"):
        instruction = instruction + ":"
    return instruction


def format_lyrics(lyrics: str, language: str) -> str:
    """Format lyrics text with language header."""
    return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"


def parse_metas(metas_str: str) -> str:
    """Parse metadata string."""
    return metas_str


def build_text_prompt(instruction: str, caption: str, metas: str) -> str:
    """Build text prompt matching Swift's format."""
    return SFT_GEN_PROMPT.format(instruction, caption, metas)


def build_lyrics_prompt(lyrics: str, language: str) -> str:
    """Build lyrics prompt matching Swift's format."""
    return format_lyrics(lyrics, language)


def load_text_encoder(checkpoint_dir: str, device: str = "cpu"):
    """Load text encoder and tokenizer."""
    from transformers import AutoModel, AutoTokenizer

    text_encoder_path = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")
    if not os.path.exists(text_encoder_path):
        raise FileNotFoundError(f"Text encoder not found at {text_encoder_path}")

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
    model = AutoModel.from_pretrained(text_encoder_path)
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    return model, tokenizer


def encode_text(model, tokenizer, text: str, device: str = "cpu") -> torch.Tensor:
    """Encode text and return hidden states."""
    inputs = tokenizer(
        text,
        padding="longest",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)

    with torch.inference_mode():
        outputs = model(input_ids)
        hidden_states = outputs.last_hidden_state

    return hidden_states


def encode_lyrics(model, tokenizer, text: str, device: str = "cpu") -> torch.Tensor:
    """Encode lyrics using embed_tokens."""
    inputs = tokenizer(
        text,
        padding="longest",
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    )
    token_ids = inputs.input_ids.to(device)

    with torch.inference_mode():
        # Use embed_tokens directly (like Python's infer_lyric_embeddings)
        hidden_states = model.embed_tokens(token_ids)

    return hidden_states


def compute_stats(tensor: torch.Tensor) -> dict:
    """Compute statistics for a tensor."""
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def export_reference(
    checkpoint_dir: str,
    instruction: str,
    caption: str,
    lyrics: str,
    language: str,
    metas: str,
    output_path: str,
):
    """Export reference encoder_hidden_states for a given prompt."""
    device = "cpu"  # Use CPU for consistent results

    print(f"Loading text encoder from {checkpoint_dir}...")
    model, tokenizer = load_text_encoder(checkpoint_dir, device)

    # Build prompts matching Swift format
    text_prompt = build_text_prompt(instruction, caption, metas)
    lyrics_prompt = build_lyrics_prompt(lyrics, language)

    print(f"\n=== Text Prompt ===")
    print(text_prompt)
    print(f"\n=== Lyrics Prompt ===")
    print(lyrics_prompt)

    # Encode text
    print(f"\nEncoding text...")
    text_hidden_states = encode_text(model, tokenizer, text_prompt, device)
    text_stats = compute_stats(text_hidden_states)
    print(f"Text hidden states: {text_stats}")

    # Encode lyrics
    print(f"\nEncoding lyrics...")
    lyric_hidden_states = encode_lyrics(model, tokenizer, lyrics_prompt, device)
    lyric_stats = compute_stats(lyric_hidden_states)
    print(f"Lyric hidden states: {lyric_stats}")

    # Save reference data
    reference_data = {
        "text_prompt": text_prompt,
        "lyrics_prompt": lyrics_prompt,
        "text_token_ids": tokenizer.encode(text_prompt, add_special_tokens=True),
        "lyric_token_ids": tokenizer.encode(lyrics_prompt, add_special_tokens=True),
        "text_hidden_states_stats": text_stats,
        "lyric_hidden_states_stats": lyric_stats,
        "text_hidden_states_shape": list(text_hidden_states.shape),
        "lyric_hidden_states_shape": list(lyric_hidden_states.shape),
    }

    # Save as JSON (not the actual tensors - too large)
    with open(output_path, "w") as f:
        json.dump(reference_data, f, indent=2)

    print(f"\nReference data saved to {output_path}")

    # Also print token IDs for comparison
    print(f"\n=== Token IDs ===")
    print(
        f"Text tokens ({len(reference_data['text_token_ids'])}): {reference_data['text_token_ids'][:50]}..."
    )
    print(
        f"Lyric tokens ({len(reference_data['lyric_token_ids'])}): {reference_data['lyric_token_ids'][:50]}..."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export reference encoder_hidden_states"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Path to checkpoints directory",
    )
    parser.add_argument(
        "--instruction",
        default="Fill the audio semantic mask based on the given conditions:",
        help="Instruction text",
    )
    parser.add_argument(
        "--caption", default="A beautiful piano melody", help="Caption text"
    )
    parser.add_argument("--lyrics", default="[Instrumental]", help="Lyrics text")
    parser.add_argument("--language", default="en", help="Language")
    parser.add_argument(
        "--metas",
        default="language=en\ninstrumental=true\nbpm=120\nkey=C\ntimesig=4/4\nduration=30",
        help="Metadata string",
    )
    parser.add_argument(
        "--output",
        default="/Users/erich/git/github/erichchampion/cadenza-audio/.cursor/reference_encoder_hidden_states.json",
        help="Output path for reference data",
    )

    args = parser.parse_args()

    export_reference(
        checkpoint_dir=args.checkpoint_dir,
        instruction=args.instruction,
        caption=args.caption,
        lyrics=args.lyrics,
        language=args.language,
        metas=args.metas,
        output_path=args.output,
    )
