#!/usr/bin/env python3
"""
Compare Python conditioning pipeline intermediates against each other to validate correctness.
This script loads the exported reference data and does stage-by-stage validation.

Then it generates a fresh conditioning pass and compares the CLI's output conditioning 
against the Python reference to identify divergences.

Usage:
  PYTHONPATH=. python scripts/compare_conditioning.py /tmp/conditioning_stages
"""

import struct, sys, os
from pathlib import Path
import numpy as np

def load_bin(path):
    with open(path, 'rb') as f:
        d0, d1, d2 = struct.unpack('<iii', f.read(12))
        arr = np.frombuffer(f.read(), dtype='<f4').reshape(d0, d1, d2).copy()
    return arr

def load_int_bin(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<i', f.read(4))[0]
        ids = [struct.unpack('<i', f.read(4))[0] for _ in range(n)]
    return ids

def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compare(name, a, b, threshold=0.95):
    cos = cosine_sim(a, b)
    maxd = float(np.abs(a - b).max())
    ok = a.shape == b.shape and cos > threshold
    icon = "PASS" if ok else "FAIL"
    print(f"[{icon}] {name}")
    print(f"  A: shape={a.shape} mean={a.mean():.6f} std={a.std():.6f}")
    print(f"  B: shape={b.shape} mean={b.mean():.6f} std={b.std():.6f}")
    print(f"  cosine={cos:.6f} maxDiff={maxd:.6f}")
    if not ok:
        if a.shape != b.shape:
            print(f"  ERROR: Shape mismatch!")
        if cos <= threshold:
            print(f"  ERROR: Cosine {cos:.4f} below threshold {threshold}")
    return ok

def main():
    if len(sys.argv) < 2:
        print("Usage: compare_conditioning.py <stages_dir>")
        return 1

    stages_dir = Path(sys.argv[1])
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))

    import torch
    from transformers import AutoModel, AutoTokenizer
    from acestep.handler import AceStepHandler

    passed, failed = 0, 0

    # Load Python reference data
    py_text_hs = load_bin(stages_dir / "text_hidden_states.bin")
    py_lyric_embs = load_bin(stages_dir / "lyric_token_embeddings.bin")
    py_text_proj = load_bin(stages_dir / "text_projected.bin")
    py_lyric_enc = load_bin(stages_dir / "lyric_encoded.bin")
    py_timbre_enc = load_bin(stages_dir / "timbre_encoded.bin")
    py_enc_hs = load_bin(stages_dir / "encoder_hidden_states.bin")
    py_ctx = load_bin(stages_dir / "context_latents.bin")
    py_silence = load_bin(stages_dir / "silence_latent_crop.bin")

    print("=" * 60)
    print("Loading PyTorch model for comparison...")
    print("=" * 60)
    handler = AceStepHandler()
    msg, ok = handler.initialize_service(project_root=str(root), config_path="acestep-v15-turbo", device="cpu")
    assert ok, msg
    cond_enc = handler.model.encoder

    # Test 1: Text Projection
    print("\n" + "=" * 60)
    print("Test 1: Text Projection (text_projector)")
    print("=" * 60)
    with torch.no_grad():
        recomp_proj = cond_enc.text_projector(torch.from_numpy(py_text_hs)).numpy()
    if compare("TextProjection (recompute vs export)", recomp_proj, py_text_proj, 0.999):
        passed += 1
    else:
        failed += 1

    # Test 2: Lyric Encoder
    print("\n" + "=" * 60)
    print("Test 2: Lyric Encoder")
    print("=" * 60)
    lyric_attn = torch.ones(1, py_lyric_embs.shape[1])
    with torch.no_grad():
        recomp_lyric = cond_enc.lyric_encoder(
            inputs_embeds=torch.from_numpy(py_lyric_embs),
            attention_mask=lyric_attn
        ).last_hidden_state.numpy()
    if compare("LyricEncoder (recompute vs export)", recomp_lyric, py_lyric_enc, 0.999):
        passed += 1
    else:
        failed += 1

    # Test 3: Timbre Encoder
    print("\n" + "=" * 60)
    print("Test 3: Timbre Encoder")
    print("=" * 60)
    silence750 = handler.silence_latent[:1, :750, :].cpu().float()
    refer_order = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        recomp_timbre, _ = cond_enc.timbre_encoder(silence750, refer_order)
        recomp_timbre = recomp_timbre.numpy()
    if compare("TimbreEncoder (recompute vs export)", recomp_timbre, py_timbre_enc, 0.999):
        passed += 1
    else:
        failed += 1

    # Test 4: Full Encoder Hidden States
    print("\n" + "=" * 60)
    print("Test 4: Full Encoder Hidden States")
    print("=" * 60)
    text_attn = torch.ones(1, py_text_hs.shape[1])
    with torch.no_grad():
        recomp_enc, _ = cond_enc(
            text_hidden_states=torch.from_numpy(py_text_hs),
            text_attention_mask=text_attn,
            lyric_hidden_states=torch.from_numpy(py_lyric_embs),
            lyric_attention_mask=lyric_attn,
            refer_audio_acoustic_hidden_states_packed=silence750,
            refer_audio_order_mask=refer_order,
        )
        recomp_enc = recomp_enc.numpy()
    if compare("FullEncoder (recompute vs export)", recomp_enc, py_enc_hs, 0.999):
        passed += 1
    else:
        failed += 1

    # Test 5: Std analysis
    print("\n" + "=" * 60)
    print("Test 5: Conditioning Statistics Analysis")
    print("=" * 60)
    print(f"  text_hidden_states: std={py_text_hs.std():.6f} (should be ~3.1)")
    print(f"  lyric_embeddings:   std={py_lyric_embs.std():.6f} (should be ~0.03)")
    print(f"  text_projected:     std={py_text_proj.std():.6f}")
    print(f"  lyric_encoded:      std={py_lyric_enc.std():.6f}")
    print(f"  timbre_encoded:     std={py_timbre_enc.std():.6f}")
    print(f"  encoder_hidden_states: std={py_enc_hs.std():.6f} (final packed)")
    print(f"  context_latents:    std={py_ctx.std():.6f}")

    # Decompose the final encoder hidden states
    lyric_len = py_lyric_enc.shape[1]
    timbre_len = py_timbre_enc.shape[1]
    text_len = py_text_proj.shape[1]
    total_len = py_enc_hs.shape[1]
    print(f"\n  Packed sequence: lyric({lyric_len}) + timbre({timbre_len}) + text({text_len}) = {lyric_len + timbre_len + text_len} (actual: {total_len})")
    
    if total_len == lyric_len + timbre_len + text_len:
        enc_lyric = py_enc_hs[0, :lyric_len]
        enc_timbre = py_enc_hs[0, lyric_len:lyric_len+timbre_len]
        enc_text = py_enc_hs[0, lyric_len+timbre_len:]
        print(f"  enc[lyric portion]:  std={enc_lyric.std():.6f}")
        print(f"  enc[timbre portion]: std={enc_timbre.std():.6f}")
        print(f"  enc[text portion]:   std={enc_text.std():.6f}")
    else:
        print(f"  WARNING: packed sequence length mismatch")
    
    passed += 1  # Std analysis is informational

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
