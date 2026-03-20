#!/usr/bin/env python3
"""Export reference tensors at each conditioning pipeline stage for Swift comparison tests."""

import argparse, struct, sys, os
from pathlib import Path
import numpy as np, torch

def _root():
    return Path(__file__).resolve().parent.parent

def write_bin(path, tensor):
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim == 1: arr = arr.reshape(1, 1, len(arr))
    elif arr.ndim == 2: arr = arr.reshape(1, *arr.shape)
    assert arr.ndim == 3, f"Expected 3D, got {arr.shape}"
    d0, d1, d2 = arr.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<iii", d0, d1, d2))
        f.write(arr.astype("<f4").tobytes())
    print(f"  {path.name}: shape=({d0},{d1},{d2}) mean={arr.mean():.6f} std={arr.std():.6f}")

def write_int_bin(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(ids)))
        for v in ids: f.write(struct.pack("<i", v))
    print(f"  {path.name}: count={len(ids)} first5={ids[:5]}")

# Build the special end-of-text token dynamically to avoid tool truncation
_EOT = "<" + "|endoftext|" + ">"

def build_text_prompt(caption, duration):
    """Match SFT_GEN_PROMPT.format(instruction, caption, metas) from acestep/constants.py"""
    instruction = "Fill the audio semantic mask based on the given conditions:"
    metas = (
        "- bpm: N/A\n"
        "- timesignature: N/A\n"
        "- keyscale: N/A\n"
        "- duration: " + str(int(duration)) + " seconds\n"
    )
    # SFT_GEN_PROMPT = "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<EOT>\n"
    return "# Instruction\n" + instruction + "\n\n# Caption\n" + caption + "\n\n# Metas\n" + metas + _EOT + "\n"

def build_lyric_prompt():
    return "# Languages\nunknown\n\n# Lyric\n[Instrumental]" + _EOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--caption", type=str, default="Smoke test")
    args = parser.parse_args()
    root = _root()
    out = args.output_dir.resolve()
    sys.path.insert(0, str(root))
    from transformers import AutoModel, AutoTokenizer

    print("=" * 60)
    print("Stage 1: Tokenization")
    print("=" * 60)
    tok = AutoTokenizer.from_pretrained(str(root / "checkpoints" / "Qwen3-Embedding-0.6B"), trust_remote_code=True)
    text_prompt = build_text_prompt(args.caption, args.duration)
    lyric_prompt = build_lyric_prompt()
    print(f"  text_prompt repr: {repr(text_prompt[:80])}...")
    print(f"  lyric_prompt repr: {repr(lyric_prompt[:80])}...")
    text_ids = tok.encode(text_prompt, add_special_tokens=True)
    lyric_ids = tok.encode(lyric_prompt, add_special_tokens=True)
    write_int_bin(out / "text_token_ids.bin", text_ids)
    write_int_bin(out / "lyric_token_ids.bin", lyric_ids)

    print("\n" + "=" * 60)
    print("Stage 2: Text Hidden States (Qwen full forward)")
    print("=" * 60)
    text_encoder = AutoModel.from_pretrained(
        str(root / "checkpoints" / "Qwen3-Embedding-0.6B"),
        trust_remote_code=True, dtype=torch.float32
    )
    text_encoder.eval()
    input_ids_t = torch.tensor([text_ids], dtype=torch.long)
    attn_mask_t = torch.ones_like(input_ids_t)
    with torch.no_grad():
        text_hs = text_encoder(input_ids=input_ids_t, attention_mask=attn_mask_t, return_dict=False)[0]
    write_bin(out / "text_hidden_states.bin", text_hs)

    print("\n" + "=" * 60)
    print("Stage 3: Lyric Token Embeddings (embed_tokens lookup)")
    print("=" * 60)
    lyric_input_ids = torch.tensor([lyric_ids], dtype=torch.long)
    lyric_attn = torch.ones(1, len(lyric_ids))
    with torch.no_grad():
        embed_layer = getattr(text_encoder, 'embed_tokens', None) or getattr(text_encoder.model, 'embed_tokens', None)
        lyric_embs = embed_layer(lyric_input_ids)
    write_bin(out / "lyric_token_embeddings.bin", lyric_embs)

    print("\n" + "=" * 60)
    print("Stage 4-7: ConditionEncoder stages")
    print("=" * 60)
    from acestep.handler import AceStepHandler
    handler = AceStepHandler()
    msg, ok = handler.initialize_service(project_root=str(root), config_path="acestep-v15-turbo", device="cpu")
    assert ok, msg
    model = handler.model
    cond_enc = model.encoder

    with torch.no_grad():
        text_proj = cond_enc.text_projector(text_hs)
    write_bin(out / "text_projected.bin", text_proj)

    with torch.no_grad():
        lyric_enc_out = cond_enc.lyric_encoder(inputs_embeds=lyric_embs, attention_mask=lyric_attn)
        lyric_enc = lyric_enc_out.last_hidden_state
    write_bin(out / "lyric_encoded.bin", lyric_enc)

    silence = handler.silence_latent[:1, :750, :].cpu().float()
    refer_order = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        timbre_enc, timbre_mask = cond_enc.timbre_encoder(silence, refer_order)
    write_bin(out / "timbre_encoded.bin", timbre_enc)

    with torch.no_grad():
        enc_hs, enc_mask = cond_enc(
            text_hidden_states=text_hs,
            text_attention_mask=attn_mask_t.float(),
            lyric_hidden_states=lyric_embs,
            lyric_attention_mask=lyric_attn,
            refer_audio_acoustic_hidden_states_packed=silence,
            refer_audio_order_mask=refer_order,
        )
    write_bin(out / "encoder_hidden_states.bin", enc_hs)

    print("\n" + "=" * 60)
    print("Stage 8: Context Latents")
    print("=" * 60)
    latent_len = int(args.duration * 48000 / 1920)
    sil_src = handler.silence_latent[:1, :latent_len, :].cpu().float()
    chunk_masks = torch.ones(1, latent_len, 64)
    ctx = torch.cat([sil_src, chunk_masks], dim=-1)
    write_bin(out / "context_latents.bin", ctx)
    write_bin(out / "silence_latent_crop.bin", sil_src)

    print("\n" + "=" * 60)
    print("Stage 9: Noise + Null Condition")
    print("=" * 60)
    import mlx.core as mx
    noise_mlx = mx.random.normal(shape=[1, latent_len, 64], key=mx.random.key(42))
    noise_np = np.array(noise_mlx, dtype=np.float32)
    write_bin(out / "initial_noise.bin", noise_np)
    null_cond = model.null_condition_emb.detach().cpu().float()
    write_bin(out / "null_condition_embedding.bin", null_cond)

    print("\n" + "=" * 60)
    print("Stage 10: DiT Velocity at t=1.0")
    print("=" * 60)
    noise_pt = torch.from_numpy(noise_np)
    attn_mask_hs = torch.ones(1, latent_len)
    enc_attn = torch.ones(1, enc_hs.shape[1])
    pos_ids = torch.arange(latent_len // 2).unsqueeze(0)
    cache_pos = torch.arange(latent_len // 2)
    with torch.no_grad():
        vel = model.decoder(
            hidden_states=noise_pt,
            timestep=torch.tensor([1.0]),
            timestep_r=torch.tensor([1.0]),
            attention_mask=attn_mask_hs,
            encoder_hidden_states=enc_hs,
            encoder_attention_mask=enc_attn,
            context_latents=ctx,
            position_ids=pos_ids,
            cache_position=cache_pos,
        )
        if isinstance(vel, tuple): vel = vel[0]
    write_bin(out / "velocity_t1.bin", vel)

    print("\nDone! All reference tensors saved to:", out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
