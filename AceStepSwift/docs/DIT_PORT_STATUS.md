# DiT model port status

The **VAE decoder** is fully ported and pluggable via `MLXVAEDecoder` (load weights with `loadWeights(from:)`, use in `ContractGenerationPipeline`).

The **DiT** (diffusion transformer) from `acestep/models/mlx/dit_model.py` is **ported**. All components are implemented in Swift.

## Python reference

- **dit_model.py**: `MLXRotaryEmbedding`, `MLXCrossAttentionCache`, `MLXSwiGLUMLP`, `MLXAttention`, `MLXDiTLayer`, `MLXTimestepEmbedding`, `MLXDiTDecoder`
- **dit_convert.py**: Converts PyTorch decoder weights for `load_weights()` (Conv1d/ConvTranspose1d axis swap; skip rotary_emb)
- **dit_generate.py**: `mlx_generate_diffusion` uses the decoder in a loop; we have `apgForward`, timestep schedule, and ODE step in Swift

## Swift DiT components (all present)

- **DiTRotaryEmbedding** – cos/sin tables, `call(seqLen:)` → (cos, sin)
- **DiTTimestepEmbedding** – sinusoidal + MLP, `call(_:)` → (temb, timestepProj)
- **DiTSwiGLUMLP** – gate_proj, up_proj, down_proj, SiLU
- **DiTCrossAttentionCache** – KV cache for cross-attention
- **DiTRotaryHelpers** – `diTRotateHalf`, `diTApplyRotaryPosEmb`, `diTCreateSlidingWindowMask`
- **DiTAttention** – Q/K/V, QK-RMSNorm, RoPE, GQA, SDPA, o_proj; cache and sliding-window mask
- **DiTLayer** – AdaLN, self_attn, cross_attn, MLP
- **DiTDecoder** – proj_in, time_embed, time_embed_r, condition_embedder, layers, norm_out, proj_out; pad/patch, cache
- **MLXDiTStepper** – conforms to `DiffusionStepper`; runs decoder forward and applies ODE step (`xt - vt * dt` or `xt - vt * t`)

## Pipeline integration

- `DiffusionStepper.step(..., nextTimestep: Float?)` – when non-nil, ODE uses `dt = timestep - nextTimestep`; when nil (last step), uses `xt - vt * timestep`.
- `ContractGenerationPipeline` passes `nextTimestep` from the schedule so the stepper can compute the correct step.

### Task types (cover, repaint, lego, extract, complete)

When `params.taskType` is `cover`, `repaint`, `lego`, `extract`, or `complete` and `params.srcAudio != nil`, the conditioning provider should load and encode the source audio (VAE encode), use the task-specific instruction from `TaskInstructions.instruction(for: params)`, and set `DiTConditions.initialLatents` from the encoded source so the diffusion loop starts from it. For extract and complete (no repaint span), use the full encoded source as initial latents; no repaint-span masking is required.

## Weight loading

After porting, use Python `dit_convert.convert_and_load` (or equivalent) to produce a safetensors file, then in Swift:

```swift
let params = try loadParameters(from: ditURL)
ditDecoder.update(parameters: params)
```

Parameter names must match the Python decoder’s state dict (e.g. `proj_in.weight`, `layers.0.self_attn.q_proj.weight`).
