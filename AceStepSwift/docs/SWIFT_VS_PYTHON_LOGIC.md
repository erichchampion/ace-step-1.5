# Swift vs Python Logic Comparison

Detailed comparison of the AceStepSwift package against the original Python scripts (`acestep/models/mlx/`, `acestep/core/generation/handler/`) to surface possible bugs and layout/semantic differences.

---

## 1. DiT Decoder (`DiTDecoder` vs `MLXDiTDecoder`)

### 1.1 Timestep embeddings

- **Python** (`dit_model.py`): `temb_t, proj_t = self.time_embed(timestep)`; `temb_r, proj_r = self.time_embed_r(timestep - timestep_r)`. In the generation loop (`dit_generate.py`) both `timestep` and `timestep_r` are set to the **current** timestep, so `timestep - timestep_r = 0` and the second embedding always sees **0**.
- **Swift** (`DiTDecoder.swift`): Same formula `timeEmbedR.call(timestep - timestepR)`. Correct.

**Bug (see §5):** The **caller** `MLXDiTStepper` passes `timestepR = 0` instead of the current timestep, so Swift effectively passes `timestep - 0 = timestep` into the second embedding, changing semantics vs Python.

### 1.2 Input concat, padding, patch projection

- **Python:** `hidden_states = mx.concatenate([context_latents, hidden_states], axis=-1)`; pad to multiple of `patch_size`; `proj_in`; `condition_embedder(encoder_hidden_states)`.
- **Swift:** Same: `h = concatenated([contextLatents, hiddenStates], axis: 2)`; `padLength = (patchSize - (h.dim(1) % patchSize)) % patchSize`; same concat + `projIn`; `conditionEmbedder(encoderHiddenStates)`. **Matches.**

### 1.3 Layer types and sliding mask

- **Python:** `layer_types = ["sliding_attention" if (i+1) % 2 == 1 else "full_attention" for i in range(num_hidden_layers)]`; sliding mask only when `any(lt == "sliding_attention" for lt in self._layer_types)`.
- **Swift:** `($0 + 1) % 2 == 1 ? "sliding_attention" : "full_attention"`; `layerTypes.contains("sliding_attention")` for creating mask. **Matches.**

### 1.4 Output adaptive layer norm (scale_shift_table)

- **Python:** `scale_shift_table` is `[1, 2, inner_dim]`; `shift, scale = mx.split(self.scale_shift_table + mx.expand_dims(temb, axis=1), 2, axis=1)` → `[B, 2, D]` then split to shift/scale `[B, 1, D]`; `hidden_states = self.norm_out(hidden_states) * (1.0 + scale) + shift`.
- **Swift:** Handles both `[1, 2, D]` and checkpoint layout `[1, D, 2]` via `scaleShiftTable.dim(2) == 2` transpose; then `combined = table + tembExp` → `[B, 2, D]`; flatten and slice to get shift/scale `[B, 1, D]`; same formula. **Matches** (and supports alternative checkpoint layout).

### 1.5 Crop to original sequence length

- **Python:** `hidden_states = hidden_states[:, :original_seq_len, :]`.
- **Swift:** `h = h[0..<h.dim(0), 0..<originalSeqLen, 0..<h.dim(2)]`. **Matches.**

---

## 2. DiT Layer (`DiTLayer` vs `MLXDiTLayer`)

- **AdaLN:** Python uses `scale_shift_table` `[1, 6, D]` + `temb` `[B, 6, D]` → split into 6 parts (shift/scale/gate for self-attn and cross-attn/MLP). Swift handles `[1, 6, D]` vs `[1, D, 6]` with `scaleShiftTable.dim(1) == 6` and transpose; then same indexing for the 6 modulation terms. **Matches.**
- **Order:** Self-attn (with gate) → cross-attn → MLP (with c_shift, c_scale, c_gate). **Matches.**

---

## 3. Attention (`DiTAttention` vs `MLXAttention`)

- **Q/K/V projection, QK-RMSNorm, GQA repeat:** Same structure; Swift `repeatKv` matches Python `_repeat_kv`. **Matches.**
- **RoPE:** Applied only for self-attention; Python `_apply_rotary_pos_emb(q, k, cos, sin)`; Swift `diTApplyRotaryPosEmb`. **Matches.**
- **Cross-attention KV cache:** Python `MLXCrossAttentionCache` update/get by `layer_idx`; Swift `DiTCrossAttentionCache` same. **Matches.**
- **Scaled dot-product attention:** Same scale `head_dim ** -0.5` / `pow(Float(headDim), -0.5)`. **Matches.**

---

## 4. Rotary and sliding-window mask

- **RoPE:** Python `inv_freq = 1.0 / (base ** (arange(0, head_dim, 2) / head_dim))`; `freqs = positions[:, None] * inv_freq[None, :]` then `concatenate([freqs, freqs], axis=-1)`. Swift same formula and `concatenated([freqsBroadcast, freqsBroadcast], axis: 1)`. **Matches.**
- **Sliding mask:** Python `diff = abs(i - j)`; mask `0` where `diff <= window_size` else `-1e9`; shape `[1, 1, L, L]`. Swift `diTCreateSlidingWindowMask` same logic and shape. **Matches.**

---

## 5. Timestep embedding (`DiTTimestepEmbedding` vs `MLXTimestepEmbedding`)

- **Sinusoidal:** Python `t * scale`; `freqs = exp(-log(max_period) * arange(half) / half)`; `concat(cos(args), sin(args))`; odd `dim` gets extra zero. Swift `sinusoidalEmbedding` same. **Matches.**
- **MLP:** linear1 → SiLU → linear2 → SiLU → time_proj → reshape to `[B, 6, D]`. **Matches.**

---

## 6. Diffusion schedule (`DiffusionSchedule` vs `get_timestep_schedule`)

- **Custom timesteps:** Strip trailing zeros; cap at 20; map each to nearest in `VALID_TIMESTEPS`. **Matches.**
- **infer_steps:** `raw = [1.0 - i / infer_steps for i in range(infer_steps)]`; if `shift != 1`: `shift * t / (1 + (shift - 1) * t)`. Swift same. **Matches.**
- **Default:** Nearest valid shift; return `SHIFT_TIMESTEPS[shift]`. **Matches.**

---

## 7. ODE step (Python loop vs `MLXDiTStepper`)

- **Python:** `dt = current_t - next_t`; `xt = xt - vt * dt_arr`; last step: `xt = xt - vt * t_unsq`.
- **Swift:** `dt = timestep - nextT`; `currentLatent - vt * dt`; when `nextTimestep == nil`: `currentLatent - vt * timestep`. **Matches.**

---

## 8. **BUG: `timestep_r` in `MLXDiTStepper`**

- **Python** (`dit_generate.py`): `t_curr = mx.full(..., current_t)`; decoder called with `timestep=t_curr, timestep_r=t_curr`. So `timestep - timestep_r = 0` → second time embedding always gets **0**.
- **Swift** (`MLXDiTStepper.swift`): `timestepRArr = MLXArray([Float](repeating: 0, count: b))`. So `timestep - timestepR = timestep` → second time embedding gets the **current timestep**.

**Effect:** In Swift the second timestep embedding is conditioned on the full timestep instead of a fixed reference (0). This changes the additive conditioning and can alter generation quality or behavior compared to Python.

**Fix:** Pass the current timestep as `timestep_r` in the stepper (same as Python), e.g. `timestepRArr = MLXArray([Float](repeating: timestep, count: b))`, so that `timestep - timestep_r = 0` in the decoder.

---

## 9. VAE residual unit (`VAEResidualUnit` vs `MLXOobleckResidualUnit`)

- **Padding:** `((7 - 1) * dilation) // 2`. **Matches.**
- **Path:** conv1(snake1(x)); conv2(snake2(output)); trim input by `(hidden_len - output_len) // 2` on each side; residual add. **Matches.**

---

## 10. Snake1d (`VAESnake1d` vs `MLXSnake1d`)

- **Formula:** `x + (1 / (beta + 1e-9)) * sin(alpha * x)^2`; `alpha = exp(alpha)` when logscale. **Matches.** (Python can use float32 for stability with float16 weights; Swift does not currently special-case dtype.)

---

## 11. APG (`apgForward` vs `_mlx_apg_forward`)

- **Diff + momentum:** `diff = diff + running`; `running = diff`. **Matches.**
- **Norm cap:** `scale_factor = min(1, norm_threshold / (diff_norm + 1e-8))`; apply to diff. **Matches.**
- **Projection:** v1 = cond / ||cond||; parallel = (diff·v1) v1; orthogonal = diff - parallel; return `pred_cond + (guidance_scale - 1) * orthogonal`. **Matches.**

---

## 12. SwiGLU MLP

- **Python:** `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- **Swift:** `downProj(silu(gateProj(x)) * upProj(x))`. **Matches.**

---

## Summary of issues

| Item | Severity | Description |
|------|----------|-------------|
| **MLXDiTStepper `timestep_r`** | **Bug** | Uses `timestep_r = 0` instead of current timestep; second time embedding sees `timestep` instead of `0`, diverging from Python. |
| DiTDecoder/DiTLayer scale_shift transpose | OK | Handles both `[1,2,D]` and `[1,D,2]` (and 6-dim) for checkpoints; logic matches Python. |
| Snake1d dtype | Minor | Python can upcast to float32 for exp/sin with float16 weights; Swift does not. May matter only for float16. |

Recommendation: fix `timestep_r` in `MLXDiTStepper` so decoder semantics match Python; optionally add tests that compare decoder output for the same (timestep, timestep_r) inputs against Python/MLX.
