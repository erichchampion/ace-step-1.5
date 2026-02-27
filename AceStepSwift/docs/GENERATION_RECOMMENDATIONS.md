# Generation Recommendations

Recommendations for getting **meaningful audio** from the AceStepSwift pipeline (DiT diffusion + VAE decode). The pipeline can produce sound files even without proper conditioning, but output will be unstructured or noise-like unless the following are in place.

---

## 1. Provide Real Conditioning

The diffusion model expects **conditioning** on every step. In the Python reference, this always comes from `model.prepare_condition(...)`:

- **Encoder hidden states** `[B, enc_L, 2048]` — from caption, lyrics, and reference audio (condition encoder).
- **Context latents** `[B, T, 128]` — from chunk masks, source latents, and task setup.

If the Swift pipeline has no conditioning provider, or the provider returns empty conditions, the stepper uses **zeros** for both. That yields unstructured or noise-like audio, not coherent music.

**Recommendation:** Implement and pass a **`ConditioningProvider`** that returns real `encoderHiddenStates` and `contextLatents` derived from your prompt/metadata and context (equivalent to Python’s `prepare_condition`).

---

## 2. Use the ConditioningProvider API Correctly

The pipeline calls the provider with:

```swift
conditioningProvider?(params, latentLength, sampleRate)
```

- **`params`** — `GenerationParams` (caption, lyrics, duration, etc.).
- **`latentLength`** — Number of latent time steps `T` used in this run (from `latentLengthFromDuration(duration, sampleRate)`).
- **`sampleRate`** — Pipeline sample rate (e.g. 48000).

**Recommendation:**

- Return **`encoderHiddenStates`** with shape **`[B, encL, 2048]`** (batch, encoder sequence length, hidden size). Build this from your condition encoder (text/lyrics/reference → 2048‑dim hidden states).
- Return **`contextLatents`** with shape **`[B, latentLength, 128]`** (batch, latent time steps, context channels). Use the **`latentLength`** and **`sampleRate`** arguments so this matches the pipeline’s `T`; do not assume a fixed length.
- Optionally set **`initialLatents`** for cover/repaint (start from encoded source instead of noise); if set, it must have shape `[B, latentLength, 64]`.

---

## 3. Match Python Conditioning Semantics

For behavior comparable to the Python service:

- **Encoder path:** Caption, lyrics, metadata, and reference audio should be encoded (e.g. via your condition encoder / text encoder) into a sequence of 2048‑dimensional vectors. That sequence is `encoderHiddenStates` `[B, encL, 2048]`.
- **Context path:** Build context latents from chunk masks, source latents, and task type (text2music, cover, repaint, etc.) so that `contextLatents` has shape `[B, T, 128]` with `T = latentLength` for the current run.

You can build conditions in Swift using the **prepare-condition API** (see below) so that context latents are computed correctly; encoder hidden states can be supplied by the app (e.g. from Python or a future Swift condition encoder).

---

## 3.1 Swift prepare-condition API (context latents + precomputed encoder)

The module **PrepareCondition** provides a Swift port of the context path of Python’s `prepare_condition` and a single API that returns `DiTConditions`:

- **`PrepareConditionInputs`** — Holds `srcLatents` [B, T, 64], `chunkMasks` [B, T] or [B, T, 64], optional `precomputedEncoderHiddenStates` / `precomputedEncoderAttentionMask`, optional **`textHiddenStates`** [B, L, 1024] and **`textAttentionMask`** [B, L] for the Swift condition encoder, optional `precomputedLmHints25Hz` and `isCovers` (cover path), and optional `nullConditionEmbedding` / `initialLatents`.
- **`buildContextLatents(srcLatents:chunkMasks:)`** — Expands 2D chunk masks to [B, T, 64] if needed, then concatenates with src_latents to produce **context latents** [B, T, 128]. Use this when you only need context latents.
- **`prepareCondition(inputs:)`** — Builds context latents and returns **`DiTConditions`**; uses `precomputedEncoderHiddenStates` when set, or (when using the overload with `conditionEncoder`) runs the Swift encoder when `textHiddenStates` is set.
- **`prepareCondition(inputs:conditionEncoder:)`** — When `conditionEncoder` is non-nil and `inputs.textHiddenStates` is non-nil, **encoder hidden states are produced in Swift** via the condition encoder (text-only path). Otherwise precomputed values are used. Use this for a true Swift condition encoder instead of app-supplied fallbacks.
- **`fullChunkMask(batchSize:latentLength:)`** — Returns a default “full” chunk mask [B, T] of ones for text2music (all positions valid).
- **`applyCoverHints(srcLatents:lmHints25Hz:isCovers:)`** — Optional: where `isCovers` > 0 use LM hints, else src_latents; use when you have precomputed LM hints and no tokenize/detokenize.

**Swift condition encoder (full):** The **`ConditionEncoder`** module (see **ConditionEncoder.swift**) implements the full encoder: **text projector**, **lyric encoder** (8 layers), **timbre encoder** (4 layers), and **pack_sequences**. Load weights from the same DiT checkpoint under **`encoder.text_projector.*`**, **`encoder.lyric_encoder.*`**, and **`encoder.timbre_encoder.*`**. Apply the encoder subtree to the condition encoder with `conditionEncoder.update(parameters: encoderParameters)`. Then pass `ConditionEncoder` and inputs into `prepareCondition(inputs:conditionEncoder:)`:

- **Text-only:** Set `inputs.textHiddenStates` (and optionally `textAttentionMask`). Lyric and timbre are filled with empty tensors internally.
- **Full caption + lyric + timbre:** Set `textHiddenStates`, `textAttentionMask`, `lyricHiddenStates`, `lyricAttentionMask`, and when you have reference audio set `referAudioPacked` [N, T, 64] and `referAudioOrderMask` [N] (values 0..<B-1).

**Recommendation:** Create a `ConditionEncoder`, load all encoder weights from the checkpoint, and call `prepareCondition(inputs:conditionEncoder:)` with `textHiddenStates` and optionally `lyricHiddenStates`/`lyricAttentionMask` and `referAudioPacked`/`referAudioOrderMask` for full conditioning in Swift.

---

## 4. DEBUG Warning When Conditioning Is Missing

In **DEBUG** builds, if the pipeline uses default (zero) conditions, it prints a message like:

```
[ContractGenerationPipeline] No conditioning provided; using zeros. Output will not be meaningful. Provide a ConditioningProvider that returns encoderHiddenStates [B, encL, 2048] and contextLatents [B, T, 128].
```

**Recommendation:** Treat this as a reminder that you must supply a provider and real conditioning for meaningful generation. In release builds the warning is not printed, but behavior is the same (zeros → non-meaningful output).

---

## 5. Optional: Same Sample Rate and Latent Length Formula

The pipeline uses:

- **Sample rate** — From `ContractGenerationPipeline(sampleRate:)` (e.g. `AceStepConstants.defaultSampleRate`).
- **Latent length** — `latentLengthFromDuration(durationSeconds:params.duration, sampleRate:sampleRate)` with `vaeLatentToSamplesFactor = 2048`.

Your conditioning provider receives the same `sampleRate` and the precomputed `latentLength` for the run. Use them when building `contextLatents` so lengths always match the diffusion loop.

---

## Summary Checklist

- [ ] Implement a **ConditioningProvider** that returns non-nil, non-empty conditions for normal generation.
- [ ] Return **encoderHiddenStates** `[B, encL, 2048]` from your condition encoder (caption/lyrics/reference), or supply them via **PrepareConditionInputs.precomputedEncoderHiddenStates** (e.g. from Python), or use the **Swift condition encoder**: load **ConditionEncoder** with `encoder.text_projector` weights and pass **textHiddenStates** + **prepareCondition(inputs:conditionEncoder:)** for text-only generation.
- [ ] Return **contextLatents** `[B, latentLength, 128]` using the provider’s **latentLength** (and sample rate as needed). Use **prepareCondition(inputs:)** or **buildContextLatents(srcLatents:chunkMasks:)** with **fullChunkMask(batchSize:latentLength:)** for text2music.
- [ ] Pass that provider into **ContractGenerationPipeline(stepper:decoder:sampleRate:conditioningProvider:)**.
- [ ] For cover/repaint, optionally set **initialLatents** `[B, latentLength, 64]` in the returned `DiTConditions` (or via **PrepareConditionInputs.initialLatents**).

For more detail on Swift vs Python logic (including conditioning and timestep semantics), see **SWIFT_VS_PYTHON_LOGIC.md**.
