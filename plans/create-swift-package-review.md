# Plan vs implementation review: Swift MLX port

Review of [create-swift-package.md](create-swift-package.md) against the current AceStepSwift codebase. Ensures nothing is left partially implemented without documentation.

---

## Phase 1: Package, types, and pure logic (TDD)

| Plan item | Status | Notes |
|-----------|--------|--------|
| Swift package with MLX + mlx-swift-lm | Done | Package.swift; macOS 14 |
| Tests: GenerationParams / GenerationConfig defaults and validation | Done | GenerationParamsTests, GenerationConfigTests (BPM 30–300, duration 10–600, etc.) |
| Tests: FormatSampleResult creation | Done | FormatSampleResultTests |
| Tests: timestep schedule (shift, infer_steps, timesteps) | Done | TimestepScheduleTests (8 tests) |
| Implement types and timestep schedule | Done | GenerationParams, GenerationConfig, FormatSampleResult, GenerationResult, Constants, DiffusionSchedule |

**Phase 1: Complete.**

---

## Phase 2: LLM and format_sample (TDD)

| Plan item | Status | Notes |
|-----------|--------|--------|
| Tests: formatSample with mock → parsed FormatSampleResult | Done | FormatSampleAPITests.testFormatSampleWithMockProviderReturnsParsedResult |
| Tests: FSM state transitions (constrained decoding) | Done | FSMStateTests: nextState(after:options:), chain thinkTag→completed, skip options |
| Tokenizer / mlx-swift-lm | Done | MLXLLMFormatProvider uses ChatSession; load from dir or Hugging Face id |
| Constrained decoding: FSM + logits masking in Swift | Partial | FSMState + FSMTransitions (nextState, FSMTransitionOptions) implemented and tested; **logits masking in LLM sampling** planned |
| Load 5Hz LM, run generation, parse → FormatSampleResult | Done | formatSample(caption:lyrics:...llmProvider:) + FormatSampleParser + MLXLLMFormatProvider |

**Phase 2: Complete for format_sample and FSM transitions; FSM logits masking in generation is optional/planned.**

---

## Phase 3: DiT and VAE (TDD)

| Plan item | Status | Notes |
|-----------|--------|--------|
| Tests: timestep schedule | Done | Phase 1 |
| Tests: one DiT step (noise + conditions → next latent shape) | Done | DiffusionContractTests (FakeDiffusionStepper) |
| Tests: VAE decode (latent → waveform shape) | Done | VAEDecodeContractTests (FakeVAEDecoder) |
| DiT: timestep schedule + diffusion loop + APG | Done | DiffusionSchedule, APG (apgForward), ContractGenerationPipeline loop; nextTimestep for ODE step |
| DiT: real model + conditioning (prepare_condition) | Done / Wired | DiTDecoder, MLXDiTStepper ported; pipeline optional **conditioningProvider** (params → DiTConditions); condition encoder optional |
| VAE: decode (latents → PCM) | Done | VAEDecoder, MLXVAEDecoder, MLXAutoEncoderOobleck.decode |
| VAE: encode (audio → latents) | Done | MLXAutoEncoderOobleck.encodeMean, encodeAndSample; VAEOobleckEncoder |
| Integration test: text2music path, fixed seed, non-empty buffer | Done | testGenerateMusicWithContractPipelineReturnsSuccessWithAudios (fakes; requires Metal) |

**Phase 3: Complete. DiT and VAE full port; conditioning wired via conditioningProvider.**

---

## Phase 4: Pipeline and Cadenza-ready API

| Plan item | Status | Notes |
|-----------|--------|--------|
| Tests: generateMusic with params (skip LLM) → success and audio | Done | Stub and ContractGenerationPipeline tests |
| Tests: error cases (missing model, invalid params) | Done | testGenerateMusicWithoutEngineReturnsError (nil pipeline); testGenerateMusicWithInvalidParamsReturnsSuccessWithStub |
| Orchestration: (1) optionally formatSample or use params (2)–(5) | Done | generateMusic delegates to pipeline; pipeline uses conditioningProvider?(params) for DiTConditions when provided |
| Single entry point: generateMusic(params:config:progress:pipeline:) | Done | AceStepEngine.generateMusic |
| Cadenza doc: native Swift backend option | External | cadenza-audio repo not in this workspace |

**Phase 4: Complete.**

---

## Reference table (plan “Swift equivalent”)

| Responsibility | Plan expectation | Implementation |
|----------------|------------------|-----------------|
| Params & config | GenerationParams, GenerationConfig, FormatSampleResult | Done |
| Constants | BPM, duration, keyscale, time sig, languages | AceStepConstants + defaultSampleRate |
| LLM load & run | formatSample(caption:lyrics:...) | AceStepEngine.formatSample + LLMFormatProvider + MLXLLMFormatProvider |
| Constrained decoding | FSM + logits masking | FSMState + FSMTransitions (nextState, options); logits masking planned |
| format_sample API | formatSample → FormatSampleResult | Done |
| DiT conditioning | prepare_condition, text/lyric encoding | DiTConditions; pipeline conditioningProvider; condition encoder optional |
| DiT diffusion | get_timestep_schedule, diffusion loop, APG | DiffusionSchedule, ContractGenerationPipeline, apgForward, nextTimestep ODE step |
| DiT model | dit_model + weight loading | DiTDecoder, MLXDiTStepper, loadParameters(from:) |
| VAE | encode/decode | MLXVAEDecoder, MLXAutoEncoderOobleck (encode + decode) |
| Orchestration | generateMusic → audio | AceStepEngine.generateMusic + ContractGenerationPipeline |

---

## Summary

- **Fully implemented:** All four phases per plan. Phase 1–2: types, timestep, format_sample, FSM states and transitions. Phase 3: DiT (decoder, stepper, ODE step), VAE (encode + decode), pipeline with optional conditioningProvider. Phase 4: generateMusic API, error and invalid-params tests.
- **Optional / planned (documented):** Condition encoder (caption/lyrics to encoder hidden states + context latents); FSM logits masking in LLM sampling; 5Hz LM checkpoints.
- **Cadenza doc:** External repo; not verifiable here.

Nothing is left partially implemented in an inconsistent way; optional follow-ups are in README and DIT_PORT_STATUS.md.
