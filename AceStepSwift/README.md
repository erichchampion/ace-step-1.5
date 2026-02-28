# AceStepSwift

Swift package that ports core ACE-Step LLM and music-generation functionality for use with [MLX Swift](https://github.com/ml-explore/mlx-swift) on macOS (Apple Silicon). Intended for apps such as [Cadenza Audio](https://github.com/erichchampion/cadenza-audio) that want a native Swift backend instead of embedding the Python stack.

## Requirements

- macOS 14+ (arm64) or **iOS 17+**
- Xcode (Swift 5.9+)
- [MLX Swift](https://github.com/ml-explore/mlx-swift) (pulled in via SwiftPM)

When using this package in an iOS app, set the **app target’s minimum deployment target to iOS 17** (or higher) in Xcode (General → Minimum Deployments) so it matches the package’s requirement.

## What’s included

- **Types:** `GenerationParams`, `GenerationConfig`, `FormatSampleResult`, `GenerationResult`, and constants (BPM, duration, keyscale, time signature, languages) aligned with the Python `acestep/inference.py` and `acestep/constants.py`.
- **Diffusion schedule:** `DiffusionSchedule.getTimestepSchedule(shift:timesteps:inferSteps:)` matching the Python DiT timestep logic.
- **Format sample:** Parser for LLM output (`<think>`…`</think>`) and `AceStepEngine.formatSample(caption:lyrics:userMetadata:temperature:llmProvider:)` returning `FormatSampleResult`. Use `LLMFormatProvider`: **MockLLMFormatProvider** (tests) or **MLXLLMFormatProvider** (load from directory or Hugging Face id via mlx-swift-lm).
- **Generate music:** `AceStepEngine.generateMusic(params:config:progress:pipeline:)` returning `GenerationResult`. Use `GenerationPipeline`: **StubGenerationPipeline** (no audio), or **ContractGenerationPipeline** (inject `DiffusionStepper` + `VAEDecoder` to run diffusion loop + VAE decode and return audios).
- **FSM states and transitions:** `FSMState` enum and `nextState(after:options:)` with `FSMTransitionOptions` (canonical field order; optional FSM logits masking in generation is planned).

## Usage

```swift
import AceStepSwift

// Format caption/lyrics via LLM (mock for tests, or MLXLLMFormatProvider for real model)
let result = AceStepEngine.formatSample(
    caption: "Latin pop, reggaeton",
    lyrics: "[Verse 1]\n...",
    userMetadata: nil,
    temperature: 0.85,
    llmProvider: myLLMProvider
)
if result.success {
    print(result.caption, result.bpm, result.duration)
}

// With MLXLLMFormatProvider (load from local dir or Hugging Face id):
let llm = MLXLLMFormatProvider()
try await llm.load(from: URL(fileURLWithPath: "/path/to/acestep-5Hz-lm-0.6B"))  // or llm.load(id: "mlx-community/...")
let formatResult = AceStepEngine.formatSample(caption: "jazz", lyrics: "[Instrumental]", userMetadata: nil, temperature: 0.85, llmProvider: llm)
```

```swift
// Generate music (stub returns no audio; contract pipeline with fakes returns placeholder audios)
let pipeline = ContractGenerationPipeline(
    stepper: FakeDiffusionStepper(),
    decoder: FakeVAEDecoder(),
    sampleRate: AceStepConstants.defaultSampleRate
)
let genResult = AceStepEngine.generateMusic(
    params: GenerationParams(caption: "jazz", lyrics: "[Instrumental]"),
    config: GenerationConfig(batchSize: 1),
    progress: nil,
    pipeline: pipeline
)
// genResult.audios: [["tensor": [Float], "sample_rate": 48000], ...]
```

## Adding to your project

In Xcode: File → Add Package Dependencies → add the ace-step-1.5 repo and select the `AceStepSwift` package (or add a local path to `AceStepSwift`).

Or in `Package.swift`:

```swift
dependencies: [
    .package(path: "../ace-step-1.5/AceStepSwift"),  // or git URL
],
targets: [
    .target(name: "YourApp", dependencies: ["AceStepSwift"]),
]
```

## Tests

Run tests from the package directory:

```bash
swift test
```

Some tests (APG, diffusion contract, VAE decode contract) use MLX arrays and require Metal; they may be skipped or fail in environments where the default Metal library is not available (e.g. some CI). Pure-logic tests (constants, params, timestep schedule, parser, format/generate API with mocks) do not require Metal.

## Smoke test (Python + Swift)

From the **repository root** (parent of `AceStepSwift`), a script runs both the original Python pipeline and the Swift package to generate short clips, then validates that each output file contains a non-silent, variable waveform:

```bash
./scripts/run_generation_smoke_test.sh
```

- **Output directory:** `OUTPUT_DIR` (default: `./generation_smoke_output`). Python writes `python_out.wav`, Swift writes `swift_out.wav`.
- **Python:** Defaults to `acestep-v15-turbo`; looks for a valid checkpoint under `checkpoints/` (tries `acestep-v15-turbo` then `acestep-v15-base`). Set `ACESTEP_CONFIG_PATH` to override. If no checkpoint dir exists, the Python step is skipped.
- **Swift:** Uses the existing pipeline with minimal conditioning and a fake VAE so it runs without DiT/VAE weights; it still produces a synthetic waveform for validation. For real decoded audio, you would set `DIT_WEIGHTS_PATH` (and VAE weights) and use `MLXVAEDecoder` in the test (not implemented in the script by default).

Validation (variable waveform, no silence/NaN) is done by `scripts/validate_audio.py` (uses `soundfile` or `scipy.io.wavfile` if available).

## Status

- **Done:** Types, constants, timestep schedule, format-sample parser and API, generate-music API, stubs and mocks, **MLXLLMFormatProvider** (load from directory or Hugging Face id, ChatSession-based generation). FSM states and transition logic (`FSMTransitions`, `nextState(after:options:)`) with tests. Invalid-params test for `generateMusic`.
- **Phase 3 (contracts):** APG (`apgForward`), diffusion step contract (`DiffusionStepper`, `FakeDiffusionStepper`, `nextTimestep` for ODE step), VAE decode contract (`VAEDecoder`, `FakeVAEDecoder`, `vaeLatentToSamplesFactor`). **ContractGenerationPipeline** runs diffusion loop then VAE decode; optional `conditioningProvider` to supply `DiTConditions` from params.
- **Real VAE:** **MLXVAEDecoder** and **MLXAutoEncoderOobleck**: full encode/decode (Snake1d, residual units, encoder/decoder blocks). `encodeMean(audioNLC:)`, `encodeAndSample(audioNLC:)`, `decode(latentsNLC:)`. Load decoder weights with `loadWeights(from: url)` (safetensors from Python `vae_convert`).
- **Weight loading:** `loadParameters(from: url)` and `loadDiTParameters(from: url)` / `loadDiTParametersForDecoder(from: url)` load safetensors into nested `ModuleParameters` for `module.update(parameters:)`. They use **MLX**’s `loadArrays(url:stream:)` to read the file (returns `[String: MLXArray]`); the MLX dependency must be linked. See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for key normalization and Conv weight conversion.
- **DiT:** Full port; see **docs/DIT_PORT_STATUS.md**. **DiTDecoder**, **MLXDiTStepper** (ODE step with `nextTimestep`). Use `ContractGenerationPipeline(stepper: MLXDiTStepper(decoder: ditDecoder), decoder: ..., ...)` after loading DiT weights.
- **Planned (optional):** Condition encoder (caption/lyrics → encoder hidden states + context latents); FSM logits masking in LLM sampling; 5Hz LM checkpoints.
