/**
 Pipeline that runs diffusion (via DiffusionStepper) then VAE decode (via VAEDecoder).
 Uses timestep schedule and optional seed; returns GenerationResult with audios.
 Enables end-to-end path with fake or real DiT/VAE.
 */

import Foundation
import MLX
import MLXRandom

/// Default latent time dimension when duration is invalid or not set.
private let defaultLatentLength = 100

/// Latent channel dimension (DiT/VAE).
private let latentChannels = 64

/// Returns latent time steps T from duration (seconds) and sample rate.
/// If duration <= 0, returns defaultLatentLength. Exposed for testing.
public func latentLengthFromDuration(durationSeconds: Double, sampleRate: Int) -> Int {
    guard durationSeconds > 0 else { return defaultLatentLength }
    let samples = durationSeconds * Double(sampleRate)
    let t = Int(ceil(samples / Double(vaeLatentToSamplesFactor)))
    return max(1, t)
}

/// Optional provider of DiT conditions from generation params (e.g. condition encoder).
public typealias ConditioningProvider = (GenerationParams) -> DiTConditions?

/// Pipeline that runs diffusion loop then VAE decode using injected stepper and decoder.
public final class ContractGenerationPipeline: GenerationPipeline {
    private let stepper: DiffusionStepper
    private let decoder: VAEDecoder
    private let sampleRate: Int
    private let conditioningProvider: ConditioningProvider?

    public var isInitialized: Bool { true }

    /// - Parameters:
    ///   - stepper: Per-step diffusion (e.g. FakeDiffusionStepper or MLXDiTStepper).
    ///   - decoder: VAE decode latent → audio (e.g. FakeVAEDecoder or MLXVAEDecoder).
    ///   - sampleRate: Output waveform sample rate (default 48000).
    ///   - conditioningProvider: When non-nil, used to build DiTConditions from params (e.g. caption/lyrics → encoder hidden states + context latents). When nil, empty conditions are used.
    public init(
        stepper: DiffusionStepper,
        decoder: VAEDecoder,
        sampleRate: Int = AceStepConstants.defaultSampleRate,
        conditioningProvider: ConditioningProvider? = nil
    ) {
        self.stepper = stepper
        self.decoder = decoder
        self.sampleRate = sampleRate
        self.conditioningProvider = conditioningProvider
    }

    public func run(params: GenerationParams, config: GenerationConfig, progress: ((Double, String) -> Void)?) async throws -> GenerationResult {
        let b = config.batchSize
        let t = latentLengthFromDuration(durationSeconds: params.duration, sampleRate: sampleRate)
        let schedule = DiffusionSchedule.getTimestepSchedule(
            shift: params.shift,
            timesteps: params.timesteps,
            inferSteps: params.inferenceSteps > 0 ? params.inferenceSteps : nil
        )
        progress?(0, "Starting diffusion")

        let key: MLXArray? = (params.seed >= 0) ? MLXRandom.key(UInt64(params.seed)) : nil
        let noise = MLXRandom.normal([b, t, latentChannels], key: key)
        var xt = noise
        let conditions = conditioningProvider?(params) ?? DiTConditions()

        for (stepIdx, timestepVal) in schedule.enumerated() {
            let nextT: Float? = (stepIdx + 1 < schedule.count) ? Float(schedule[stepIdx + 1]) : nil
            xt = stepper.step(currentLatent: xt, timestep: Float(timestepVal), conditions: conditions, nextTimestep: nextT)
            let frac = Double(stepIdx + 1) / Double(schedule.count)
            progress?(frac * 0.9, "Diffusion step \(stepIdx + 1)/\(schedule.count)")
        }

        progress?(0.95, "Decoding")
        let audio = decoder.decode(latent: xt)
        audio.eval()

        let audios = buildAudiosFromDecoded(audio)
        progress?(1.0, "Done")
        return GenerationResult(
            audios: audios,
            statusMessage: "Generated \(audios.count) clip(s)",
            extraOutputs: [:],
            success: true,
            error: nil
        )
    }

    /// Build audios list from decoded [B, samples] or [B, L, C] (e.g. stereo) array.
    private func buildAudiosFromDecoded(_ audio: MLXArray) -> [[String: Any]] {
        let shape = audio.shape
        guard shape.count >= 2 else { return [] }
        let b = shape[0]
        let samplesPerBatch = shape.count > 2 ? (shape[1] * shape[2]) : shape[1]
        var result: [[String: Any]] = []
        let floats = audio.asArray(Float.self)
        let total = b * samplesPerBatch
        guard floats.count >= total else { return [] }
        for i in 0..<b {
            let start = i * samplesPerBatch
            let end = start + samplesPerBatch
            let slice = Array(floats[start..<end])
            result.append([
                "tensor": slice,
                "sample_rate": sampleRate
            ] as [String: Any])
        }
        return result
    }
}
