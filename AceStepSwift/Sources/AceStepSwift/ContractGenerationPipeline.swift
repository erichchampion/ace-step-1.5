/**
 Pipeline that runs diffusion (via DiffusionStepper) then VAE decode (via VAEDecoder).
 Uses timestep schedule and optional seed; returns GenerationResult with audios.
 Enables end-to-end path with fake or real DiT/VAE.
 */

import Foundation
import MLX
import MLXRandom

private enum ContractGenerationPipelineError: LocalizedError {
    case missingConditioning
    case conditionBatchMismatch(name: String, expected: Int, actual: Int)

    var errorDescription: String? {
        switch self {
        case .missingConditioning:
            return "Missing DiT conditioning (encoderHiddenStates/contextLatents). Real generation requires a ConditioningProvider matching Python prepare_condition."
        case .conditionBatchMismatch(let name, let expected, let actual):
            return "Condition batch mismatch for \(name): expected batch \(expected), got \(actual). Provide batch-aligned conditioning or batch=1."
        }
    }
}

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

/// Optional provider of DiT conditions. Receives params, latent length T (from duration/sample rate), and sample rate
/// so it can build context_latents with shape [B, T, 128] matching the diffusion loop. Matches Python prepare_condition.
public typealias ConditioningProvider = (GenerationParams, _ latentLength: Int, _ sampleRate: Int) -> DiTConditions?

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
    ///   - conditioningProvider: When non-nil, called with (params, latentLength, sampleRate) to build DiTConditions. Return encoderHiddenStates [B, encL, 2048] and contextLatents [B, latentLength, 128] for meaningful output. When nil, zeros are used and output is not meaningful (matches Python only when real conditioning from prepare_condition is passed).
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

        var conditions = conditioningProvider?(params, t, sampleRate) ?? DiTConditions()
        conditions = try alignConditionsToBatch(conditions, batchSize: b)
        let usingDefaultConditions = (conditions.encoderHiddenStates == nil && conditions.contextLatents == nil)
        if usingDefaultConditions && stepper is MLXDiTStepper {
            throw ContractGenerationPipelineError.missingConditioning
        }
        if usingDefaultConditions {
            // Python always passes real conditioning from prepare_condition (text/lyric/refer → encoder_hidden_states, context_latents).
            // Zeros produce unstructured/noise-like audio. App should provide a ConditioningProvider that returns real encoder + context.
            #if DEBUG
            debugPrint("[ContractGenerationPipeline] No conditioning provided; using zeros. Output will not be meaningful. Provide a ConditioningProvider that returns encoderHiddenStates [B, encL, 2048] and contextLatents [B, \(t), 128].")
            #endif
        }
        let key: MLXArray? = (params.seed >= 0) ? MLXRandom.key(UInt64(params.seed)) : nil
        let noise = MLXRandom.normal([b, t, latentChannels], key: key)
        var xt: MLXArray
        if let initial = conditions.initialLatents, initial.shape[0] == b, initial.shape[1] == t {
            xt = initial
        } else {
            xt = noise
        }

        for (stepIdx, timestepVal) in schedule.enumerated() {
            let nextT: Float? = (stepIdx + 1 < schedule.count) ? Float(schedule[stepIdx + 1]) : nil
            xt = runDiffusionStep(
                currentLatent: xt,
                timestep: Float(timestepVal),
                nextTimestep: nextT,
                conditions: conditions,
                params: params
            )
            let frac = Double(stepIdx + 1) / Double(schedule.count)
            progress?(frac * 0.9, "Diffusion step \(stepIdx + 1)/\(schedule.count)")
        }

        progress?(0.95, "Decoding")
        var decodeLatent = xt
        if params.latentShift != 0.0 || params.latentRescale != 1.0 {
            decodeLatent = decodeLatent * Float(params.latentRescale) + Float(params.latentShift)
        }
        var audio = decoder.decode(latent: decodeLatent)
        audio = normalizePeakIfNeeded(audio)
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

    private func alignConditionsToBatch(_ conditions: DiTConditions, batchSize: Int) throws -> DiTConditions {
        func align(_ name: String, _ x: MLXArray?) throws -> MLXArray? {
            guard let x else { return nil }
            guard x.ndim > 0 else { return x }
            let actual = x.dim(0)
            if actual == batchSize { return x }
            if actual == 1 && batchSize > 1 {
                var target = x.shape
                target[0] = batchSize
                return MLX.broadcast(x, to: target)
            }
            throw ContractGenerationPipelineError.conditionBatchMismatch(name: name, expected: batchSize, actual: actual)
        }

        return DiTConditions(
            encoderHiddenStates: try align("encoderHiddenStates", conditions.encoderHiddenStates),
            contextLatents: try align("contextLatents", conditions.contextLatents),
            nullConditionEmbedding: conditions.nullConditionEmbedding,
            initialLatents: try align("initialLatents", conditions.initialLatents)
        )
    }

    private func runDiffusionStep(
        currentLatent: MLXArray,
        timestep: Float,
        nextTimestep: Float?,
        conditions: DiTConditions,
        params: GenerationParams
    ) -> MLXArray {
        guard
            let mlxStepper = stepper as? MLXDiTStepper,
            params.guidanceScale > 1.0,
            let nullCond = conditions.nullConditionEmbedding,
            let enc = conditions.encoderHiddenStates,
            let ctx = conditions.contextLatents
        else {
            return stepper.step(
                currentLatent: currentLatent,
                timestep: timestep,
                conditions: conditions,
                nextTimestep: nextTimestep
            )
        }

        var momentumState: [String: MLXArray]? = [:]
        let b = currentLatent.dim(0)
        let t = currentLatent.dim(1)
        let c = currentLatent.dim(2)

        let xIn = concatenated([currentLatent, currentLatent], axis: 0)
        let nullExpanded = MLX.broadcast(nullCond, to: enc.shape)
        let encDoubled = concatenated([enc, nullExpanded], axis: 0)
        let ctxDoubled = concatenated([ctx, ctx], axis: 0)
        let cfgConditions = DiTConditions(
            encoderHiddenStates: encDoubled,
            contextLatents: ctxDoubled,
            nullConditionEmbedding: nil,
            initialLatents: nil
        )

        let vtDoubled = mlxStepper.predictVelocity(
            currentLatent: xIn,
            timestep: timestep,
            conditions: cfgConditions,
            useCache: false
        )

        let predCond = vtDoubled[0..<b, 0..<t, 0..<c]
        let predUncond = vtDoubled[b..<(b * 2), 0..<t, 0..<c]
        let applyCFG = params.cfgIntervalStart <= Double(timestep) && Double(timestep) <= params.cfgIntervalEnd
        let vt: MLXArray
        if applyCFG {
            vt = apgForward(
                predCond: predCond,
                predUncond: predUncond,
                guidanceScale: Float(params.guidanceScale),
                momentumState: &momentumState
            )
        } else {
            vt = predCond
        }

        if let nextTimestep {
            let dt = timestep - nextTimestep
            return currentLatent - vt * dt
        } else {
            return currentLatent - vt * timestep
        }
    }

    private func normalizePeakIfNeeded(_ audio: MLXArray) -> MLXArray {
        guard audio.ndim == 3 else { return audio }
        let peak = MLX.abs(audio).max(axes: [1, 2], keepDims: true)
        let safePeak = MLX.where(peak .> 1.0, peak, MLXArray(1.0 as Float))
        return audio / safePeak
    }

    /// Build audios list from decoded [B, samples] or [B, L, C] (e.g. stereo) array.
    private func buildAudiosFromDecoded(_ audio: MLXArray) -> [[String: Any]] {
        let shape = audio.shape
        guard shape.count >= 2 else { return [] }
        let b = shape[0]
        let channels = shape.count > 2 ? shape[2] : 1
        let samplesPerBatch = shape.count > 2 ? (shape[1] * shape[2]) : shape[1]
        var result: [[String: Any]] = []
        let floats = audio.asArray(Float.self)
        let total = b * samplesPerBatch
        guard floats.count >= total else { return [] }
        for i in 0..<b {
            let start = i * samplesPerBatch
            let end = start + samplesPerBatch
            var slice = Array(floats[start..<end])
            if shape.count > 2 {
                // Convert NLC flat order to channels-first [C, L] flattened.
                let length = shape[1]
                let base = start
                var channelsFirst: [Float] = []
                channelsFirst.reserveCapacity(samplesPerBatch)
                for ch in 0..<channels {
                    for t in 0..<length {
                        channelsFirst.append(floats[base + t * channels + ch])
                    }
                }
                slice = channelsFirst
            }
            result.append([
                "tensor": slice,
                "sample_rate": sampleRate,
                "channels": channels
            ] as [String: Any])
        }
        return result
    }
}
