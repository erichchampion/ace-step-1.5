/**
 Pipeline that runs diffusion (via DiffusionStepper) then VAE decode (via VAEDecoder).
 Uses timestep schedule and optional seed; returns GenerationResult with audios.
 Enables end-to-end path with fake or real DiT/VAE.
 (Python reference: acestep/core/generation/handler/generate_music_execute.py and acestep_v15_pipeline.py)
 */

import Foundation
import MLX
import MLXRandom

private enum ContractGenerationPipelineError: LocalizedError {
    case missingConditioning
    case conditionBatchMismatch(name: String, expected: Int, actual: Int)
    case invalidLatentShape([Int])
    case invalidDecodedAudioShape([Int])

    var errorDescription: String? {
        switch self {
        case .missingConditioning:
            return "Missing DiT conditioning (encoderHiddenStates/contextLatents). Real generation requires a ConditioningProvider matching Python prepare_condition."
        case .conditionBatchMismatch(let name, let expected, let actual):
            return "Condition batch mismatch for \(name): expected batch \(expected), got \(actual). Provide batch-aligned conditioning or batch=1."
        case .invalidLatentShape(let shape):
            return "Invalid latent shape before VAE decode: \(shape). Expected [B, T, \(latentChannels)]."
        case .invalidDecodedAudioShape(let shape):
            return "Invalid decoded audio shape: \(shape). Expected [B, samples] or [B, samples, channels]."
        }
    }
}

/// Default latent time dimension when duration is invalid or not set.
private let defaultLatentLength = 100

/// Latent channel dimension (DiT/VAE).
private let latentChannels = 64

/// Minimum latent length (matches Python conditioning_target max(128, ...) padding for short clips).
private let minLatentLength = 128

/// Returns latent time steps T from duration (seconds) and sample rate.
/// If duration <= 0, returns defaultLatentLength. Uses at least minLatentLength so short clips
/// match Python's padded length and precomputed conditioning (e.g. 128 frames) fits. Exposed for testing.
public func latentLengthFromDuration(durationSeconds: Double, sampleRate: Int) -> Int {
    guard durationSeconds > 0 else { return defaultLatentLength }
    let samples = durationSeconds * Double(sampleRate)
    let t = Int(ceil(samples / Double(vaeLatentToSamplesFactor)))
    return max(minLatentLength, max(1, t))
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
        #if DEBUG
        if let firstT = schedule.first, let lastT = schedule.last {
            debugPrint("[GenerationDiagnostics] timestep_schedule count=\(schedule.count) first=\(formatStat(Float(firstT))) last=\(formatStat(Float(lastT)))")
        }
        #endif
        progress?(0, "Starting diffusion")

        var conditions = conditioningProvider?(params, t, sampleRate) ?? DiTConditions()
        conditions = try alignConditionsToBatch(conditions, batchSize: b)
        #if DEBUG
        if let enc = conditions.encoderHiddenStates, let ctx = conditions.contextLatents {
            let encStats = tensorMeanStd(enc)
            let ctxStats = tensorMeanStd(ctx)
            debugPrint(
                "[ConditioningDiagnostics] pipeline enc(std=\(formatStat(encStats.std)),shape=\(enc.shape)) " +
                "ctx(std=\(formatStat(ctxStats.std)),shape=\(ctx.shape))"
            )
        }
        #endif
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
        #if DEBUG
        let initialStats = tensorMeanStd(xt)
        debugPrint(
            "[GenerationDiagnostics] initial_latent mean=\(formatStat(initialStats.mean)) std=\(formatStat(initialStats.std))"
        )
        #endif

        var apgMomentumState: [String: MLXArray]? = [:]
        for (stepIdx, timestepVal) in schedule.enumerated() {
            let nextT: Float? = (stepIdx + 1 < schedule.count) ? Float(schedule[stepIdx + 1]) : nil
            xt = runDiffusionStep(
                currentLatent: xt,
                timestep: Float(timestepVal),
                nextTimestep: nextT,
                conditions: conditions,
                params: params,
                momentumState: &apgMomentumState
            )
            #if DEBUG
            let stepStats = tensorMeanStd(xt)
            let variance = stepStats.std * stepStats.std
            debugPrint(
                "[GenerationDiagnostics] step=\(stepIdx + 1)/\(schedule.count) t=\(formatStat(Float(timestepVal))) latent_var=\(formatStat(variance)) mean=\(formatStat(stepStats.mean)) std=\(formatStat(stepStats.std))"
            )
            #endif
            let frac = Double(stepIdx + 1) / Double(schedule.count)
            progress?(frac * 0.9, "Diffusion step \(stepIdx + 1)/\(schedule.count)")
        }

        progress?(0.95, "Decoding")
        var decodeLatent = xt
        if params.latentShift != 0.0 || params.latentRescale != 1.0 {
            decodeLatent = decodeLatent * Float(params.latentRescale) + Float(params.latentShift)
        }
        guard decodeLatent.ndim == 3, decodeLatent.dim(2) == latentChannels else {
            throw ContractGenerationPipelineError.invalidLatentShape(decodeLatent.shape)
        }
        #if DEBUG
        debugPrint("[GenerationDiagnostics] decode_latent shape=\(decodeLatent.shape)")
        #endif
        var audio = decoder.decode(latent: decodeLatent)
        guard audio.ndim == 2 || audio.ndim == 3 else {
            throw ContractGenerationPipelineError.invalidDecodedAudioShape(audio.shape)
        }
        // Enforce contract length T*2048: some VAE implementations (e.g. MLX ConvTranspose1d) can
        // produce slightly longer output; trimming avoids extra samples that can sound like repetition.
        let tLatent = decodeLatent.dim(1)
        let expectedSamples = tLatent * vaeLatentToSamplesFactor
        if audio.ndim == 3, audio.dim(1) > expectedSamples {
            audio = audio[0..<audio.dim(0), 0..<expectedSamples, 0..<audio.dim(2)]
        } else if audio.ndim == 2, audio.dim(1) > expectedSamples {
            audio = audio[0..<audio.dim(0), 0..<expectedSamples]
        }
        #if DEBUG
        if audio.ndim == 3, audio.dim(2) == 2 {
            let b0 = 0..<1
            let t = 0..<audio.dim(1)
            let left = audio[b0, t, 0..<1].squeezed(axis: 2)
            let right = audio[b0, t, 1..<2].squeezed(axis: 2)
            let lStats = tensorMeanStd(left)
            let rStats = tensorMeanStd(right)
            debugPrint(
                "[ChannelDiagnostics] directIndexing L(mean=\(formatStat(lStats.mean)),std=\(formatStat(lStats.std))) " +
                "R(mean=\(formatStat(rStats.mean)),std=\(formatStat(rStats.std)))"
            )
        }
        #endif
        audio = normalizePeakIfNeeded(audio)
        #if DEBUG
        let audioStats = tensorMeanStd(audio)
        let audioPeak = MLX.abs(audio).max().item(Float.self)
        debugPrint(
            "[GenerationDiagnostics] final_audio shape=\(audio.shape) peak=\(formatStat(audioPeak)) mean=\(formatStat(audioStats.mean)) std=\(formatStat(audioStats.std))"
        )
        #endif
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
            encoderAttentionMask: try align("encoderAttentionMask", conditions.encoderAttentionMask),
            nullConditionEmbedding: conditions.nullConditionEmbedding,
            initialLatents: try align("initialLatents", conditions.initialLatents)
        )
    }

    private func runDiffusionStep(
        currentLatent: MLXArray,
        timestep: Float,
        nextTimestep: Float?,
        conditions: DiTConditions,
        params: GenerationParams,
        momentumState: inout [String: MLXArray]?
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

        let b = currentLatent.dim(0)
        let t = currentLatent.dim(1)
        let c = currentLatent.dim(2)

        let xIn = concatenated([currentLatent, currentLatent], axis: 0)
        let nullExpanded = MLX.broadcast(nullCond, to: enc.shape)
        let encDoubled = concatenated([enc, nullExpanded], axis: 0)
        let ctxDoubled = concatenated([ctx, ctx], axis: 0)
        let maskDoubled: MLXArray? = conditions.encoderAttentionMask.map { concatenated([$0, $0], axis: 0) }
        let cfgConditions = DiTConditions(
            encoderHiddenStates: encDoubled,
            contextLatents: ctxDoubled,
            encoderAttentionMask: maskDoubled,
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

    /// Peak-normalize to at most 1.0 when any value exceeds 1.0 (matches Python generate_music_decode). Handles 2D [B, samples] and 3D [B, L, C].
    private func normalizePeakIfNeeded(_ audio: MLXArray) -> MLXArray {
        let peak: MLXArray
        if audio.ndim == 3 {
            peak = MLX.abs(audio).max(axes: [1, 2], keepDims: true)
        } else if audio.ndim == 2 {
            peak = MLX.abs(audio).max(axes: [1], keepDims: true)
        } else {
            return audio
        }
        let maxPeak = peak.max().item(Float.self)
        guard maxPeak > 1.0 else { return audio }
        let safePeak = MLX.where(peak .> 1.0, peak, MLXArray(1.0 as Float))
        return audio / safePeak
    }

    private func tensorMeanStd(_ x: MLXArray) -> (mean: Float, std: Float) {
        let mean = x.mean().item(Float.self)
        let centered = x - MLXArray(mean)
        let variance = (centered * centered).mean().item(Float.self)
        return (mean, sqrt(max(variance, 0.0)))
    }

    private func formatStat(_ value: Float) -> String {
        String(format: "%.6f", value)
    }

    private func stats(_ x: [Float]) -> (mean: Float, std: Float) {
        guard !x.isEmpty else { return (0, 0) }
        let mean = x.reduce(0, +) / Float(x.count)
        let variance = x.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(x.count)
        return (mean, sqrt(max(variance, 0.0)))
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
            let slice = Array(floats[start..<end])
            #if DEBUG
            if channels == 2 && slice.count >= 4 {
                var leftInterleaved: [Float] = []
                var rightInterleaved: [Float] = []
                leftInterleaved.reserveCapacity(slice.count / 2)
                rightInterleaved.reserveCapacity(slice.count / 2)
                for j in stride(from: 0, to: slice.count - 1, by: 2) {
                    leftInterleaved.append(slice[j])
                    rightInterleaved.append(slice[j + 1])
                }
                let interLeft = stats(leftInterleaved)
                let interRight = stats(rightInterleaved)

                let half = slice.count / 2
                let leftChannelsFirst = Array(slice[0..<half])
                let rightChannelsFirst = Array(slice[half..<(half * 2)])
                let cfLeft = stats(leftChannelsFirst)
                let cfRight = stats(rightChannelsFirst)

                let preview = Array(slice.prefix(12)).map { formatStat($0) }.joined(separator: ",")
                debugPrint(
                    "[LayoutDiagnostics] batch=\(i) interleaved(Lstd=\(formatStat(interLeft.std)),Rstd=\(formatStat(interRight.std)),Lmean=\(formatStat(interLeft.mean)),Rmean=\(formatStat(interRight.mean))) " +
                    "channelsFirst(Lstd=\(formatStat(cfLeft.std)),Rstd=\(formatStat(cfRight.std)),Lmean=\(formatStat(cfLeft.mean)),Rmean=\(formatStat(cfRight.mean))) first12=[\(preview)]"
                )
            }
            #endif
            result.append([
                "tensor": slice,
                "sample_rate": sampleRate,
                "channels": channels
            ] as [String: Any])
        }
        return result
    }
}
