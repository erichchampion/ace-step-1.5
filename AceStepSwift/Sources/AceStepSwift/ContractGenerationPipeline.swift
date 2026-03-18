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
    var t = Int(ceil(samples / Double(vaeLatentToSamplesFactor)))
    if t % 2 != 0 {
        t -= 1 // Ensure parity for patch_size=2 downsampling without dropping residuals in CoreML
    }
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
    private var conditioningProvider: ConditioningProvider?
    public let maxDuration: Double

    public var isInitialized: Bool { true }

    /// - Parameters:
    ///   - stepper: Per-step diffusion (e.g. FakeDiffusionStepper or MLXDiTStepper).
    ///   - decoder: VAE decode latent → audio (e.g. FakeVAEDecoder or MLXVAEDecoder).
    ///   - sampleRate: Output waveform sample rate (default 48000).
    ///   - conditioningProvider: When non-nil, called with (params, latentLength, sampleRate) to build DiTConditions. Return encoderHiddenStates [B, encL, 2048] and contextLatents [B, latentLength, 128] for meaningful output. When nil, zeros are used and output is not meaningful (matches Python only when real conditioning from prepare_condition is passed).
    ///   - maxDuration: Maximum duration supported by this pipeline instance.
    public init(
        stepper: DiffusionStepper,
        decoder: VAEDecoder,
        sampleRate: Int = AceStepConstants.defaultSampleRate,
        conditioningProvider: ConditioningProvider? = nil,
        maxDuration: Double = 120.0
    ) {
        self.stepper = stepper
        self.decoder = decoder
        self.sampleRate = sampleRate
        self.conditioningProvider = conditioningProvider
        self.maxDuration = maxDuration
    }

    /// Replace the conditioning provider (e.g. after lazily loading cover models).
    public func updateConditioningProvider(_ provider: ConditioningProvider?) {
        self.conditioningProvider = provider
    }

    public func run(params: GenerationParams, config: GenerationConfig, progress: ((Double, String) -> Void)?) async throws -> GenerationResult {
        let b = config.batchSize
        let t = latentLengthFromDuration(durationSeconds: params.duration, sampleRate: sampleRate)

        var schedule = DiffusionSchedule.getTimestepSchedule(
            shift: params.shift,
            timesteps: params.timesteps,
            inferSteps: params.inferenceSteps > 0 ? params.inferenceSteps : nil
        )

        // Cover noise blending: truncate schedule when coverNoiseStrength > 0.
        // Python: effective_noise_level = 1 - cover_noise_strength, find nearest_t,
        // truncate schedule to start from nearest_t.
        if params.coverNoiseStrength > 0, !schedule.isEmpty {
            let effectiveNoiseLevel = 1.0 - params.coverNoiseStrength
            let nearestT = schedule.min(by: { abs($0 - effectiveNoiseLevel) < abs($1 - effectiveNoiseLevel) })!
            if let startIdx = schedule.firstIndex(of: nearestT) {
                let originalCount = schedule.count
                schedule = Array(schedule[startIdx...])
                print("[ContractGenerationPipeline] Cover noise: strength=\(params.coverNoiseStrength), effectiveNoiseLevel=\(effectiveNoiseLevel), nearestT=\(nearestT), steps \(originalCount)→\(schedule.count)")
            }
        }

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
        if conditions.encoderHiddenStates == nil && (stepper is MLXDiTStepper || stepper is CoreMLDiTStepper) {
            throw ContractGenerationPipelineError.missingConditioning
        }
        let usingDefaultConditions = (conditions.encoderHiddenStates == nil && conditions.contextLatents == nil)
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

        MLX.GPU.clearCache()
        
        // Mid-loop conditioning switch: Python switches from cover→text2music at step `cover_steps`
        let coverSteps = params.audioCoverStrength < 1.0 ? Int(Double(schedule.count) * params.audioCoverStrength) : schedule.count
        var switchedToNonCover = false
        
        var apgMomentumState: [String: MLXArray]? = [:]
        for (stepIdx, timestepVal) in schedule.enumerated() {
            print("[PipelineLoop] stepIdx=\(stepIdx) t=\(timestepVal)")
            
            // Switch to non-cover conditions at step `coverSteps` (Python: audio_cover_strength)
            if stepIdx >= coverSteps, !switchedToNonCover, let nonCover = conditions.nonCoverConditions?.value {
                switchedToNonCover = true
                conditions.encoderHiddenStates = nonCover.encoderHiddenStates
                conditions.encoderAttentionMask = nonCover.encoderAttentionMask
                conditions.contextLatents = nonCover.contextLatents
                // Reset APG momentum state on conditioning switch (matches Python KV cache reset)
                apgMomentumState = [:]
                print("[PipelineLoop] Switched to non-cover conditions at step \(stepIdx)/\(schedule.count)")
            }
            
            let nextT: Float? = (stepIdx + 1 < schedule.count) ? Float(schedule[stepIdx + 1]) : nil
            let previousLatent = xt
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
            let vtDiff = previousLatent - xt
            let vtStats = tensorMeanStd(vtDiff)
            debugPrint(
                "[GenerationDiagnostics] step=\(stepIdx + 1)/\(schedule.count) t=\(formatStat(Float(timestepVal))) latent_mean=\(formatStat(stepStats.mean)) latent_std=\(formatStat(stepStats.std)) vtDiff_mean=\(formatStat(vtStats.mean)) vtDiff_std=\(formatStat(vtStats.std))"
            )
            #endif
            let frac = Double(stepIdx + 1) / Double(schedule.count)
            progress?(frac * 0.9, "Diffusion step \(stepIdx + 1)/\(schedule.count)")
        }

        progress?(0.95, "Decoding")
        MLX.GPU.clearCache()
        var decodeLatent = xt
        if params.latentShift != 0.0 || params.latentRescale != 1.0 {
            decodeLatent = decodeLatent * Float(params.latentRescale) + Float(params.latentShift)
        }
        guard decodeLatent.ndim == 3, decodeLatent.dim(2) == latentChannels else {
            throw ContractGenerationPipelineError.invalidLatentShape(decodeLatent.shape)
        }
        #if DEBUG
        if decodeLatent.ndim == 3 {
            MLX.eval(decodeLatent) // Force evaluation for memory safety
            let flatX = decodeLatent.asArray(Float.self)
            let b = decodeLatent.dim(0)
            let len = decodeLatent.dim(1)
            let c = decodeLatent.dim(2)
            if b == 1 && c == 64 && len > 0 {
                var lSum: Float = 0
                var rSum: Float = 0
                for i in 0..<len {
                    for j in 0..<32 { lSum += flatX[i * 64 + j] }
                    for j in 32..<64 { rSum += flatX[i * 64 + j] }
                }
                let lMean = lSum / Float(len * 32)
                let rMean = rSum / Float(len * 32)
                
                var num: Float = 0
                var lDenom: Float = 0
                var rDenom: Float = 0
                
                for i in 0..<len {
                    for j in 0..<32 {
                        let lVal = flatX[i * 64 + j]
                        let rVal = flatX[i * 64 + j + 32]
                        let lDiff = lVal - lMean
                        let rDiff = rVal - rMean
                        num += lDiff * rDiff
                        lDenom += lDiff * lDiff
                        rDenom += rDiff * rDiff
                    }
                }
                let corr = num / sqrt(lDenom * rDenom)
                debugPrint("[GenerationDiagnostics] decode_latent shape=\(decodeLatent.shape) L/R correlation=\(String(format: "%.6f", corr))")
            }
        }
        do {
            let supportDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
                ?? URL(fileURLWithPath: NSTemporaryDirectory())
            let npyURL = supportDir.appendingPathComponent("swift_latents.npy")
            try MLX.save(array: decodeLatent, url: npyURL)
            print("[DEBUG] Wrote swift_latents.npy to \(npyURL.path)")
        } catch {
            print("[DEBUG] Failed to write swift_latents.npy: \(error)")
        }
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
        MLX.GPU.clearCache()

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
            initialLatents: try align("initialLatents", conditions.initialLatents),
            nonCoverConditions: conditions.nonCoverConditions != nil
                ? try alignConditionsToBatch(conditions.nonCoverConditions!.value, batchSize: batchSize)
                : nil
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
        // Guidance criteria: scale > 1.0 and all required conditions for unconditioned branch exist.
        guard
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

        let predCond: MLXArray
        let predUncond: MLXArray

        if let mlxStepper = stepper as? MLXDiTStepper {
            // MLX Optimized Path: run shared forward pass with doubled batch [2*B, ...]
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
                useCache: false // Cross-attention cache currently expects stable batch dimension
            )
            predCond = vtDoubled[0..<b, 0..<t, 0..<c]
            predUncond = vtDoubled[b..<(b * 2), 0..<t, 0..<c]
        } else {
            // General Path (e.g. CoreML): run two sequential forward passes
            // 1. Conditional branch
            predCond = stepper.predictVelocity(
                currentLatent: currentLatent,
                timestep: timestep,
                conditions: conditions,
                useCache: true
            )

            // 2. Unconditional branch (using nullConditionEmbedding instead of normal encoder hidden states)
            let nullExpanded = MLX.broadcast(nullCond, to: enc.shape)
            var uncondConditions = conditions
            uncondConditions.encoderHiddenStates = nullExpanded
            
            predUncond = stepper.predictVelocity(
                currentLatent: currentLatent,
                timestep: timestep,
                conditions: uncondConditions,
                useCache: true
            )
        }

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
        let samplesPerBatch = shape.count > 2 ? shape[1] : shape[1]
        
        var result: [[String: Any]] = []
        
        for i in 0..<b {
            let bIndex = i..<(i + 1)
            let tIndex = 0..<samplesPerBatch
            
            let slice: [Float]
            if channels == 2 {
                let leftNode = audio[bIndex, tIndex, 0..<1].squeezed().contiguous()
                let rightNode = audio[bIndex, tIndex, 1..<2].squeezed().contiguous()
                MLX.eval(leftNode, rightNode)
                let leftChannel = leftNode.asArray(Float.self)
                let rightChannel = rightNode.asArray(Float.self)
                
                var interleaved = [Float](repeating: 0, count: samplesPerBatch * 2)
                for f in 0..<samplesPerBatch {
                    interleaved[f * 2] = leftChannel[f]
                    interleaved[f * 2 + 1] = rightChannel[f]
                }
                slice = interleaved
            } else {
                let monoNode = audio[bIndex, tIndex].squeezed().contiguous()
                MLX.eval(monoNode)
                slice = monoNode.asArray(Float.self)
            }
            
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
