/**
 Real DiT stepper: runs DiTDecoder forward and applies ODE step. Conforms to DiffusionStepper.
 */

import Foundation
import MLX
import MLXNN

/// DiT decoder output latent dim (audio_acoustic_hidden_dim).
public let ditOutputLatentChannels = 64

/// Context latent channels (in_channels - 64 = 192 - 64).
private let ditContextChannels = 128

/// Encoder hidden size (condition embedder dim).
private let ditEncoderHiddenSize = 2048

/// Real diffusion stepper using DiTDecoder. Load weights with decoder.update(parameters:).
public final class MLXDiTStepper: DiffusionStepper {
    public let decoder: DiTDecoder
    private let cache: DiTCrossAttentionCache

    public init(decoder: DiTDecoder) {
        self.decoder = decoder
        self.cache = DiTCrossAttentionCache()
    }

    public func step(currentLatent: MLXArray, timestep: Float, conditions: DiTConditions, nextTimestep: Float?) -> MLXArray {
        let b = currentLatent.dim(0)
        let t = currentLatent.dim(1)

        let enc = conditions.encoderHiddenStates ?? MLXArray.zeros([b, 1, ditEncoderHiddenSize])
        let ctx = conditions.contextLatents ?? MLXArray.zeros([b, t, ditContextChannels])

        let timestepArr = MLXArray([Float](repeating: timestep, count: b))
        let timestepRArr = MLXArray([Float](repeating: 0, count: b))

        let (vt, _) = decoder.call(
            hiddenStates: currentLatent,
            timestep: timestepArr,
            timestepR: timestepRArr,
            encoderHiddenStates: enc,
            contextLatents: ctx,
            cache: cache,
            useCache: true
        )

        if let nextT = nextTimestep {
            let dt = timestep - nextT
            return currentLatent - vt * dt
        } else {
            return currentLatent - vt * timestep
        }
    }
}
