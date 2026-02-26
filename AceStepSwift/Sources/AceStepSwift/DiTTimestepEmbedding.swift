/**
 Sinusoidal timestep embedding + MLP. Returns (temb, timestep_proj). Mirrors dit_model.MLXTimestepEmbedding.
 */

import Foundation
import MLX
import MLXNN

public class DiTTimestepEmbedding: Module {
    public let inChannels: Int
    public let scale: Float
    public let linear1: Linear
    public let act1: SiLU
    public let linear2: Linear
    public let act2: SiLU
    public let timeProj: Linear

    public init(inChannels: Int = 256, timeEmbedDim: Int = 2048, scale: Float = 1000) {
        self.inChannels = inChannels
        self.scale = scale
        self.linear1 = Linear(inChannels, timeEmbedDim, bias: true)
        self.act1 = SiLU()
        self.linear2 = Linear(timeEmbedDim, timeEmbedDim, bias: true)
        self.act2 = SiLU()
        self.timeProj = Linear(timeEmbedDim, timeEmbedDim * 6, bias: true)
        super.init()
    }

    /// t: [B]. Returns (temb [B, D], timestepProj [B, 6, D]).
    public func call(_ t: MLXArray) -> (MLXArray, MLXArray) {
        let tFreq = sinusoidalEmbedding(t, dim: inChannels, scale: scale)
        var temb = linear1(tFreq)
        temb = act1(temb)
        temb = linear2(temb)
        let proj = timeProj(act2(temb))
        let (b, d6) = (proj.shape[0], proj.shape[1])
        let d = d6 / 6
        let timestepProj = proj.reshaped([b, 6, d])
        return (temb, timestepProj)
    }
}

private func sinusoidalEmbedding(_ t: MLXArray, dim: Int, scale: Float, maxPeriod: Int = 10000) -> MLXArray {
    let tScaled = t * scale
    let half = dim / 2
    let freqs = MLX.exp(-log(Float(maxPeriod)) * MLXArray((0..<half).map { Float($0) / Float(half) }))
    let args = tScaled.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
    let emb = concatenated([MLX.cos(args), MLX.sin(args)], axis: 1)
    if dim % 2 != 0 {
        return concatenated([emb, MLXArray.zeros([t.dim(0), 1])], axis: 1)
    }
    return emb
}
