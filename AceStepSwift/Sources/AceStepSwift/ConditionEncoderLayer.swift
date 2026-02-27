/**
 Encoder layer for condition encoder (lyric/timbre): self-attention + MLP with residual and RoPE.
 Mirrors Python AceStepEncoderLayer (no cross-attention, no scale-shift).
 */

import Foundation
import MLX
import MLXNN

public class ConditionEncoderLayer: Module {
    public let inputNorm: RMSNorm
    public let selfAttn: DiTAttention
    public let postAttentionNorm: RMSNorm
    public let mlp: DiTSwiGLUMLP
    public let layerType: String

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        rmsNormEps: Float,
        attentionBias: Bool,
        layerIdx: Int,
        layerType: String
    ) {
        self.layerType = layerType
        self.inputNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.selfAttn = DiTAttention(
            hiddenSize: hiddenSize,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim,
            rmsNormEps: rmsNormEps,
            attentionBias: attentionBias,
            layerIdx: layerIdx,
            isCrossAttention: false
        )
        self.postAttentionNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.mlp = DiTSwiGLUMLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        super.init()
    }

    /// hiddenStates [B, L, D], positionCosSin (cos, sin) for RoPE, attentionMask [B, 1, L, L] additive (0 keep, -1e9 mask) or nil.
    public func call(
        hiddenStates: MLXArray,
        positionCosSin: (MLXArray, MLXArray),
        attentionMask: MLXArray?
    ) -> MLXArray {
        var h = hiddenStates
        var normed = inputNorm(h)
        h = h + selfAttn.call(hiddenStates: normed, positionCosSin: positionCosSin, attentionMask: attentionMask)
        normed = postAttentionNorm(h)
        h = h + mlp(normed)
        return h
    }
}
