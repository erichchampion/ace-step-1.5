/**
 Single DiT transformer layer: AdaLN, self-attn, cross-attn, MLP. Mirrors dit_model.MLXDiTLayer.
 */

import Foundation
import MLX
import MLXNN

public class DiTLayer: Module {
    public let layerType: String
    public let selfAttnNorm: RMSNorm
    public let selfAttn: DiTAttention
    public let crossAttnNorm: RMSNorm
    public let crossAttn: DiTAttention
    public let mlpNorm: RMSNorm
    public let mlp: DiTSwiGLUMLP
    public let scaleShiftTable: MLXArray

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        rmsNormEps: Float,
        attentionBias: Bool,
        layerIdx: Int,
        layerType: String,
        slidingWindow: Int?
    ) {
        self.layerType = layerType
        self.selfAttnNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
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
        self.crossAttnNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.crossAttn = DiTAttention(
            hiddenSize: hiddenSize,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim,
            rmsNormEps: rmsNormEps,
            attentionBias: attentionBias,
            layerIdx: layerIdx,
            isCrossAttention: true
        )
        self.mlpNorm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.mlp = DiTSwiGLUMLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        self.scaleShiftTable = MLXArray.zeros([1, 6, hiddenSize])
        super.init()
    }

    public func call(
        hiddenStates: MLXArray,
        positionCosSin: (MLXArray, MLXArray),
        temb: MLXArray,
        selfAttnMask: MLXArray?,
        encoderHiddenStates: MLXArray?,
        encoderAttentionMask: MLXArray?,
        cache: DiTCrossAttentionCache? = nil,
        useCache: Bool = false
    ) -> MLXArray {
        let tembExpanded = temb.expandedDimensions(axis: 1)
        let modulation = scaleShiftTable + tembExpanded
        let parts = split(modulation, parts: 6, axis: 1)
        let (shiftMsa, scaleMsa, gateMsa) = (parts[0], parts[1], parts[2])
        let (cShiftMsa, cScaleMsa, cGateMsa) = (parts[3], parts[4], parts[5])

        var h = hiddenStates
        var normed = selfAttnNorm(h)
        normed = normed * (1.0 + scaleMsa) + shiftMsa
        h = h + selfAttn.call(hiddenStates: normed, positionCosSin: positionCosSin, attentionMask: selfAttnMask) * gateMsa

        normed = crossAttnNorm(h)
        h = h + crossAttn.call(hiddenStates: normed, encoderHiddenStates: encoderHiddenStates, cache: cache, useCache: useCache)

        normed = mlpNorm(h)
        normed = normed * (1.0 + cScaleMsa) + cShiftMsa
        h = h + mlp(normed) * cGateMsa
        return h
    }
}
