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
        // Ensure [1, 6, D]; checkpoint may have [1, D, 6]. temb is timestep_proj [B, 6, D]. Flatten + 1D slice so layout is guaranteed.
        let table = scaleShiftTable.dim(1) == 6 ? scaleShiftTable : scaleShiftTable.transposed(axes: [0, 2, 1])
        let modulation = table + temb
        let b = modulation.dim(0)
        let d = modulation.dim(2)
        let flatLen = modulation.dim(0) * modulation.dim(1) * modulation.dim(2)
        let flat = modulation.reshaped([flatLen])
        let step = b * d
        func part(_ i: Int) -> MLXArray {
            let p = flat[(i * step)..<((i + 1) * step)].reshaped([b, 1, d])
            return p.dim(1) != 1 ? p.transposed(axes: [0, 2, 1]) : p
        }
        let shiftMsa = part(0)
        let scaleMsa = part(1)
        let gateMsa = part(2)
        let cShiftMsa = part(3)
        let cScaleMsa = part(4)
        let cGateMsa = part(5)

        var h = hiddenStates
        var normed = selfAttnNorm(h)
        normed = normed * (1.0 + scaleMsa) + shiftMsa
        h = h + selfAttn.call(hiddenStates: normed, positionCosSin: positionCosSin, attentionMask: selfAttnMask) * gateMsa

        normed = crossAttnNorm(h)
        h = h + crossAttn.call(hiddenStates: normed, attentionMask: encoderAttentionMask, encoderHiddenStates: encoderHiddenStates, cache: cache, useCache: useCache)

        normed = mlpNorm(h)
        normed = normed * (1.0 + cScaleMsa) + cShiftMsa
        h = h + mlp(normed) * cGateMsa
        return h
    }
}
