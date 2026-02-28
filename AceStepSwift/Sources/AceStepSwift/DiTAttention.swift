/**
 Multi-head attention with QK-RMSNorm, RoPE (self), optional KV cache (cross), sliding-window mask. Mirrors dit_model.MLXAttention.
 */

import Foundation
import MLX
import MLXNN

public class DiTAttention: Module {
    public let hiddenSize: Int
    public let numHeads: Int
    public let numKvHeads: Int
    public let headDim: Int
    public let nRep: Int
    public let scale: Float
    public let layerIdx: Int
    public let isCrossAttention: Bool

    public let qProj: Linear
    public let kProj: Linear
    public let vProj: Linear
    public let oProj: Linear
    public let qNorm: RMSNorm
    public let kNorm: RMSNorm

    public init(
        hiddenSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        rmsNormEps: Float,
        attentionBias: Bool,
        layerIdx: Int,
        isCrossAttention: Bool
    ) {
        self.hiddenSize = hiddenSize
        self.numHeads = numAttentionHeads
        self.numKvHeads = numKeyValueHeads
        self.headDim = headDim
        self.nRep = numAttentionHeads / numKeyValueHeads
        self.scale = pow(Float(headDim), -0.5)
        self.layerIdx = layerIdx
        self.isCrossAttention = isCrossAttention

        self.qProj = Linear(hiddenSize, numAttentionHeads * headDim, bias: attentionBias)
        self.kProj = Linear(hiddenSize, numKeyValueHeads * headDim, bias: attentionBias)
        self.vProj = Linear(hiddenSize, numKeyValueHeads * headDim, bias: attentionBias)
        self.oProj = Linear(numAttentionHeads * headDim, hiddenSize, bias: attentionBias)
        self.qNorm = RMSNorm(dimensions: headDim, eps: rmsNormEps)
        self.kNorm = RMSNorm(dimensions: headDim, eps: rmsNormEps)
        super.init()
    }

    /// Self-attn: pass positionCosSin, no encoder. Cross-attn: pass encoderHiddenStates, optional cache.
    public func call(
        hiddenStates: MLXArray,
        positionCosSin: (MLXArray, MLXArray)? = nil,
        attentionMask: MLXArray? = nil,
        encoderHiddenStates: MLXArray? = nil,
        cache: DiTCrossAttentionCache? = nil,
        useCache: Bool = false
    ) -> MLXArray {
        let b = hiddenStates.dim(0)
        let l = hiddenStates.dim(1)

        var q = qProj(hiddenStates)
        q = q.reshaped([b, l, numHeads, headDim])
        q = qNorm(q)
        q = q.transposed(axes: [0, 2, 1, 3])

        var k: MLXArray
        var v: MLXArray
        if isCrossAttention, let enc = encoderHiddenStates {
            let encL = enc.dim(1)
            if let cache = cache, cache.isUpdated(layerIdx: layerIdx) {
                (k, v) = cache.get(layerIdx: layerIdx)
            } else {
                k = kProj(enc)
                k = k.reshaped([b, encL, numKvHeads, headDim])
                k = kNorm(k)
                k = k.transposed(axes: [0, 2, 1, 3])
                v = vProj(enc).reshaped([b, encL, numKvHeads, headDim]).transposed(axes: [0, 2, 1, 3])
                if let cache = cache, useCache {
                    cache.update(key: k, value: v, layerIdx: layerIdx)
                }
            }
        } else {
            k = kProj(hiddenStates)
            k = k.reshaped([b, l, numKvHeads, headDim])
            k = kNorm(k)
            k = k.transposed(axes: [0, 2, 1, 3])
            v = vProj(hiddenStates).reshaped([b, l, numKvHeads, headDim]).transposed(axes: [0, 2, 1, 3])
            if let (cos, sin) = positionCosSin {
                (q, k) = diTApplyRotaryPosEmb(q: q, k: k, cos: cos, sin: sin)
            }
        }

        k = repeatKv(k, nRep: nRep)
        v = repeatKv(v, nRep: nRep)

        // Cross-attn encoder mask is [B, encL]; MLX expects mask to broadcast with scores [B, numHeads, L, encL].
        // Expand to [B, 1, 1, encL] (Python: attention_mask.view(B, 1, 1, seq_len)).
        var maskForAttn = attentionMask
        if let m = maskForAttn, m.ndim == 2 {
            maskForAttn = m.expandedDimensions(axis: 1).expandedDimensions(axis: 2)
        }

        let attnOut = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: maskForAttn
        )
        let out = attnOut.transposed(axes: [0, 2, 1, 3]).reshaped([b, l, numHeads * headDim])
        return oProj(out)
    }

    private func repeatKv(_ x: MLXArray, nRep: Int) -> MLXArray {
        if nRep == 1 { return x }
        let (b, nKv, seqL, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        var y = x.expandedDimensions(axis: 2)
        y = MLX.broadcast(y, to: [b, nKv, nRep, seqL, d])
        return y.reshaped([b, nKv * nRep, seqL, d])
    }
}
