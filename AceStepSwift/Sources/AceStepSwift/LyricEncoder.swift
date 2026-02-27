/**
 Lyric encoder: embed -> 8 encoder layers (RoPE, bidirectional) -> norm. Mirrors Python AceStepLyricEncoder.
 Load encoder.lyric_encoder.* from checkpoint.
 */

import Foundation
import MLX
import MLXNN

public class LyricEncoder: Module {
    public let embedTokens: Linear
    public let layers: [ConditionEncoderLayer]
    public let norm: RMSNorm
    public let rotaryEmb: DiTRotaryEmbedding
    private let slidingWindow: Int

    public init(
        textHiddenDim: Int = 1024,
        hiddenSize: Int = 2048,
        intermediateSize: Int = 6144,
        numLayers: Int = 8,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        rmsNormEps: Float = 1e-6,
        attentionBias: Bool = false,
        slidingWindow: Int = 128
    ) {
        self.slidingWindow = slidingWindow
        self.embedTokens = Linear(textHiddenDim, hiddenSize)
        self.rotaryEmb = DiTRotaryEmbedding(headDim: headDim, maxLen: 32768, base: 1_000_000)
        self.layers = (0..<numLayers).map { i in
            let layerType = (i + 1) % 2 == 1 ? "sliding_attention" : "full_attention"
            return ConditionEncoderLayer(
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                numAttentionHeads: numAttentionHeads,
                numKeyValueHeads: numKeyValueHeads,
                headDim: headDim,
                rmsNormEps: rmsNormEps,
                attentionBias: attentionBias,
                layerIdx: i,
                layerType: layerType
            )
        }
        self.norm = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        super.init()
    }

    /// inputs_embeds [B, L, text_hidden_dim], attention_mask [B, L] (1=valid, 0=padding).
    public func call(inputsEmbeds: MLXArray, attentionMask: MLXArray) -> MLXArray {
        var h = embedTokens(inputsEmbeds)
        guard h.ndim >= 3 else {
            return MLXArray.zeros([1, 0, 2048])
        }
        let seqLen = h.dim(1)
        let (cos, sin) = rotaryEmb.call(seqLen: seqLen)
        let fullMask = createEncoderPaddingMask(attentionMask: attentionMask)
        let slidingMask: MLXArray? = slidingWindow > 0 ? diTCreateSlidingWindowMask(seqLen: seqLen, windowSize: slidingWindow) : nil
        for layer in layers {
            let layerMask = layer.layerType == "sliding_attention" ? slidingMask : nil
            let combinedMask: MLXArray? = {
                guard let slide = layerMask else { return fullMask }
                let b = fullMask.dim(0)
                let expanded = MLX.broadcast(slide, to: [b, 1, seqLen, seqLen])
                return MLX.minimum(fullMask, expanded)
            }()
            h = layer.call(hiddenStates: h, positionCosSin: (cos, sin), attentionMask: combinedMask)
        }
        return norm(h)
    }
}
