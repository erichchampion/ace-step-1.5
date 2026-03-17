/**
 MLX Audio Detokenizer: converts quantized audio tokens back to continuous acoustic representations.
 Process: embed→expand with special_tokens→encoder layers→project to acoustic dim.
 Mirrors Python AudioTokenDetokenizer.
 Load from audio_detokenizer.safetensors exported by export_mlx_tokenizer.py.
 */

import Foundation
import MLX
import MLXNN

/// Audio detokenizer module.
/// Input: quantized [B, T/pool, D] → Output: [B, T, acousticDim]
public class MLXAudioDetokenizer: Module {
    public let embedTokens: Linear
    public let layers: [ConditionEncoderLayer]
    public let norm: RMSNorm
    public let rotaryEmb: DiTRotaryEmbedding
    public let projOut: Linear
    public let config: AudioTokenizerConfig
    /// Learnable special tokens [1, poolWindowSize, D] added to each expanded patch.
    let specialTokens: MLXArray

    public init(config: AudioTokenizerConfig = .turbo) {
        self.config = config
        self.embedTokens = Linear(config.hiddenSize, config.hiddenSize)
        self.rotaryEmb = DiTRotaryEmbedding(headDim: config.headDim, maxLen: 32768, base: 1_000_000)
        self.layers = (0..<config.numAttentionPoolerLayers).map { i in
            let layerType = (i + 1) % 2 == 1 ? "sliding_attention" : "full_attention"
            return ConditionEncoderLayer(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                numAttentionHeads: config.numAttentionHeads,
                numKeyValueHeads: config.numKeyValueHeads,
                headDim: config.headDim,
                rmsNormEps: config.rmsNormEps,
                attentionBias: false,
                layerIdx: i,
                layerType: layerType
            )
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.projOut = Linear(config.hiddenSize, config.acousticHiddenDim)
        // Will be loaded from checkpoint; initialize to zeros.
        self.specialTokens = MLXArray.zeros([1, config.poolWindowSize, config.hiddenSize])
        super.init()
    }

    /// Detokenize quantized tokens back to continuous representations.
    /// Input: [B, T/pool, D] → Output: [B, T, acousticDim]
    public func detokenize(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let P = config.poolWindowSize
        var h = embedTokens(x)  // [B, T, D]
        // Expand: [B, T, D] → [B, T, P, D] (repeat each token P times)
        h = h.expandedDimensions(axis: 2)  // [B, T, 1, D]
        h = MLX.broadcast(h, to: [B, T, P, h.dim(3)])  // [B, T, P, D]
        // Add learnable special tokens
        let expandedSpecial = MLX.broadcast(specialTokens, to: [B, T, P, specialTokens.dim(2)])
        h = h + expandedSpecial
        // Reshape to [(B*T), P, D] for processing
        h = h.reshaped([B * T, P, h.dim(3)])

        let (cos, sin) = rotaryEmb.call(seqLen: P)
        for layer in layers {
            h = layer.call(hiddenStates: h, positionCosSin: (cos, sin), attentionMask: nil)
        }
        h = norm(h)
        // Project to acoustic dimension
        h = projOut(h)  // [(B*T), P, acousticDim]
        // Reshape back: [(B*T), P, acousticDim] → [B, T*P, acousticDim]
        return h.reshaped([B, T * P, -1])
    }

    /// Load weights from safetensors file.
    public func loadWeights(from url: URL) throws {
        let params = try loadDiTParameters(from: url)
        update(parameters: params)
        MLX.eval(parameters())
    }
}
