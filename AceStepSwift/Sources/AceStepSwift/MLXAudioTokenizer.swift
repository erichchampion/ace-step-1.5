/**
 MLX Audio Tokenizer: converts continuous acoustic latents into quantized tokens.
 Process: project→pool patches (AttentionPooler)→quantize (ResidualFSQ).
 Mirrors Python AceStepAudioTokenizer + top-level tokenize() method.
 Load from audio_tokenizer.safetensors exported by export_mlx_tokenizer.py.
 */

import Foundation
import MLX
import MLXNN

/// Configuration for audio tokenizer/detokenizer.
public struct AudioTokenizerConfig: Sendable {
    public var hiddenSize: Int
    public var acousticHiddenDim: Int
    public var poolWindowSize: Int
    public var fsqDim: Int
    public var fsqInputLevels: [Int]
    public var fsqInputNumQuantizers: Int
    public var numAttentionPoolerLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var slidingWindow: Int

    /// Default turbo config from config.json.
    public static let turbo = AudioTokenizerConfig(
        hiddenSize: 2048,
        acousticHiddenDim: 64,
        poolWindowSize: 5,
        fsqDim: 2048,
        fsqInputLevels: [8, 8, 8, 5, 5, 5],
        fsqInputNumQuantizers: 1,
        numAttentionPoolerLayers: 2,
        numAttentionHeads: 16,
        numKeyValueHeads: 8,
        headDim: 128,
        intermediateSize: 6144,
        rmsNormEps: 1e-6,
        slidingWindow: 128
    )
}

/// Finite Scalar Quantizer: project to levels, round, project back.
/// Python: vector_quantize_pytorch.ResidualFSQ with project_in, project_out.
public class ResidualFSQ: Module {
    public let projectIn: Linear
    public let projectOut: Linear
    private let levels: [Int]

    public init(dim: Int, levels: [Int], numQuantizers: Int) {
        let codebookDim = levels.count
        self.levels = levels
        self.projectIn = Linear(dim, codebookDim)
        self.projectOut = Linear(codebookDim, dim)
        super.init()
    }

    /// Quantize: project → round-to-levels → project back.
    /// Input: [B, T, D] → Output: (quantized [B, T, D], indices [B, T, codebookDim])
    public func call(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let projected = projectIn(x)  // [B, T, codebookDim]
        // FSQ: for each dim, round to nearest level in [-(L-1)/2, ..., (L-1)/2]
        var quantized = projected
        for (i, level) in levels.enumerated() {
            let halfSteps = Float(level - 1) / 2.0
            // Bound encoding via tanh → scale to [-halfSteps, halfSteps] → round
            let slice = quantized[.ellipsis, i...i]
            let bounded = MLX.tanh(slice) * halfSteps
            let rounded = MLX.round(bounded)
            // Replace in-place via concatenation
            let before = i > 0 ? quantized[.ellipsis, 0..<i] : nil
            let after = i + 1 < levels.count ? quantized[.ellipsis, (i+1)..<levels.count] : nil
            var parts: [MLXArray] = []
            if let b = before { parts.append(b) }
            parts.append(rounded)
            if let a = after { parts.append(a) }
            quantized = concatenated(parts, axis: -1)
        }
        // Straight-through estimator not needed at inference (no gradients)
        let reconstructed = projectOut(quantized)  // [B, T, D]
        return (reconstructed, quantized)
    }
}

/// Attention-based pooler: prepend CLS token, run encoder layers, extract CLS output.
/// Mirrors Python AttentionPooler.
public class AttentionPooler: Module {
    public let embedTokens: Linear
    public let layers: [ConditionEncoderLayer]
    public let norm: RMSNorm
    public let rotaryEmb: DiTRotaryEmbedding
    /// CLS-like special token [1, 1, D].
    let specialToken: MLXArray

    public init(config: AudioTokenizerConfig) {
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
        // Will be loaded from checkpoint; initialize to zeros.
        self.specialToken = MLXArray.zeros([1, 1, config.hiddenSize])
        super.init()
    }

    /// Input: [B, T, poolWindowSize, D] → Output: [B, T, D] (CLS pooled).
    public func call(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        var h = embedTokens(x)
        // Prepend CLS token: [B, T, P, D] → [B, T, P+1, D]
        let cls = MLX.broadcast(specialToken, to: [B, T, 1, h.dim(3)])
        h = concatenated([cls, h], axis: 2)
        // Reshape to [(B*T), P+1, D]
        let P1 = h.dim(2)
        h = h.reshaped([B * T, P1, h.dim(3)])

        let (cos, sin) = rotaryEmb.call(seqLen: P1)
        for layer in layers {
            h = layer.call(hiddenStates: h, positionCosSin: (cos, sin), attentionMask: nil)
        }
        h = norm(h)
        // Extract CLS (first position): [(B*T), D]
        let clsOutput = h[0..<(B*T), 0...0, .ellipsis].squeezed(axis: 1)  // [(B*T), D]
        return clsOutput.reshaped([B, T, -1])  // [B, T, D]
    }
}

/// Complete audio tokenizer module.
/// Input: srcLatents [B, T, 64] → Output: (quantized [B, T/pool, D], indices [B, T/pool, codebookDim])
public class MLXAudioTokenizer: Module {
    public let audioAcousticProj: Linear
    public let attentionPooler: AttentionPooler
    public let quantizer: ResidualFSQ
    public let config: AudioTokenizerConfig

    public init(config: AudioTokenizerConfig = .turbo) {
        self.config = config
        self.audioAcousticProj = Linear(config.acousticHiddenDim, config.hiddenSize)
        self.attentionPooler = AttentionPooler(config: config)
        self.quantizer = ResidualFSQ(
            dim: config.fsqDim,
            levels: config.fsqInputLevels,
            numQuantizers: config.fsqInputNumQuantizers
        )
        super.init()
    }

    /// Tokenize source latents.
    /// Input: [B, T, 64] → Output: (quantized [B, T/pool, D], indices)
    public func tokenize(_ x: MLXArray, silenceLatent: MLXArray? = nil) -> (MLXArray, MLXArray) {
        var padded = x
        let T = x.dim(1)
        let P = config.poolWindowSize
        // Pad to multiple of pool_window_size (Python tokenize method)
        if T % P != 0 {
            let padLen = P - (T % P)
            if let sil = silenceLatent {
                let silPad = sil[0..<1, 0..<padLen, .ellipsis]
                let expanded = MLX.broadcast(silPad, to: [x.dim(0), padLen, x.dim(2)])
                padded = concatenated([x, expanded], axis: 1)
            } else {
                let zeros = MLXArray.zeros([x.dim(0), padLen, x.dim(2)])
                padded = concatenated([x, zeros], axis: 1)
            }
        }
        let newT = padded.dim(1) / P
        // Reshape: [B, T, D] → [B, T/P, P, D]
        let reshaped = padded.reshaped([padded.dim(0), newT, P, padded.dim(2)])
        // Project acoustic features to hidden size
        let projected = audioAcousticProj(reshaped)  // [B, T/P, P, hiddenSize]
        // Attention-pool: [B, T/P, P, D] → [B, T/P, D]
        let pooled = attentionPooler.call(projected)
        // Quantize: [B, T/P, D] → (quantized, indices)
        return quantizer.call(pooled)
    }

    /// Load weights from safetensors file.
    public func loadWeights(from url: URL) throws {
        let params = try loadDiTParameters(from: url)
        update(parameters: params)
        MLX.eval(parameters())
    }
}
