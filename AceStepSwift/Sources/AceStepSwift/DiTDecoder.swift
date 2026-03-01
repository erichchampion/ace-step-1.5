/**
 DiT diffusion decoder: patch in, timestep + encoder conditioning, N layers, patch out. 
 Mirrors dit_model.MLXDiTDecoder and Python AceStepDiTModel 
 (Python reference: acestep/models/base/modeling_acestep_v15_base.py).
 */

import Foundation
import MLX
import MLXNN

public class DiTDecoder: Module {
    public let hiddenSize: Int
    public let patchSize: Int
    public let projIn: Conv1d
    public let timeEmbed: DiTTimestepEmbedding
    public let timeEmbedR: DiTTimestepEmbedding
    public let conditionEmbedder: Linear
    public let layers: [DiTLayer]
    public let normOut: RMSNorm
    public let projOut: ConvTranspose1d
    public let scaleShiftTable: MLXArray
    public let rotaryEmb: DiTRotaryEmbedding
    private let slidingWindow: Int
    private let layerTypes: [String]
    private var slidingMasks: [Int: MLXArray] = [:]

    public init(
        hiddenSize: Int = 2048,
        intermediateSize: Int = 6144,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        rmsNormEps: Float = 1e-6,
        attentionBias: Bool = false,
        inChannels: Int = 192,
        audioAcousticHiddenDim: Int = 64,
        patchSize: Int = 2,
        slidingWindow: Int = 128,
        layerTypes: [String]? = nil,
        ropeTheta: Float = 1_000_000,
        maxPositionEmbeddings: Int = 32768
    ) {
        self.hiddenSize = hiddenSize
        self.patchSize = patchSize
        self.slidingWindow = slidingWindow
        let ltypes = layerTypes ?? (0..<numHiddenLayers).map { ($0 + 1) % 2 == 1 ? "sliding_attention" : "full_attention" }
        self.layerTypes = ltypes

        self.projIn = Conv1d(
            inputChannels: inChannels,
            outputChannels: hiddenSize,
            kernelSize: patchSize,
            stride: patchSize,
            padding: 0
        )
        self.timeEmbed = DiTTimestepEmbedding(inChannels: 256, timeEmbedDim: hiddenSize)
        self.timeEmbedR = DiTTimestepEmbedding(inChannels: 256, timeEmbedDim: hiddenSize)
        self.conditionEmbedder = Linear(hiddenSize, hiddenSize, bias: true)
        self.layers = (0..<numHiddenLayers).map { i in
            DiTLayer(
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                numAttentionHeads: numAttentionHeads,
                numKeyValueHeads: numKeyValueHeads,
                headDim: headDim,
                rmsNormEps: rmsNormEps,
                attentionBias: attentionBias,
                layerIdx: i,
                layerType: ltypes[i],
                slidingWindow: ltypes[i] == "sliding_attention" ? slidingWindow : nil
            )
        }
        self.normOut = RMSNorm(dimensions: hiddenSize, eps: rmsNormEps)
        self.projOut = ConvTranspose1d(
            inputChannels: hiddenSize,
            outputChannels: audioAcousticHiddenDim,
            kernelSize: patchSize,
            stride: patchSize,
            padding: 0
        )
        self.scaleShiftTable = MLXArray.zeros([1, 2, hiddenSize])
        self.rotaryEmb = DiTRotaryEmbedding(headDim: headDim, maxLen: maxPositionEmbeddings, base: ropeTheta)
        super.init()
    }

    func getSlidingMask(seqLen: Int) -> MLXArray {
        if let m = slidingMasks[seqLen] { return m }
        let m = diTCreateSlidingWindowMask(seqLen: seqLen, windowSize: slidingWindow)
        slidingMasks[seqLen] = m
        return m
    }

    /// hiddenStates [B, T, 64], timestep [B], timestepR [B], encoderHiddenStates [B, encL, D], contextLatents [B, T, C_ctx], optional encoderAttentionMask [B, encL] (1=valid, 0=pad). Returns [B, T, 64], cache.
    // Python reference: AceStepDiTModel.forward
    public func call(
        hiddenStates: MLXArray,
        timestep: MLXArray,
        timestepR: MLXArray,
        encoderHiddenStates: MLXArray,
        contextLatents: MLXArray,
        encoderAttentionMask: MLXArray? = nil,
        cache: DiTCrossAttentionCache? = nil,
        useCache: Bool = true
    ) -> (MLXArray, DiTCrossAttentionCache?) {
        // Compute timestep embeddings for diffusion conditioning
        // Python equivalent: temb_t, timestep_proj_t = self.time_embed(timestep)
        let (tembT, projT) = timeEmbed.call(timestep)
        let (tembR, projR) = timeEmbedR.call(timestep - timestepR)
        let temb = tembT + tembR
        let timestepProj = projT + projR

        // Concatenate context latents (source latents + chunk masks) with hidden states
        // Python equivalent: torch.cat([context_latents, hidden_states], dim=-1)
        var h = concatenated([contextLatents, hiddenStates], axis: 2)
        let originalSeqLen = h.dim(1)
        let padLength = (patchSize - (h.dim(1) % patchSize)) % patchSize
        if padLength > 0 {
            let padding = MLXArray.zeros([h.dim(0), padLength, h.dim(2)])
            h = concatenated([h, padding], axis: 1)
        }

        h = projIn(h)
        // Project encoder hidden states to model dimension
        // Python equivalent: condition_embedder(encoder_hidden_states)
        let encProj = conditionEmbedder(encoderHiddenStates)
        let seqLen = h.dim(1)
        let (cos, sin) = rotaryEmb.call(seqLen: seqLen)

        let slidingMask: MLXArray? = layerTypes.contains("sliding_attention") ? getSlidingMask(seqLen: seqLen) : nil

        for layer in layers {
            let selfMask = layer.layerType == "sliding_attention" ? slidingMask : nil
            h = layer.call(
                hiddenStates: h,
                positionCosSin: (cos, sin),
                temb: timestepProj,
                selfAttnMask: selfMask,
                encoderHiddenStates: encProj,
                encoderAttentionMask: encoderAttentionMask,
                cache: cache,
                useCache: useCache
            )
        }

        // Output norm: need shift/scale as [B, 1, D] to broadcast with h [B, L, D].
        // Python equivalent: scale_shift_table split + Adaptive RMSNorm
        let table: MLXArray = scaleShiftTable.dim(2) == 2
            ? scaleShiftTable.transposed(axes: [0, 2, 1])
            : scaleShiftTable
        let tembExp = temb.expandedDimensions(axis: 1)
        let combined = table + tembExp
        let parts = split(combined, parts: 2, axis: 1)
        let shift = parts[0]
        let scale = parts[1]
        h = normOut(h) * (1.0 + scale) + shift
        
        // Final projection back to audio acoustic hidden dim
        h = projOut(h)
        // Trim padding from axis 1. Only use 3D subscript when ndim == 3 to avoid MLX getItemND
        // out-of-bounds (starts[axis]) when run on concurrent queues with unexpected shapes.
        if h.ndim == 3, h.dim(1) > originalSeqLen {
            h = h[0..<h.dim(0), 0..<originalSeqLen, 0..<h.dim(2)]
        }
        return (h, cache)
    }
}
