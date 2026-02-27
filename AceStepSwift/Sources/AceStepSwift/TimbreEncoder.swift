/**
 Timbre encoder: embed -> 4 encoder layers -> norm -> take position 0 -> unpack to batch. Mirrors Python AceStepTimbreEncoder.
 Load encoder.timbre_encoder.* from checkpoint.
 */

import Foundation
import MLX
import MLXNN

public class TimbreEncoder: Module {
    public let embedTokens: Linear
    public let layers: [ConditionEncoderLayer]
    public let norm: RMSNorm
    public let rotaryEmb: DiTRotaryEmbedding
    private let slidingWindow: Int

    public init(
        timbreHiddenDim: Int = 64,
        hiddenSize: Int = 2048,
        intermediateSize: Int = 6144,
        numLayers: Int = 4,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        rmsNormEps: Float = 1e-6,
        attentionBias: Bool = false,
        slidingWindow: Int = 128
    ) {
        self.slidingWindow = slidingWindow
        self.embedTokens = Linear(timbreHiddenDim, hiddenSize)
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

    /// refer_audio_packed [N, T, timbre_hidden_dim], refer_audio_order_mask [N] (values 0..<B-1).
    /// Returns (timbre_embs_unpack [B, max_count, hidden_size], timbre_embs_mask [B, max_count]).
    public func call(
        referAudioPacked: MLXArray,
        referAudioOrderMask: MLXArray
    ) -> (MLXArray, MLXArray) {
        var h = embedTokens(referAudioPacked)
        guard h.ndim >= 3 else {
            return (MLXArray.zeros([1, 0, 2048]), MLXArray.zeros([1, 0]))
        }
        let seqLen = h.dim(1)
        let (cos, sin) = rotaryEmb.call(seqLen: seqLen)
        for layer in layers {
            let layerMask: MLXArray? = layer.layerType == "sliding_attention"
                ? diTCreateSlidingWindowMask(seqLen: seqLen, windowSize: slidingWindow)
                : nil
            h = layer.call(hiddenStates: h, positionCosSin: (cos, sin), attentionMask: layerMask)
        }
        h = norm(h)
        let packed = h[0..<h.dim(0), 0..<1, 0..<h.dim(2)].squeezed(axis: 1)
        return unpackTimbreEmbeddings(timbreEmbsPacked: packed, referAudioOrderMask: referAudioOrderMask)
    }
}

/// Unpack [N, D] with order mask [N] (values 0..<B-1) to [B, max_count, D] and mask [B, max_count].
/// Returns (zeros([1, 0, 2048]), zeros([1, 0])) when input is rank < 2 to avoid SmallVector out of range in MLX.
func unpackTimbreEmbeddings(timbreEmbsPacked: MLXArray, referAudioOrderMask: MLXArray) -> (MLXArray, MLXArray) {
    guard timbreEmbsPacked.ndim >= 2 else {
        return (MLXArray.zeros([1, 0, 2048]), MLXArray.zeros([1, 0]))
    }
    let n = timbreEmbsPacked.dim(0)
    let d = timbreEmbsPacked.dim(1)
    referAudioOrderMask.eval()
    timbreEmbsPacked.eval()
    let maskInt = referAudioOrderMask.asType(.int32)
    var counts = [Int: Int]()
    for i in 0..<n {
        let slice = maskInt[i..<(i + 1)]
        let v = Int(slice.item(Int32.self))
        counts[v, default: 0] += 1
    }
    let B = (counts.keys.max() ?? -1) + 1
    let countList = (0..<B).map { counts[$0] ?? 0 }
    let maxCount = countList.max() ?? 0
    guard B > 0, maxCount > 0 else {
        return (MLXArray.zeros([1, 0, d]), MLXArray.zeros([1, 0]))
    }
    var batchStarts = [Int](repeating: 0, count: B)
    for b in 1..<B { batchStarts[b] = batchStarts[b - 1] + countList[b - 1] }
    var order = [Int](repeating: 0, count: n)
    var pos = [Int](repeating: 0, count: B)
    for i in 0..<n {
        let slice = maskInt[i..<(i + 1)]
        let b = Int(slice.item(Int32.self))
        order[batchStarts[b] + pos[b]] = i
        pos[b] += 1
    }
    var outF = [[Float]](repeating: [Float](repeating: 0, count: d), count: B * maxCount)
    for b in 0..<B {
        for p in 0..<maxCount where p < countList[b] {
            let src = order[batchStarts[b] + p]
            let row = timbreEmbsPacked[src..<(src + 1), 0..<d].squeezed(axis: 0)
            row.eval()
            for c in 0..<d {
                let cell = row[c..<(c + 1)]
                outF[b * maxCount + p][c] = cell.item(Float.self)
            }
        }
    }
    let validMask: [Float] = (0..<(B * maxCount)).map { idx in
        let b = idx / maxCount
        let p = idx % maxCount
        return p < countList[b] ? 1 : 0
    }
    let outArr = MLXArray(outF.flatMap { $0 }, [B, maxCount, d])
    let maskArr = MLXArray(validMask, [B, maxCount])
    return (outArr, maskArr)
}
