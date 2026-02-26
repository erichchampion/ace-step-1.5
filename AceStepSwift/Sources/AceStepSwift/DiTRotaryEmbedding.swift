/**
 Pre-computes cos/sin tables for rotary position embeddings. Mirrors dit_model.MLXRotaryEmbedding.
 */

import Foundation
import MLX
import MLXNN

public class DiTRotaryEmbedding: Module {
    public let headDim: Int
    public let maxLen: Int
    private let cos: MLXArray
    private let sin: MLXArray

    public init(headDim: Int, maxLen: Int = 32768, base: Float = 1_000_000) {
        self.headDim = headDim
        self.maxLen = maxLen
        let invFreqFloats = stride(from: 0, to: headDim, by: 2).map { 1.0 / pow(base, Float($0) / Float(headDim)) }
        let invFreqArr = MLXArray(invFreqFloats)
        let positions = MLXArray((0..<maxLen).map { Float($0) })
        let freqsBroadcast = positions.expandedDimensions(axis: 1) * invFreqArr.expandedDimensions(axis: 0)
        let freqs = concatenated([freqsBroadcast, freqsBroadcast], axis: 1)
        self.cos = MLX.cos(freqs)
        self.sin = MLX.sin(freqs)
        super.init()
    }

    /// Returns (cos, sin) each [1, 1, seqLen, headDim].
    public func call(seqLen: Int) -> (MLXArray, MLXArray) {
        let cosSlice = cos[0..<seqLen, .ellipsis]
        let sinSlice = sin[0..<seqLen, .ellipsis]
        let c = cosSlice.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let s = sinSlice.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        return (c, s)
    }
}
