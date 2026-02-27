/**
 Rotary and mask helpers for DiT attention. Mirrors dit_model _rotate_half, _apply_rotary_pos_emb, _create_sliding_window_mask.
 */

import Foundation
import MLX

/// Rotate last dimension by splitting in half and swapping with negation. x: [..., head_dim].
func diTRotateHalf(_ x: MLXArray) -> MLXArray {
    let headDim = x.dim(x.ndim - 1)
    let half = headDim / 2
    let x1 = x[.ellipsis, 0..<half]
    let x2 = x[.ellipsis, half..<headDim]
    return concatenated([-x2, x1], axis: x.ndim - 1)
}

/// Apply RoPE to Q,K. q,k: [B, n_heads, L, head_dim]; cos, sin: [1, 1, L, head_dim].
func diTApplyRotaryPosEmb(q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
    let qEmbed = (q * cos) + (diTRotateHalf(q) * sin)
    let kEmbed = (k * cos) + (diTRotateHalf(k) * sin)
    return (qEmbed, kEmbed)
}

/// Bidirectional sliding-window mask: positions within window_size get 0, others -1e9. Returns [1, 1, seqLen, seqLen].
func diTCreateSlidingWindowMask(seqLen: Int, windowSize: Int) -> MLXArray {
    let indices = MLXArray((0..<seqLen).map { Float($0) })
    let diff = MLX.abs(indices.expandedDimensions(axis: 1) - indices.expandedDimensions(axis: 0))
    let zeros = MLXArray.zeros([seqLen, seqLen])
    let negInf = MLXArray([Float](repeating: -1e9, count: seqLen * seqLen), [seqLen, seqLen])
    let mask = MLX.where(diff .<= Float(windowSize), zeros, negInf)
    return mask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
}

/// Encoder padding mask from [B, L] attention mask (1=valid, 0=padding).
/// Returns additive mask [B, 1, L, L] where valid keys are 0 and padded keys are -1e9.
/// This mirrors the Python encoder mask behavior (key masking only).
func createEncoderPaddingMask(attentionMask: MLXArray) -> MLXArray {
    let b = attentionMask.dim(0)
    let l = attentionMask.dim(1)
    let keyMask = MLX.broadcast(
        attentionMask.expandedDimensions(axis: 1).expandedDimensions(axis: 1),
        to: [b, 1, l, l]
    )
    let zeros = MLXArray.zeros([b, 1, l, l])
    let negInf = MLXArray([Float](repeating: -1e9, count: b * l * l), [b, 1, l, l])
    return MLX.where(keyMask .> 0, zeros, negInf)
}
