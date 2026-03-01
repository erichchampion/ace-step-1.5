/**
 Adaptive Projected Guidance (APG) for diffusion. Mirrors acestep/models/mlx/dit_generate.py _mlx_apg_forward.
 Projection is along axis 1 (time/sequence dimension).
 */

import Foundation
import MLX

/// APG: combines conditional and unconditional predictions with guidance along the orthogonal component.
/// - Parameters:
///   - predCond: conditional prediction [B, T, C]
///   - predUncond: unconditional prediction [B, T, C]
///   - guidanceScale: CFG strength (>1)
///   - momentumState: optional dict with "running" key for momentum; mutated in place
///   - normThreshold: cap on diff norm along axis 1 (default 2.5)
/// - Returns: same shape as predCond
public func apgForward(
    predCond: MLXArray,
    predUncond: MLXArray,
    guidanceScale: Float,
    momentumState: inout [String: MLXArray]?,
    normThreshold: Float = 2.5
) -> MLXArray {
    let projAxis = 1
    var diff = predCond - predUncond
    if var state = momentumState {
        if let running = state["running"] {
            diff = diff + running
        }
        state["running"] = diff
        momentumState = state
    }
    if normThreshold > 0 {
        let diffSq = (diff * diff).sum(axis: projAxis, keepDims: true)
        let diffNorm = sqrt(diffSq)
        let scaleFactor = minimum(MLXArray(1.0 as Float), normThreshold / (diffNorm + 1e-8))
        diff = diff * scaleFactor
    }
    let condSq = (predCond * predCond).sum(axis: projAxis, keepDims: true)
    let v1 = predCond / (sqrt(condSq + 1e-8))
    let parallel = (diff * v1).sum(axis: projAxis, keepDims: true) * v1
    let orthogonal = diff - parallel
    return predCond + (guidanceScale - 1) * orthogonal
}

/// APG without momentum (convenience).
public func apgForward(
    predCond: MLXArray,
    predUncond: MLXArray,
    guidanceScale: Float,
    normThreshold: Float = 2.5
) -> MLXArray {
    var state: [String: MLXArray]? = nil
    return apgForward(
        predCond: predCond,
        predUncond: predUncond,
        guidanceScale: guidanceScale,
        momentumState: &state,
        normThreshold: normThreshold
    )
}
