/**
 Diffusion step contract for DiT: one step from current latent + conditions → next latent.
 Enables TDD with a fake stepper before a real DiT model is implemented.
 */

import Foundation
import MLX

/// Reference-type wrapper to break recursive struct cycles.
public final class Box<T> {
    public var value: T
    public init(_ value: T) { self.value = value }
}

/// Conditioning inputs for one DiT step (encoder hidden states and context latents).
/// For cover/repaint/lego: set initialLatents so the pipeline starts from src_latents instead of noise.
public struct DiTConditions {
    public var encoderHiddenStates: MLXArray?
    public var contextLatents: MLXArray?
    /// Encoder attention mask [B, encL] for cross-attention (1 = valid, 0 = pad). Optional; when nil no padding mask is applied.
    public var encoderAttentionMask: MLXArray?
    /// Null condition embedding [1, 1, D] used for CFG/APG unconditional branch.
    public var nullConditionEmbedding: MLXArray?
    /// When non-nil, pipeline uses this as the initial latent (e.g. encoded src_audio or repaint base). When nil, starts from noise.
    public var initialLatents: MLXArray?
    /// When non-nil with audioCoverStrength < 1.0, the pipeline switches to these conditions
    /// at step `cover_steps` to transition from cover to text2music mode mid-diffusion.
    /// Uses `Box` wrapper to avoid recursive struct (Swift value type limitation).
    public var nonCoverConditions: Box<DiTConditions>?

    public init(
        encoderHiddenStates: MLXArray? = nil,
        contextLatents: MLXArray? = nil,
        encoderAttentionMask: MLXArray? = nil,
        nullConditionEmbedding: MLXArray? = nil,
        initialLatents: MLXArray? = nil,
        nonCoverConditions: DiTConditions? = nil
    ) {
        self.encoderHiddenStates = encoderHiddenStates
        self.contextLatents = contextLatents
        self.encoderAttentionMask = encoderAttentionMask
        self.nullConditionEmbedding = nullConditionEmbedding
        self.initialLatents = initialLatents
        self.nonCoverConditions = nonCoverConditions.map { Box($0) }
    }
}

/// Protocol for a single diffusion step: current latent + timestep + conditions → next latent (same shape).
/// nextTimestep: when non-nil, ODE step uses dt = timestep - nextTimestep; when nil (e.g. last step), uses xt - vt * timestep.
public protocol DiffusionStepper: AnyObject {
    func predictVelocity(currentLatent: MLXArray, timestep: Float, conditions: DiTConditions, useCache: Bool) -> MLXArray
    func step(currentLatent: MLXArray, timestep: Float, conditions: DiTConditions, nextTimestep: Float?) -> MLXArray
}
