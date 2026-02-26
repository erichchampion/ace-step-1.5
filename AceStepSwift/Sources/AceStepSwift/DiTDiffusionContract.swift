/**
 Diffusion step contract for DiT: one step from current latent + conditions → next latent.
 Enables TDD with a fake stepper before a real DiT model is implemented.
 */

import Foundation
import MLX

/// Conditioning inputs for one DiT step (encoder hidden states and context latents).
/// For cover/repaint/lego: set initialLatents so the pipeline starts from src_latents instead of noise.
public struct DiTConditions {
    public var encoderHiddenStates: MLXArray?
    public var contextLatents: MLXArray?
    /// When non-nil, pipeline uses this as the initial latent (e.g. encoded src_audio or repaint base). When nil, starts from noise.
    public var initialLatents: MLXArray?

    public init(
        encoderHiddenStates: MLXArray? = nil,
        contextLatents: MLXArray? = nil,
        initialLatents: MLXArray? = nil
    ) {
        self.encoderHiddenStates = encoderHiddenStates
        self.contextLatents = contextLatents
        self.initialLatents = initialLatents
    }
}

/// Protocol for a single diffusion step: current latent + timestep + conditions → next latent (same shape).
/// nextTimestep: when non-nil, ODE step uses dt = timestep - nextTimestep; when nil (e.g. last step), uses xt - vt * timestep.
public protocol DiffusionStepper: AnyObject {
    func step(currentLatent: MLXArray, timestep: Float, conditions: DiTConditions, nextTimestep: Float?) -> MLXArray
}
