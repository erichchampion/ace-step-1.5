/**
 Diffusion step contract for DiT: one step from current latent + conditions → next latent.
 Enables TDD with a fake stepper before a real DiT model is implemented.
 */

import Foundation
import MLX

/// Conditioning inputs for one DiT step (encoder hidden states and context latents).
public struct DiTConditions {
    public var encoderHiddenStates: MLXArray?
    public var contextLatents: MLXArray?

    public init(encoderHiddenStates: MLXArray? = nil, contextLatents: MLXArray? = nil) {
        self.encoderHiddenStates = encoderHiddenStates
        self.contextLatents = contextLatents
    }
}

/// Protocol for a single diffusion step: current latent + timestep + conditions → next latent (same shape).
/// nextTimestep: when non-nil, ODE step uses dt = timestep - nextTimestep; when nil (e.g. last step), uses xt - vt * timestep.
public protocol DiffusionStepper: AnyObject {
    func step(currentLatent: MLXArray, timestep: Float, conditions: DiTConditions, nextTimestep: Float?) -> MLXArray
}
