/**
 Fake diffusion stepper that returns an array of the same shape as currentLatent.
 Used to satisfy the DiffusionStepper contract in tests without a real DiT model.
 */

import Foundation
import MLX

/// Fake stepper that returns zeros with the same shape as currentLatent.
public final class FakeDiffusionStepper: DiffusionStepper {
    public init() {}

    public func step(currentLatent: MLXArray, timestep: Float, conditions: DiTConditions, nextTimestep: Float?) -> MLXArray {
        let shape = currentLatent.shape
        let count = shape.reduce(1, *)
        return MLXArray([Float](repeating: 0, count: count), shape)
    }
}
