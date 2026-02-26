/**
 SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x)). Mirrors dit_model.MLXSwiGLUMLP.
 */

import Foundation
import MLX
import MLXNN

public class DiTSwiGLUMLP: Module {
    public let gateProj: Linear
    public let upProj: Linear
    public let downProj: Linear

    public init(hiddenSize: Int, intermediateSize: Int) {
        self.gateProj = Linear(hiddenSize, intermediateSize, bias: false)
        self.upProj = Linear(hiddenSize, intermediateSize, bias: false)
        self.downProj = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}
