/**
 SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x)). Mirrors dit_model.MLXSwiGLUMLP.
 Uses inline SiLU (gate * sigmoid(gate)) to avoid MLXNN's compiled silu path, which can
 crash when the MLX compile closure is invoked with empty tracers on concurrent queues.
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
        let gate = gateProj(x)
        let siluGate = gate * MLX.sigmoid(gate)
        return downProj(siluGate * upProj(x))
    }
}
