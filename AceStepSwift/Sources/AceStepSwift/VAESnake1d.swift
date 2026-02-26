/**
 Snake1d activation: x + (1/beta)*sin(alpha*x)^2. Mirrors vae_model.MLXSnake1d.
 Alpha/beta are [C]; when logscale=true use exp(alpha), exp(beta).
 */

import Foundation
import MLX
import MLXNN

public class VAESnake1d: Module, UnaryLayer {
    public let alpha: MLXArray
    public let beta: MLXArray
    private let logscale: Bool

    public init(hiddenDim: Int, logscale: Bool = true) {
        self.alpha = MLXArray.zeros([hiddenDim])
        self.beta = MLXArray.zeros([hiddenDim])
        self.logscale = logscale
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = logscale ? exp(alpha) : alpha
        let b = logscale ? exp(beta) : beta
        return x + (1 / (b + 1e-9)) * pow(sin(a * x), 2)
    }
}
