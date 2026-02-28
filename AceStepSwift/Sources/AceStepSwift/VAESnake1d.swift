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
        // Upcast to float32 for exp/sin/power to prevent overflow with float16
        // weights (exp overflows float16 at alpha > ~11). This matches Python
        // vae_model.py MLXSnake1d implementation.
        let aBase: MLXArray
        let bBase: MLXArray
        if logscale {
            aBase = exp(alpha.asType(.float32))
            bBase = exp(beta.asType(.float32))
        } else {
            aBase = alpha.asType(.float32)
            bBase = beta.asType(.float32)
        }
        let aCount = (0..<aBase.ndim).reduce(1) { $0 * aBase.dim($1) }
        let bCount = (0..<bBase.ndim).reduce(1) { $0 * bBase.dim($1) }
        let aFlat = aBase.reshaped([aCount])
        let bFlat = bBase.reshaped([bCount])

        // Make channel broadcasting explicit for 3D tensors.
        // NLC -> [1, 1, C], NCL -> [1, C, 1].
        let a: MLXArray
        let b: MLXArray
        if x.ndim == 3 && aFlat.dim(0) == x.dim(2) && bFlat.dim(0) == x.dim(2) {
            a = aFlat.reshaped([1, 1, aFlat.dim(0)])
            b = bFlat.reshaped([1, 1, bFlat.dim(0)])
        } else if x.ndim == 3 && aFlat.dim(0) == x.dim(1) && bFlat.dim(0) == x.dim(1) {
            a = aFlat.reshaped([1, aFlat.dim(0), 1])
            b = bFlat.reshaped([1, bFlat.dim(0), 1])
        } else {
            a = aFlat
            b = bFlat
        }

        // Compute in float32 for numerical stability, then cast back
        let xF32 = x.asType(.float32)
        let resultF32 = xF32 + (1 / (b + 1e-9)) * pow(sin(a * xF32), 2)
        return resultF32.asType(x.dtype)
    }
}
