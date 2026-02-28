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
        // Match Python vae_model.py MLXSnake1d: only upcast to float32 when weights are float16
        // to prevent exp overflow. For float32 weights, skip the upcast for performance.
        let aBase: MLXArray
        let bBase: MLXArray
        let inputIsFloat16 = x.dtype == .float16
        if logscale {
            aBase = inputIsFloat16 ? exp(alpha.asType(.float32)) : exp(alpha)
            bBase = inputIsFloat16 ? exp(beta.asType(.float32)) : exp(beta)
        } else {
            aBase = inputIsFloat16 ? alpha.asType(.float32) : alpha
            bBase = inputIsFloat16 ? beta.asType(.float32) : beta
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

        // Compute result - only upcast x to float32 for numerical stability when input is float16
        // to match Python vae_model.py MLXSnake1d main path
        let result: MLXArray
        if inputIsFloat16 {
            let xF32 = x.asType(.float32)
            result = xF32 + (1 / (b + 1e-9)) * pow(sin(a * xF32), 2)
            return result.asType(x.dtype)
        } else {
            result = x + (1 / (b + 1e-9)) * pow(sin(a * x), 2)
            return result
        }
    }
}
