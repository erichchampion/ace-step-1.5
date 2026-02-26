/**
 Load model parameters from safetensors into nested structure for Module.update(parameters:).
 */

import Foundation
import MLX
import MLXNN

/// Load flat [String: MLXArray] from a safetensors file and return nested ModuleParameters.
/// Use with `model.update(parameters: try loadParameters(from: url))`.
public func loadParameters(from url: URL) throws -> ModuleParameters {
    let flat = try loadArrays(url: url)
    return ModuleParameters.unflattened(flat)
}
