/**
 Load model parameters from safetensors into nested structure for Module.update(parameters:).
 Uses MLX.loadArrays(url:stream:) (from the MLX package) to read safetensors; the MLX dependency must be linked.
 */

import Foundation
import MLX
import MLXNN

/// Convert a single path segment from snake_case to camelCase (e.g. "proj_in" -> "projIn").
private func snakeToCamelSegment(_ segment: String) -> String {
    var result = ""
    var capitalizeNext = false
    for c in segment {
        if c == "_" {
            capitalizeNext = true
        } else {
            result.append(capitalizeNext ? c.uppercased() : String(c))
            capitalizeNext = false
        }
    }
    return result
}

/// Convert a parameter key from snake_case path to camelCase path so Python/MLX checkpoint keys match Swift module names.
/// E.g. "proj_in.weight" -> "projIn.weight", "layers.0.self_attn.q_proj.weight" -> "layers.0.selfAttn.qProj.weight".
public func diTParameterKeyToSwift(_ key: String) -> String {
    key.split(separator: ".").map { snakeToCamelSegment(String($0)) }.joined(separator: ".")
}

/// Normalize PyTorch Sequential keys so decoder uses bare Conv keys: "proj_in.1.*" -> "proj_in.*", "proj_out.1.*" -> "proj_out.*".
private func normalizeDiTKeyForPyTorchSequential(_ key: String) -> String {
    var k = key
    if k.contains("proj_in.1.") { k = k.replacingOccurrences(of: "proj_in.1.", with: "proj_in.") }
    if k.contains("proj_out.1.") { k = k.replacingOccurrences(of: "proj_out.1.", with: "proj_out.") }
    return k
}

/// Apply PyTorch -> MLX Conv weight layout so Swift NLC conv matches. PT Conv1d [out,in,K] -> MLX [out,K,in]. PT ConvTranspose1d [in,out,K] -> MLX [out,K,in]. Only applies when value is 3D (Conv/ConvTranspose1d); 2D weights (e.g. Linear) are unchanged.
private func convertDiTConvWeightIfNeeded(originalKey: String, value: MLXArray) -> MLXArray {
    guard value.ndim == 3, originalKey.hasSuffix(".weight") else { return value }
    if originalKey.contains("proj_in") {
        return value.transposed(axes: [0, 2, 1])
    }
    if originalKey.contains("proj_out") {
        return value.transposed(axes: [1, 2, 0])
    }
    return value
}

/// Load DiT decoder parameters from a safetensors file (PyTorch checkpoint format). Maps snake_case to camelCase, normalizes proj_in.1/proj_out.1 to proj_in/proj_out, and converts Conv/ConvTranspose1d weights from PT to MLX layout.
/// Use for decoder.update(parameters: try loadDiTParameters(from: url)).
public func loadDiTParameters(from url: URL) throws -> ModuleParameters {
    let flat = try loadArrays(url: url)
    let remapped: [String: MLXArray] = Dictionary(uniqueKeysWithValues: flat.map { entry in
        let key = entry.key
        let value = entry.value
        let normalized = normalizeDiTKeyForPyTorchSequential(key)
        let swiftKey = diTParameterKeyToSwift(normalized)
        let converted = convertDiTConvWeightIfNeeded(originalKey: key, value: value)
        return (swiftKey, converted)
    })
    return ModuleParameters.unflattened(remapped)
}

/// Prefix for full-model checkpoint decoder subtree (strip this to get decoder-only keys for update).
private let decoderKeyPrefix = "decoder."

/// Load DiT decoder parameters suitable for decoder.update(parameters:). If the checkpoint has a top-level "decoder" key (full model), flattens, strips the "decoder." prefix, and unflattens so the decoder module receives projIn.*, layers.*, etc. Otherwise returns the full parameters (decoder-only checkpoint).
public func loadDiTParametersForDecoder(from url: URL) throws -> ModuleParameters {
    let params = try loadDiTParameters(from: url)
    let flat = params.flattened()
    let decoderFlat: [(String, MLXArray)] = flat
        .filter { $0.0.hasPrefix(decoderKeyPrefix) }
        .map { (String($0.0.dropFirst(decoderKeyPrefix.count)), $0.1) }
    if decoderFlat.isEmpty {
        return params
    }
    return ModuleParameters.unflattened(decoderFlat)
}

/// Load flat [String: MLXArray] from a safetensors file and return nested ModuleParameters.
/// Use with `model.update(parameters: try loadParameters(from: url))`.
public func loadParameters(from url: URL) throws -> ModuleParameters {
    let flat = try loadArrays(url: url)
    return ModuleParameters.unflattened(flat)
}

/// Load silence latent for text2music from a safetensors file (key "latent", shape [1, T, 64]).
/// Export from Python: torch.save(latent, "silence_latent.pt") then use a one-off export to save as silence_latent.safetensors with key "latent".
/// Returns nil if file or key is missing.
public func loadSilenceLatent(from url: URL) throws -> MLXArray? {
    guard FileManager.default.fileExists(atPath: url.path) else { return nil }
    let flat = try loadArrays(url: url)
    guard let latent = flat["latent"] else { return nil }
    guard latent.ndim == 3, latent.dim(0) >= 1, latent.dim(2) == 64 else { return nil }
    return latent
}

/// Extract the DiT checkpoint's learned null condition embedding [1, 1, 2048] for CFG, if present.
/// Tries keys containing "null_condition_emb" (snake) or "nullConditionEmb" (camel after mapping).
public func extractNullConditionEmbedding(fromDitWeightsURL url: URL) throws -> MLXArray? {
    let flat = try loadArrays(url: url)
    for (key, value) in flat {
        if key.contains("null_condition_emb") || key.contains("nullConditionEmb") {
            return value
        }
    }
    return nil
}
