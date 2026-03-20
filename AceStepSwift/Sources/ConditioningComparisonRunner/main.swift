/**
 Standalone conditioning comparison runner.
 Loads Python-exported reference tensors and compares against Swift conditioning pipeline.

 Usage:
   swift run ConditioningComparisonRunner /tmp/conditioning_stages \
     /path/to/encoder.safetensors /path/to/embed_tokens_model.safetensors
 */

import Foundation
import MLX
import MLXNN
import AceStepSwift

// MARK: - Helpers

func loadFloat32Array(from path: String) -> MLXArray? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
          data.count >= 12 else { return nil }
    let d0 = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: Int32.self) }
    let d1 = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: Int32.self) }
    let d2 = data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: Int32.self) }
    let s0 = Int(d0), s1 = Int(d1), s2 = Int(d2)
    let count = s0 * s1 * s2
    guard count > 0, data.count >= 12 + count * 4 else { return nil }
    var floats = [Float](repeating: 0, count: count)
    _ = floats.withUnsafeMutableBytes { dst in
        data.copyBytes(to: dst, from: 12..<(12 + count * 4))
    }
    return MLXArray(floats, [s0, s1, s2])
}

func loadIntArray(from path: String) -> [Int]? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
          data.count >= 4 else { return nil }
    let n = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: Int32.self) }
    let count = Int(n)
    guard count > 0, data.count >= 4 + count * 4 else { return nil }
    var ids = [Int](repeating: 0, count: count)
    for i in 0..<count {
        ids[i] = Int(data.withUnsafeBytes { $0.load(fromByteOffset: 4 + i * 4, as: Int32.self) })
    }
    return ids
}

func cosineSimilarity(_ a: MLXArray, _ b: MLXArray) -> Float {
    let af = a.reshaped([-1]).asType(.float32)
    let bf = b.reshaped([-1]).asType(.float32)
    let dot = sum(af * bf).item(Float.self)
    let na = sqrt(sum(af * af)).item(Float.self)
    let nb = sqrt(sum(bf * bf)).item(Float.self)
    guard na > 0 && nb > 0 else { return 0 }
    return dot / (na * nb)
}

func compare(_ name: String, swift sw: MLXArray, python py: MLXArray, threshold: Float = 0.95) -> Bool {
    let swMean = sw.mean().item(Float.self)
    let swStd = sw.asType(.float32).variance().sqrt().item(Float.self)
    let pyMean = py.mean().item(Float.self)
    let pyStd = py.asType(.float32).variance().sqrt().item(Float.self)
    let cos = cosineSimilarity(sw, py)
    let maxDiff = abs(sw.asType(.float32) - py.asType(.float32)).max().item(Float.self)

    let pass = sw.shape == py.shape && cos > threshold
    let icon = pass ? "PASS" : "FAIL"
    print("[\(icon)] \(name)")
    print("  Swift:  shape=\(sw.shape) mean=\(String(format: "%.6f", swMean)) std=\(String(format: "%.6f", swStd))")
    print("  Python: shape=\(py.shape) mean=\(String(format: "%.6f", pyMean)) std=\(String(format: "%.6f", pyStd))")
    print("  cosine=\(String(format: "%.6f", cos)) maxDiff=\(String(format: "%.6f", maxDiff))")
    if sw.shape != py.shape {
        print("  ERROR: Shape mismatch!")
    }
    if cos <= threshold {
        print("  ERROR: Cosine \(cos) below threshold \(threshold)")
    }
    return pass
}

// MARK: - Main

guard CommandLine.arguments.count >= 4 else {
    print("Usage: ConditioningComparisonRunner <stages_dir> <encoder_weights> <embed_tokens_model>")
    exit(1)
}

let stagesDir = CommandLine.arguments[1]
let encoderWeightsPath = CommandLine.arguments[2]
let embedTokensPath = CommandLine.arguments[3]

var passed = 0
var failed = 0

// ---- Test 1: Lyric Embeddings ----
print("\n" + String(repeating: "=", count: 60))
print("Test 1: Lyric Token Embeddings (std should be ~0.03)")
print(String(repeating: "=", count: 60))
if let pyEmbs = loadFloat32Array(from: "\(stagesDir)/lyric_token_embeddings.bin"),
   let pyIds = loadIntArray(from: "\(stagesDir)/lyric_token_ids.bin") {
    do {
        let loader = try LyricTokenEmbeddingLoader.load(from: URL(fileURLWithPath: embedTokensPath))
        let swEmbs = loader.embed(tokenIDs: pyIds)
        eval(swEmbs)
        if compare("LyricEmbeddings", swift: swEmbs, python: pyEmbs, threshold: 0.99) {
            passed += 1
        } else {
            failed += 1
        }
    } catch {
        print("[FAIL] LyricEmbeddings: \(error)")
        failed += 1
    }
} else {
    print("[SKIP] lyric_token_embeddings.bin or lyric_token_ids.bin not found")
}

// ---- Load ConditionEncoder weights ----
let encoder = ConditionEncoder(config: .turbo)
do {
    let encoderParams = try loadDiTParametersForEncoder(from: URL(fileURLWithPath: encoderWeightsPath))
    encoder.update(parameters: encoderParams)
    print("\n[OK] ConditionEncoder weights loaded from \(encoderWeightsPath)")
} catch {
    print("\n[FATAL] Failed to load encoder weights: \(error)")
    exit(1)
}

// ---- Test 2: Text Projection ----
print("\n" + String(repeating: "=", count: 60))
print("Test 2: Text Projection")
print(String(repeating: "=", count: 60))
if let pyTextHS = loadFloat32Array(from: "\(stagesDir)/text_hidden_states.bin"),
   let pyTextProj = loadFloat32Array(from: "\(stagesDir)/text_projected.bin") {
    let swTextProj = encoder.textProjector(pyTextHS)
    eval(swTextProj)
    if compare("TextProjection", swift: swTextProj, python: pyTextProj, threshold: 0.99) {
        passed += 1
    } else {
        failed += 1
    }
} else {
    print("[SKIP] text_hidden_states.bin or text_projected.bin not found")
}

// ---- Test 3: Lyric Encoder ----
print("\n" + String(repeating: "=", count: 60))
print("Test 3: Lyric Encoder")
print(String(repeating: "=", count: 60))
if let pyLyricEmbs = loadFloat32Array(from: "\(stagesDir)/lyric_token_embeddings.bin"),
   let pyLyricEnc = loadFloat32Array(from: "\(stagesDir)/lyric_encoded.bin") {
    let lyricMask = MLXArray.ones([1, pyLyricEmbs.dim(1)])
    let swLyricEnc = encoder.lyricEncoder.call(inputsEmbeds: pyLyricEmbs, attentionMask: lyricMask)
    eval(swLyricEnc)
    if compare("LyricEncoder", swift: swLyricEnc, python: pyLyricEnc, threshold: 0.95) {
        passed += 1
    } else {
        failed += 1
    }
} else {
    print("[SKIP] Reference files not found")
}

// ---- Test 4: Timbre Encoder ----
print("\n" + String(repeating: "=", count: 60))
print("Test 4: Timbre Encoder")
print(String(repeating: "=", count: 60))
if let pyTimbreEnc = loadFloat32Array(from: "\(stagesDir)/timbre_encoded.bin"),
   let pySilence = loadFloat32Array(from: "\(stagesDir)/silence_latent_crop.bin") {
    let silence750: MLXArray
    if pySilence.dim(1) >= 750 {
        silence750 = pySilence[0..<1, 0..<750, 0..<64]
    } else {
        silence750 = pySilence
    }
    let referOrderMask = MLXArray([Int32(0)])
    let (swTimbreEnc, _) = encoder.timbreEncoder.call(referAudioPacked: silence750, referAudioOrderMask: referOrderMask)
    eval(swTimbreEnc)
    if compare("TimbreEncoder", swift: swTimbreEnc, python: pyTimbreEnc, threshold: 0.95) {
        passed += 1
    } else {
        failed += 1
    }
} else {
    print("[SKIP] Reference files not found")
}

// ---- Test 5: Full Encoder Hidden States ----
print("\n" + String(repeating: "=", count: 60))
print("Test 5: Full Encoder Hidden States")
print(String(repeating: "=", count: 60))
if let pyTextHS = loadFloat32Array(from: "\(stagesDir)/text_hidden_states.bin"),
   let pyLyricEmbs = loadFloat32Array(from: "\(stagesDir)/lyric_token_embeddings.bin"),
   let pyEncHS = loadFloat32Array(from: "\(stagesDir)/encoder_hidden_states.bin"),
   let pySilence = loadFloat32Array(from: "\(stagesDir)/silence_latent_crop.bin") {
    let textMask = MLXArray.ones([1, pyTextHS.dim(1)])
    let lyricMask = MLXArray.ones([1, pyLyricEmbs.dim(1)])
    let silence750: MLXArray
    if pySilence.dim(1) >= 750 {
        silence750 = pySilence[0..<1, 0..<750, 0..<64]
    } else {
        silence750 = pySilence
    }
    let referOrderMask = MLXArray([Int32(0)])
    let (swEncHS, _) = encoder.call(
        textHiddenStates: pyTextHS,
        textAttentionMask: textMask,
        lyricHiddenStates: pyLyricEmbs,
        lyricAttentionMask: lyricMask,
        referAudioPacked: silence750,
        referAudioOrderMask: referOrderMask
    )
    eval(swEncHS)
    if compare("FullEncoder", swift: swEncHS, python: pyEncHS, threshold: 0.95) {
        passed += 1
    } else {
        failed += 1
    }
} else {
    print("[SKIP] Reference files not found")
}

// ---- Test 6: Context Latents ----
print("\n" + String(repeating: "=", count: 60))
print("Test 6: Context Latents")
print(String(repeating: "=", count: 60))
if let pyCtx = loadFloat32Array(from: "\(stagesDir)/context_latents.bin"),
   let pySilence = loadFloat32Array(from: "\(stagesDir)/silence_latent_crop.bin") {
    let latentLen = pySilence.dim(1)
    let chunkMasks = fullChunkMask(batchSize: 1, latentLength: latentLen)
    let swCtx = buildContextLatents(srcLatents: pySilence, chunkMasks: chunkMasks)
    eval(swCtx)
    if compare("ContextLatents", swift: swCtx, python: pyCtx, threshold: 0.999) {
        passed += 1
    } else {
        failed += 1
    }
} else {
    print("[SKIP] Reference files not found")
}

// ---- Summary ----
print("\n" + String(repeating: "=", count: 60))
print("SUMMARY: \(passed) passed, \(failed) failed")
print(String(repeating: "=", count: 60))
exit(failed > 0 ? 1 : 0)
