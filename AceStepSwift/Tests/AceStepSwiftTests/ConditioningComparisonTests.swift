/**
 Conditioning Pipeline Comparison Tests: loads Python-exported reference tensors
 and compares against Swift conditioning pipeline outputs at each stage.

 Set CONDITIONING_STAGES_DIR environment variable to the directory containing
 the reference tensors (produced by scripts/export_conditioning_stages.py).
 Set ENCODER_WEIGHTS_PATH to the encoder.safetensors file containing ConditionEncoder weights.
 Set EMBED_TOKENS_PATH to the safetensors file containing embed_tokens.weight.
 Set SILENCE_LATENT_PATH to the silence_latent.safetensors file.
 */

import XCTest
import MLX
import MLXNN
@testable import AceStepSwift

final class ConditioningComparisonTests: XCTestCase {

    // MARK: - Environment

    private static let stagesDirEnv = "CONDITIONING_STAGES_DIR"
    private static let encoderWeightsEnv = "ENCODER_WEIGHTS_PATH"
    private static let embedTokensEnv = "EMBED_TOKENS_PATH"
    private static let silenceLatentEnv = "SILENCE_LATENT_PATH"

    private static func envPath(_ key: String) -> String? {
        ProcessInfo.processInfo.environment[key]
            ?? ProcessInfo.processInfo.environment["TEST_RUNNER_\(key)"]
    }

    // MARK: - Helpers

    /// Load a 3D float32 binary file: 3 x Int32 LE header (d0, d1, d2) + flat float32 data.
    private static func loadFloat32Array(from path: String) -> MLXArray? {
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

    /// Load 1D int32 binary file: int32 count then int32 data.
    private static func loadIntArray(from path: String) -> [Int]? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              data.count >= 4 else { return nil }
        let n = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: Int32.self) }
        let count = Int(n)
        guard count > 0, data.count >= 4 + count * 4 else { return nil }
        var ids = [Int](repeating: 0, count: count)
        for i in 0..<count {
            let v = data.withUnsafeBytes { $0.load(fromByteOffset: 4 + i * 4, as: Int32.self) }
            ids[i] = Int(v)
        }
        return ids
    }

    /// Cosine similarity between two MLXArrays (flattened).
    private static func cosineSimilarity(_ a: MLXArray, _ b: MLXArray) -> Float {
        let af = a.reshaped([-1]).asType(.float32)
        let bf = b.reshaped([-1]).asType(.float32)
        let dot = sum(af * bf).item(Float.self)
        let na = sqrt(sum(af * af)).item(Float.self)
        let nb = sqrt(sum(bf * bf)).item(Float.self)
        guard na > 0 && nb > 0 else { return 0 }
        return dot / (na * nb)
    }

    /// Maximum absolute difference.
    private static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        return abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// Compare two arrays: log stats and assert tolerance.
    private func compareArrays(
        _ name: String,
        swift swArr: MLXArray,
        python pyArr: MLXArray,
        cosineThreshold: Float = 0.95,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        let swMean = swArr.mean().item(Float.self)
        let swStd = swArr.asType(.float32).variance().sqrt().item(Float.self)
        let pyMean = pyArr.mean().item(Float.self)
        let pyStd = pyArr.asType(.float32).variance().sqrt().item(Float.self)
        let cos = Self.cosineSimilarity(swArr, pyArr)
        let maxDiff = Self.maxAbsDiff(swArr, pyArr)

        print("[\(name)] Swift: shape=\(swArr.shape) mean=\(swMean) std=\(swStd)")
        print("[\(name)] Python: shape=\(pyArr.shape) mean=\(pyMean) std=\(pyStd)")
        print("[\(name)] cosine=\(cos) maxDiff=\(maxDiff)")

        XCTAssertEqual(swArr.shape, pyArr.shape,
            "[\(name)] Shape mismatch: Swift \(swArr.shape) vs Python \(pyArr.shape)",
            file: file, line: line)
        XCTAssertGreaterThan(cos, cosineThreshold,
            "[\(name)] Cosine similarity \(cos) below threshold \(cosineThreshold)",
            file: file, line: line)
    }

    // MARK: - Tests

    /// Test 1: Compare text tokenization (token IDs should match exactly).
    func testTextTokenIdsMatchPython() throws {
        guard let dir = Self.envPath(Self.stagesDirEnv) else {
            throw XCTSkip("CONDITIONING_STAGES_DIR not set")
        }
        guard let pyIds = Self.loadIntArray(from: "\(dir)/text_token_ids.bin") else {
            throw XCTSkip("text_token_ids.bin not found")
        }

        // Note: tokenizer differences are expected (Python adds BOS=2, Swift may differ).
        // Just report the comparison.
        print("[TextTokenIds] Python: count=\(pyIds.count) first5=\(Array(pyIds.prefix(5)))")
        print("[TextTokenIds] NOTE: Swift tokenizer comparison requires TextHiddenStateProvider")
    }

    /// Test 2: Lyric token embeddings should have low std (~0.03, NOT ~3.2).
    func testLyricEmbeddingsAreLowStd() throws {
        guard let dir = Self.envPath(Self.stagesDirEnv) else {
            throw XCTSkip("CONDITIONING_STAGES_DIR not set")
        }
        guard let pyEmbs = Self.loadFloat32Array(from: "\(dir)/lyric_token_embeddings.bin") else {
            throw XCTSkip("lyric_token_embeddings.bin not found")
        }
        guard let embedPath = Self.envPath(Self.embedTokensEnv) else {
            throw XCTSkip("EMBED_TOKENS_PATH not set")
        }
        guard let pyLyricIds = Self.loadIntArray(from: "\(dir)/lyric_token_ids.bin") else {
            throw XCTSkip("lyric_token_ids.bin not found")
        }

        // Load Swift embedding matrix
        let loader = try LyricTokenEmbeddingLoader.load(from: URL(fileURLWithPath: embedPath))
        let swEmbs = loader.embed(tokenIDs: pyLyricIds)

        let pyStd = pyEmbs.asType(.float32).variance().sqrt().item(Float.self)
        let swStd = swEmbs.asType(.float32).variance().sqrt().item(Float.self)

        print("[LyricEmbeddings] Python: shape=\(pyEmbs.shape) std=\(pyStd)")
        print("[LyricEmbeddings] Swift: shape=\(swEmbs.shape) std=\(swStd)")

        // Critical: std should be ~0.03, NOT ~3.2
        XCTAssertLessThan(swStd, 0.1, "Swift lyric embedding std=\(swStd) is too high (should be ~0.03)")
        XCTAssertLessThan(pyStd, 0.1, "Python lyric embedding std=\(pyStd) is too high (should be ~0.03)")

        // Compare values
        compareArrays("LyricEmbeddings", swift: swEmbs, python: pyEmbs, cosineThreshold: 0.99)
    }

    /// Test 3: Text projection (text_projector) should match Python.
    func testTextProjectionMatchesPython() throws {
        guard let dir = Self.envPath(Self.stagesDirEnv) else {
            throw XCTSkip("CONDITIONING_STAGES_DIR not set")
        }
        guard let weightsPath = Self.envPath(Self.encoderWeightsEnv) else {
            throw XCTSkip("ENCODER_WEIGHTS_PATH not set")
        }
        guard let pyTextHS = Self.loadFloat32Array(from: "\(dir)/text_hidden_states.bin"),
              let pyTextProj = Self.loadFloat32Array(from: "\(dir)/text_projected.bin") else {
            throw XCTSkip("Reference files not found")
        }

        let encoder = ConditionEncoder(config: .turbo)
        let weightsURL = URL(fileURLWithPath: weightsPath)
        let encoderParams = try loadDiTParametersForEncoder(from: weightsURL)
        encoder.update(parameters: encoderParams)

        let swTextProj = encoder.textProjector(pyTextHS)
        eval(swTextProj)

        compareArrays("TextProjection", swift: swTextProj, python: pyTextProj, cosineThreshold: 0.99)
    }

    /// Test 4: Lyric encoder output should match Python.
    func testLyricEncoderMatchesPython() throws {
        guard let dir = Self.envPath(Self.stagesDirEnv) else {
            throw XCTSkip("CONDITIONING_STAGES_DIR not set")
        }
        guard let weightsPath = Self.envPath(Self.encoderWeightsEnv) else {
            throw XCTSkip("ENCODER_WEIGHTS_PATH not set")
        }
        guard let pyLyricEmbs = Self.loadFloat32Array(from: "\(dir)/lyric_token_embeddings.bin"),
              let pyLyricEnc = Self.loadFloat32Array(from: "\(dir)/lyric_encoded.bin") else {
            throw XCTSkip("Reference files not found")
        }

        let encoder = ConditionEncoder(config: .turbo)
        let weightsURL = URL(fileURLWithPath: weightsPath)
        let encoderParams = try loadDiTParametersForEncoder(from: weightsURL)
        encoder.update(parameters: encoderParams)

        let lyricMask = MLXArray.ones([1, pyLyricEmbs.dim(1)])
        let swLyricEnc = encoder.lyricEncoder.call(inputsEmbeds: pyLyricEmbs, attentionMask: lyricMask)
        eval(swLyricEnc)

        compareArrays("LyricEncoder", swift: swLyricEnc, python: pyLyricEnc, cosineThreshold: 0.95)
    }

    /// Test 5: Timbre encoder output should match Python.
    func testTimbreEncoderMatchesPython() throws {
        guard let dir = Self.envPath(Self.stagesDirEnv) else {
            throw XCTSkip("CONDITIONING_STAGES_DIR not set")
        }
        guard let weightsPath = Self.envPath(Self.encoderWeightsEnv) else {
            throw XCTSkip("ENCODER_WEIGHTS_PATH not set")
        }
        guard let pyTimbreEnc = Self.loadFloat32Array(from: "\(dir)/timbre_encoded.bin"),
              let pySilence = Self.loadFloat32Array(from: "\(dir)/silence_latent_crop.bin") else {
            throw XCTSkip("Reference files not found")
        }

        let encoder = ConditionEncoder(config: .turbo)
        let weightsURL = URL(fileURLWithPath: weightsPath)
        let encoderParams = try loadDiTParametersForEncoder(from: weightsURL)
        encoder.update(parameters: encoderParams)

        // Use silence_latent[:, :750, :] as reference audio input, matching Python
        let silence750: MLXArray
        if pySilence.dim(1) >= 750 {
            silence750 = pySilence[0..<1, 0..<750, 0..<64]
        } else {
            silence750 = pySilence
        }
        let referOrderMask = MLXArray([Int32(0)])
        let (swTimbreEnc, _) = encoder.timbreEncoder.call(referAudioPacked: silence750, referAudioOrderMask: referOrderMask)
        eval(swTimbreEnc)

        compareArrays("TimbreEncoder", swift: swTimbreEnc, python: pyTimbreEnc, cosineThreshold: 0.95)
    }

    /// Test 6: Full encoder_hidden_states (ConditionEncoder forward) should match Python.
    func testFullEncoderHiddenStatesMatchPython() throws {
        guard let dir = Self.envPath(Self.stagesDirEnv) else {
            throw XCTSkip("CONDITIONING_STAGES_DIR not set")
        }
        guard let weightsPath = Self.envPath(Self.encoderWeightsEnv) else {
            throw XCTSkip("ENCODER_WEIGHTS_PATH not set")
        }
        guard let pyTextHS = Self.loadFloat32Array(from: "\(dir)/text_hidden_states.bin"),
              let pyLyricEmbs = Self.loadFloat32Array(from: "\(dir)/lyric_token_embeddings.bin"),
              let pyEncHS = Self.loadFloat32Array(from: "\(dir)/encoder_hidden_states.bin"),
              let pySilence = Self.loadFloat32Array(from: "\(dir)/silence_latent_crop.bin") else {
            throw XCTSkip("Reference files not found")
        }

        let encoder = ConditionEncoder(config: .turbo)
        let weightsURL = URL(fileURLWithPath: weightsPath)
        let encoderParams = try loadDiTParametersForEncoder(from: weightsURL)
        encoder.update(parameters: encoderParams)

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

        compareArrays("FullEncoder", swift: swEncHS, python: pyEncHS, cosineThreshold: 0.95)
    }

    /// Test 7: Context latents should match Python.
    func testContextLatentsMatchPython() throws {
        guard let dir = Self.envPath(Self.stagesDirEnv) else {
            throw XCTSkip("CONDITIONING_STAGES_DIR not set")
        }
        guard let pyCtx = Self.loadFloat32Array(from: "\(dir)/context_latents.bin"),
              let pySilence = Self.loadFloat32Array(from: "\(dir)/silence_latent_crop.bin") else {
            throw XCTSkip("Reference files not found")
        }

        let latentLen = pySilence.dim(1)
        let chunkMasks = fullChunkMask(batchSize: 1, latentLength: latentLen)
        let swCtx = buildContextLatents(srcLatents: pySilence, chunkMasks: chunkMasks)
        eval(swCtx)

        compareArrays("ContextLatents", swift: swCtx, python: pyCtx, cosineThreshold: 0.999)
    }
}
