import XCTest
import MLX
@testable import AceStepSwift

final class CoreMLGenerationTests: XCTestCase {

    private static let outputDirEnv = "OUTPUT_DIR"
    
    // Configurable generation parameters from environment
    private static var smokeDuration: Double {
        let env = ProcessInfo.processInfo.environment["SMOKE_DURATION"]
            ?? ProcessInfo.processInfo.environment["TEST_RUNNER_SMOKE_DURATION"]
        return Double(env ?? "") ?? 19.53125
    }
    private static var smokeSteps: Int {
        let env = ProcessInfo.processInfo.environment["SMOKE_STEPS"]
            ?? ProcessInfo.processInfo.environment["TEST_RUNNER_SMOKE_STEPS"]
        return Int(env ?? "") ?? 8
    }
    private static var smokeSeed: Int {
        let env = ProcessInfo.processInfo.environment["SMOKE_SEED"]
            ?? ProcessInfo.processInfo.environment["TEST_RUNNER_SMOKE_SEED"]
        return Int(env ?? "") ?? 42
    }

    /// Load a flat float32 binary file with 3 x Int32 header (d0, d1, d2). Returns MLXArray with shape [d0,d1,d2] or nil.
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

    private static func loadPrecomputedConditioning(latentLength: Int) -> DiTConditions? {
        let pathEnv = ProcessInfo.processInfo.environment["CONDITIONING_DIR"]
            ?? ProcessInfo.processInfo.environment["TEST_RUNNER_CONDITIONING_DIR"]
        guard let dir = pathEnv?.trimmingCharacters(in: .whitespaces), !dir.isEmpty else { return nil }
        let expanded = (dir as NSString).expandingTildeInPath
        let base = URL(fileURLWithPath: expanded)
        let encoderURL = base.appendingPathComponent("encoder_hidden_states.bin")
        let contextURL = base.appendingPathComponent("context_latents.bin")
        guard FileManager.default.fileExists(atPath: encoderURL.path),
              FileManager.default.fileExists(atPath: contextURL.path) else { return nil }
        guard let enc = loadFloat32Array(from: encoderURL.path),
              enc.ndim == 3, enc.dim(0) == 1, enc.dim(2) == 2048 else { return nil }
        guard let ctx = loadFloat32Array(from: contextURL.path),
              ctx.ndim == 3, ctx.dim(0) == 1, ctx.dim(1) >= latentLength, ctx.dim(2) == 128 else { return nil }
        let contextSlice = ctx.dim(1) == latentLength
            ? ctx
            : ctx[0..<1, 0..<latentLength, 0..<128]
        var conditions = DiTConditions(encoderHiddenStates: enc, contextLatents: contextSlice)
        
        let nullURL = base.appendingPathComponent("null_condition_embedding.bin")
        if FileManager.default.fileExists(atPath: nullURL.path),
           let nullCond = loadFloat32Array(from: nullURL.path),
           nullCond.ndim == 3, nullCond.dim(0) == 1, nullCond.dim(1) == 1 {
            conditions.nullConditionEmbedding = nullCond
            print("[SmokeTest] Loaded null_condition_embedding.bin: shape=\(nullCond.shape)")
        }

        let noiseURL = base.appendingPathComponent("initial_noise.bin")
        if FileManager.default.fileExists(atPath: noiseURL.path),
           let noise = loadFloat32Array(from: noiseURL.path),
           noise.ndim == 3, noise.dim(0) == 1, noise.dim(1) == latentLength, noise.dim(2) == 64 {
            conditions.initialLatents = noise
        }
        return conditions
    }

    private static func makeConditioningProvider() -> ConditioningProvider {
        { params, latentLength, _ in
            if let conditions = Self.loadPrecomputedConditioning(latentLength: latentLength) {
                return conditions
            }
            let encL = 8
            let ctxLatents = MLXArray.zeros([1, latentLength, 128])
            return DiTConditions(
                encoderHiddenStates: MLXArray.zeros([1, encL, 2048]),
                contextLatents: ctxLatents
            )
        }
    }

    private func writeWAV(samples: [Float], sampleRate: Int, channels: Int, to url: URL) throws {
        let targetPeakLinear: Float = 0.8912509
        var normalized = samples
        
        if channels == 1 {
            let mean = samples.reduce(0, +) / Float(samples.count)
            normalized = samples.map { $0 - mean }
            let peak = normalized.map { abs($0) }.max() ?? 0
            if peak > 1e-6 {
                let gain = targetPeakLinear / peak
                normalized = normalized.map { $0 * gain }
            }
        } else if channels == 2 {
            var left: [Float] = []
            var right: [Float] = []
            for i in stride(from: 0, to: samples.count - 1, by: 2) {
                left.append(samples[i])
                right.append(samples[i + 1])
            }
            let lMean = left.reduce(0, +) / Float(left.count)
            let rMean = right.reduce(0, +) / Float(right.count)
            for i in stride(from: 0, to: samples.count - 1, by: 2) {
                normalized[i] -= lMean
                normalized[i + 1] -= rMean
            }
            let peak = normalized.map { abs($0) }.max() ?? 0
            if peak > 1e-6 {
                let gain = targetPeakLinear / peak
                normalized = normalized.map { $0 * gain }
            }
        }

        let count = normalized.count
        var data = Data(capacity: 44 + count * 2)
        func appendU8(_ v: UInt8) { data.append(v) }
        func appendU16(_ v: UInt16) {
            data.append(UInt8(v & 0xff))
            data.append(UInt8((v >> 8) & 0xff))
        }
        func appendU32(_ v: UInt32) {
            data.append(UInt8(v & 0xff))
            data.append(UInt8((v >> 8) & 0xff))
            data.append(UInt8((v >> 16) & 0xff))
            data.append(UInt8((v >> 24) & 0xff))
        }
        let numBytes = count * 2
        let riffSize = 4 + 24 + 8 + numBytes
        data.append(contentsOf: [0x52, 0x49, 0x46, 0x46]) // "RIFF"
        appendU32(UInt32(riffSize))
        data.append(contentsOf: [0x57, 0x41, 0x56, 0x45]) // "WAVE"
        data.append(contentsOf: [0x66, 0x6d, 0x74, 0x20]) // "fmt "
        appendU32(16)
        appendU16(1) // PCM
        appendU16(UInt16(channels))
        appendU32(UInt32(sampleRate))
        appendU32(UInt32(sampleRate * channels * 2))
        appendU16(UInt16(channels * 2))
        appendU16(16)
        data.append(contentsOf: [0x64, 0x61, 0x74, 0x61]) // "data"
        appendU32(UInt32(numBytes))
        for s in normalized {
            let clamped = max(-1.0, min(1.0, s))
            let v = Int16(clamped * 32767)
            data.append(UInt8(v & 0xff))
            data.append(UInt8((v >> 8) & 0xff))
        }
        try data.write(to: url)
    }

    private func runCoreMLGeneration(envVar: String, outputFilename: String) async throws {
        let pathEnv = ProcessInfo.processInfo.environment[envVar]
            ?? ProcessInfo.processInfo.environment["TEST_RUNNER_\(envVar)"]
        
        guard let basePath = pathEnv?.trimmingCharacters(in: .whitespaces), !basePath.isEmpty else {
            try XCTSkipIf(true, "Set \(envVar) to run Core ML generation smoke test")
            return
        }
        
        guard let outDir = ProcessInfo.processInfo.environment[Self.outputDirEnv]?.trimmingCharacters(in: .whitespaces),
              !outDir.isEmpty else {
            try XCTSkipIf(true, "Set \(Self.outputDirEnv) to run Core ML generation smoke test and write output")
            return
        }
        
        let expanded = (basePath as NSString).expandingTildeInPath
        let baseURL = URL(fileURLWithPath: expanded)
        
        // Find DiT .mlpackage and VAE .mlpackage in the specified directory
        // Determine bit depth to search for
        let is16Bit = envVar.contains("16BIT")
        let is8Bit = envVar.contains("8BIT")
        let is6Bit = envVar.contains("6BIT")
        let is4Bit = envVar.contains("4BIT")
        let bitSuffix = is16Bit ? "-16bit" : (is8Bit ? "-8bit" : (is6Bit ? "-6bit" : (is4Bit ? "-4bit" : "")))
        
        // Find DiT .mlpackage and VAE .mlpackage in the specified directory
        let ditURL = try findMLPackage(in: baseURL, prefix: "acestep-v15-turbo", suffix: bitSuffix)
        let vaeURL = try findMLPackage(in: baseURL, prefix: "vae", suffix: bitSuffix)
        
        guard let ditURL = ditURL, let vaeURL = vaeURL else {
            XCTFail("Could not find .mlpackage files in \(basePath)")
            return
        }
        
        print("[CoreML SmokeTest] Booting CoreMLDiTStepper from \(ditURL.lastPathComponent)")
        let stepper = try await CoreMLDiTStepper(modelURL: ditURL)
        
        print("[CoreML SmokeTest] Booting CoreMLVAEDecoder from \(vaeURL.lastPathComponent)")
        let vaeDecoder = try await CoreMLVAEDecoder(modelURL: vaeURL)
        
        let conditioningProvider = Self.makeConditioningProvider()
        let pipeline = ContractGenerationPipeline(
            stepper: stepper,
            decoder: vaeDecoder,
            sampleRate: AceStepConstants.defaultSampleRate,
            conditioningProvider: conditioningProvider
        )
        
        let duration = Self.smokeDuration
        let steps = Self.smokeSteps
        let seed = Self.smokeSeed
        print("[CoreML SmokeTest] Generating with duration=\(duration)s, steps=\(steps), seed=\(seed)")
        
        let result = AceStepEngine.generateMusic(
            params: GenerationParams(
                caption: "Core ML Checkpoint Test",
                lyrics: "[Instrumental]",
                duration: duration,
                inferenceSteps: steps,
                seed: seed,
                shift: 3.0
            ),
            config: GenerationConfig(batchSize: 1),
            progress: nil,
            pipeline: pipeline
        )
        
        XCTAssertTrue(result.success, result.error ?? "no error")
        guard let first = result.audios.first,
              let tensor = first["tensor"] as? [Float],
              let sampleRate = first["sample_rate"] as? Int,
              let channels = first["channels"] as? Int else {
            XCTFail("Missing audios[0] tensor/sample_rate/channels")
            return
        }
        let dirURL = URL(fileURLWithPath: outDir)
        let wavURL = dirURL.appendingPathComponent(outputFilename)
        try writeWAV(samples: tensor, sampleRate: sampleRate, channels: channels, to: wavURL)
        print("[CoreML SmokeTest] Successfully wrote \(outputFilename)")
    }
    
    private func findMLPackage(in directory: URL, prefix: String, suffix: String) throws -> URL? {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        return files.first { 
            $0.pathExtension == "mlpackage" && 
            $0.lastPathComponent.hasPrefix(prefix) && 
            $0.lastPathComponent.contains(suffix) &&
            !$0.lastPathComponent.contains("-lm-")
        }
    }

    func test4BitGeneration() async throws {
        try await runCoreMLGeneration(envVar: "COREML_4BIT_PATH", outputFilename: "swift_4bit_out.wav")
    }

    func test6BitGeneration() async throws {
        try await runCoreMLGeneration(envVar: "COREML_6BIT_PATH", outputFilename: "swift_6bit_out.wav")
    }

    func test16BitGeneration() async throws {
        try await runCoreMLGeneration(envVar: "COREML_16BIT_PATH", outputFilename: "swift_16bit_out.wav")
    }

    func test8BitGeneration() async throws {
        try await runCoreMLGeneration(envVar: "COREML_8BIT_PATH", outputFilename: "swift_8bit_out.wav")
    }
}
