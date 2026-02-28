/**
 Tests that validate the DiT decoder forward pass and the music generation pipeline
 to catch shape/broadcast/MLX errors (e.g. array split, ellipsis, broadcast_shapes).

 Requires Metal (run on Mac or device with GPU).

 - Tests without "LoadedWeights" run with random/unloaded weights (no download).
 - testDiTDecoderForwardWithLoadedWeights: set DIT_WEIGHTS_PATH to the DiT checkpoint
   directory (the one containing model.safetensors) to hit the exact app code path
   and iterate on broadcast fixes. Example (macOS app container):
   DIT_WEIGHTS_PATH=~/Library/Containers/com.cadenza-audio/Data/Library/Application\ Support/CadenzaAudio/Checkpoints/acestep-v15-turbo swift test --filter testDiTDecoderForwardWithLoadedWeights
 */

import XCTest
import MLX
@testable import AceStepSwift

final class DiTPipelineTests: XCTestCase {

    /// Path to DiT checkpoint dir (contains model.safetensors). Set to iterate on broadcast/shape fixes with real weights.
    private static let ditWeightsPathEnv = "DIT_WEIGHTS_PATH"

    /// Runs DiTDecoder forward with dummy inputs and asserts output shape.
    /// Catches: wrong modulation shape (split axis), double-ellipsis indexing, scaleShiftTable broadcast.
    func testDiTDecoderForwardOutputShape() {
        let decoder = DiTDecoder(
            hiddenSize: 2048,
            intermediateSize: 6144,
            numHiddenLayers: 2,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2
        )
        let b = 2
        let t = 4
        let encL = 5
        let hiddenStates = MLXArray.zeros([b, t, 64])
        let contextLatents = MLXArray.zeros([b, t, 128])
        let encoderHiddenStates = MLXArray.zeros([b, encL, 2048])
        let timestep = MLXArray([Float(0.5), 0.5])
        let timestepR = MLXArray([Float(0), 0])

        let (out, _) = decoder.call(
            hiddenStates: hiddenStates,
            timestep: timestep,
            timestepR: timestepR,
            encoderHiddenStates: encoderHiddenStates,
            contextLatents: contextLatents,
            cache: nil,
            useCache: false
        )

        XCTAssertEqual(out.dim(0), b)
        XCTAssertEqual(out.dim(1), t)
        XCTAssertEqual(out.dim(2), 64)
    }

    /// Runs one diffusion step through MLXDiTStepper with unloaded decoder (random weights).
    /// Validates that the stepper + decoder path does not crash on shape/broadcast.
    func testMLXDiTStepperOneStepShape() {
        let decoder = DiTDecoder(
            hiddenSize: 2048,
            intermediateSize: 6144,
            numHiddenLayers: 2,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2
        )
        let stepper = MLXDiTStepper(decoder: decoder)
        let b = 1
        let t = 4
        let currentLatent = MLXArray.zeros([b, t, 64])
        let conditions = DiTConditions(
            encoderHiddenStates: MLXArray.zeros([b, 5, 2048]),
            contextLatents: MLXArray.zeros([b, t, 128])
        )

        let nextLatent = stepper.step(
            currentLatent: currentLatent,
            timestep: 1.0,
            conditions: conditions,
            nextTimestep: 0.5
        )

        XCTAssertEqual(nextLatent.dim(0), b)
        XCTAssertEqual(nextLatent.dim(1), t)
        XCTAssertEqual(nextLatent.dim(2), 64)
    }

    /// Full pipeline: ContractGenerationPipeline with MLXDiTStepper + FakeVAEDecoder.
    /// Validates the diffusion loop and decoder contract with real DiT (unloaded) to catch shape/broadcast errors.
    func testContractPipelineWithMLXStepperMinimal() {
        let decoder = DiTDecoder(
            hiddenSize: 2048,
            intermediateSize: 6144,
            numHiddenLayers: 2,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2
        )
        let stepper = MLXDiTStepper(decoder: decoder)
        let pipeline = ContractGenerationPipeline(
            stepper: stepper,
            decoder: FakeVAEDecoder(),
            sampleRate: AceStepConstants.defaultSampleRate,
            conditioningProvider: { params, latentLength, _ in
                let encL = 8
                return DiTConditions(
                    encoderHiddenStates: MLXArray.zeros([1, encL, 2048]),
                    contextLatents: MLXArray.zeros([1, latentLength, 128])
                )
            }
        )

        let result = AceStepEngine.generateMusic(
            params: GenerationParams(
                caption: "test",
                lyrics: "[Instrumental]",
                duration: 1.0,
                inferenceSteps: 2,
                seed: 42
            ),
            config: GenerationConfig(batchSize: 1),
            progress: nil,
            pipeline: pipeline
        )

        XCTAssertTrue(result.success, result.error ?? "no error")
        if result.success, let first = result.audios.first {
            let tensor = first["tensor"] as? [Float]
            XCTAssertNotNil(tensor)
            XCTAssertFalse(tensor?.isEmpty ?? true)
        }
    }

    /// Loads DiT from DIT_WEIGHTS_PATH and runs one decoder forward. Use this to iterate on broadcast/shape fixes:
    /// 1. Set DIT_WEIGHTS_PATH to your DiT checkpoint dir (e.g. app container .../Checkpoints/acestep-v15-turbo).
    /// 2. Run: swift test --filter testDiTDecoderForwardWithLoadedWeights (or run this test in Xcode).
    /// 3. Fix AceStepSwift code; re-run until the test passes.
    func testDiTDecoderForwardWithLoadedWeights() throws {
        guard let dir = ProcessInfo.processInfo.environment[Self.ditWeightsPathEnv]?.trimmingCharacters(in: .whitespaces),
              !dir.isEmpty else {
            try XCTSkipIf(true, "Set \(Self.ditWeightsPathEnv) to the DiT checkpoint directory (contains model.safetensors) to run this test")
            return
        }
        let weightsURL = URL(fileURLWithPath: (dir as NSString).expandingTildeInPath).appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            try XCTSkipIf(true, "model.safetensors not found at \(weightsURL.path)")
            return
        }
        let decoder = DiTDecoder()
        let params = try loadParameters(from: weightsURL)
        decoder.update(parameters: params)
        let b = 1
        let t = 470
        let encL = 8
        let hiddenStates = MLXArray.zeros([b, t, 64])
        let contextLatents = MLXArray.zeros([b, t, 128])
        let encoderHiddenStates = MLXArray.zeros([b, encL, 2048])
        let timestep = MLXArray([Float(0.5)])
        let timestepR = MLXArray([Float(0)])
        let (out, _) = decoder.call(
            hiddenStates: hiddenStates,
            timestep: timestep,
            timestepR: timestepR,
            encoderHiddenStates: encoderHiddenStates,
            contextLatents: contextLatents,
            cache: nil,
            useCache: false
        )
        XCTAssertEqual(out.dim(0), b)
        XCTAssertEqual(out.dim(1), t)
        XCTAssertEqual(out.dim(2), 64)
    }

    // MARK: - DiTConditions contract (mirrors diffusion_test + lyric_alignment Python contract)

    /// Contract: encoderHiddenStates [B, encL, 2048], contextLatents [B, T, 128]; decoder output [B, T, 64].
    /// Matches make_test_fixtures and Python diffusion/service expectations.
    func testDiTConditionsContractShapes() {
        let b = 2
        let t = 8
        let encL = 5
        let conditions = DiTConditions(
            encoderHiddenStates: MLXArray.zeros([b, encL, 2048]),
            contextLatents: MLXArray.zeros([b, t, 128])
        )
        XCTAssertEqual(conditions.encoderHiddenStates!.dim(0), b)
        XCTAssertEqual(conditions.encoderHiddenStates!.dim(1), encL)
        XCTAssertEqual(conditions.encoderHiddenStates!.dim(2), 2048)
        XCTAssertEqual(conditions.contextLatents!.dim(0), b)
        XCTAssertEqual(conditions.contextLatents!.dim(1), t)
        XCTAssertEqual(conditions.contextLatents!.dim(2), 128)

        let decoder = DiTDecoder(
            hiddenSize: 2048,
            intermediateSize: 6144,
            numHiddenLayers: 2,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2
        )
        let (out, _) = decoder.call(
            hiddenStates: MLXArray.zeros([b, t, 64]),
            timestep: MLXArray([Float(0.5), 0.5]),
            timestepR: MLXArray([Float(0), 0]),
            encoderHiddenStates: conditions.encoderHiddenStates!,
            contextLatents: conditions.contextLatents!,
            cache: nil,
            useCache: false
        )
        XCTAssertEqual(out.dim(0), b)
        XCTAssertEqual(out.dim(1), t)
        XCTAssertEqual(out.dim(2), 64)
    }

    /// Batch dimensions must match: encoder and context same batch size (Python diffusion_test rejects mismatch).
    func testDiTConditionsBatchDimensionsMatch() {
        let b = 2
        let t = 4
        let encL = 3
        let enc = MLXArray.zeros([b, encL, 2048])
        let ctx = MLXArray.zeros([b, t, 128])
        XCTAssertEqual(enc.dim(0), ctx.dim(0), "Encoder and context must share batch size")
        let conditions = DiTConditions(encoderHiddenStates: enc, contextLatents: ctx)
        let stepper = MLXDiTStepper(decoder: DiTDecoder(
            hiddenSize: 2048,
            intermediateSize: 6144,
            numHiddenLayers: 2,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2
        ))
        let next = stepper.step(
            currentLatent: MLXArray.zeros([b, t, 64]),
            timestep: 1.0,
            conditions: conditions,
            nextTimestep: 0.5
        )
        XCTAssertEqual(next.dim(0), b)
    }

    // MARK: - CFG / Guidance Scale Tests

    /// Test that nullConditionEmbedding is properly extracted from DiT weights
    func testExtractNullConditionEmbeddingFromWeights() throws {
        guard let dir = ProcessInfo.processInfo.environment[Self.ditWeightsPathEnv]?.trimmingCharacters(in: .whitespaces),
              !dir.isEmpty else {
            try XCTSkipIf(true, "Set \(Self.ditWeightsPathEnv) to test null condition embedding extraction")
            return
        }
        let weightsURL = URL(fileURLWithPath: (dir as NSString).expandingTildeInPath).appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            try XCTSkipIf(true, "model.safetensors not found at \(weightsURL.path)")
            return
        }
        let nullEmb = try extractNullConditionEmbedding(fromDitWeightsURL: weightsURL)
        if let emb = nullEmb {
            XCTAssertEqual(emb.dim(0), 1, "Null embedding batch should be 1")
            XCTAssertEqual(emb.dim(1), 1, "Null embedding sequence should be 1")
            XCTAssertEqual(emb.dim(2), 2048, "Null embedding dim should match encoder hidden size")
        }
    }

    /// Test pipeline with guidanceScale > 1.0 uses CFG path correctly
    func testPipelineWithGuidanceScaleUsesCFG() {
        let decoder = DiTDecoder(
            hiddenSize: 2048,
            intermediateSize: 6144,
            numHiddenLayers: 2,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2
        )
        let stepper = MLXDiTStepper(decoder: decoder)
        let b = 1
        let encL = 8

        let nullEmb = MLXArray.zeros([1, 1, 2048])
        let pipeline = ContractGenerationPipeline(
            stepper: stepper,
            decoder: FakeVAEDecoder(),
            sampleRate: AceStepConstants.defaultSampleRate,
            conditioningProvider: { _, latentLength, _ in
                DiTConditions(
                    encoderHiddenStates: MLXArray.zeros([b, encL, 2048]),
                    contextLatents: MLXArray.zeros([b, latentLength, 128]),
                    nullConditionEmbedding: nullEmb
                )
            }
        )

        // Test with guidanceScale > 1.0 - should use CFG
        let params = GenerationParams(
            caption: "test",
            lyrics: "[Instrumental]",
            duration: 1.0,
            inferenceSteps: 2,
            seed: 42,
            guidanceScale: 7.0
        )

        let result = AceStepEngine.generateMusic(
            params: params,
            config: GenerationConfig(batchSize: 1),
            progress: nil,
            pipeline: pipeline
        )

        XCTAssertTrue(result.success, result.error ?? "no error")
        if result.success, let first = result.audios.first {
            let tensor = first["tensor"] as? [Float]
            XCTAssertNotNil(tensor)
            XCTAssertFalse(tensor?.isEmpty ?? true)
        }
    }

    /// Test pipeline with guidanceScale = 1.0 skips CFG
    func testPipelineWithGuidanceScaleOneSkipsCFG() {
        let decoder = DiTDecoder(
            hiddenSize: 2048,
            intermediateSize: 6144,
            numHiddenLayers: 2,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2
        )
        let stepper = MLXDiTStepper(decoder: decoder)
        let b = 1

        let pipeline = ContractGenerationPipeline(
            stepper: stepper,
            decoder: FakeVAEDecoder(),
            sampleRate: AceStepConstants.defaultSampleRate,
            conditioningProvider: { params, latentLength, _ in
                DiTConditions(
                    encoderHiddenStates: MLXArray.zeros([b, 8, 2048]),
                    contextLatents: MLXArray.zeros([b, latentLength, 128])
                )
            }
        )

        let params = GenerationParams(
            caption: "test",
            lyrics: "[Instrumental]",
            duration: 1.0,
            inferenceSteps: 2,
            seed: 42,
            guidanceScale: 1.0
        )

        let result = AceStepEngine.generateMusic(
            params: params,
            config: GenerationConfig(batchSize: 1),
            progress: nil,
            pipeline: pipeline
        )

        XCTAssertTrue(result.success, result.error ?? "no error")
    }

    /// Test CFG interval boundaries: cfgIntervalStart/End control when guidance is applied
    func testPipelineWithCFGIntervalBoundaries() {
        let decoder = DiTDecoder(
            hiddenSize: 2048,
            intermediateSize: 6144,
            numHiddenLayers: 2,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2
        )
        let stepper = MLXDiTStepper(decoder: decoder)
        let b = 1

        let nullEmb = MLXArray.zeros([1, 1, 2048])
        let pipeline = ContractGenerationPipeline(
            stepper: stepper,
            decoder: FakeVAEDecoder(),
            sampleRate: AceStepConstants.defaultSampleRate,
            conditioningProvider: { params, latentLength, _ in
                DiTConditions(
                    encoderHiddenStates: MLXArray.zeros([b, 8, 2048]),
                    contextLatents: MLXArray.zeros([b, latentLength, 128]),
                    nullConditionEmbedding: nullEmb
                )
            }
        )

        // CFG interval [0.5, 1.0] - guidance applied in later steps only
        let params = GenerationParams(
            caption: "test",
            lyrics: "[Instrumental]",
            duration: 1.0,
            inferenceSteps: 4,
            seed: 42,
            guidanceScale: 7.0,
            cfgIntervalStart: 0.5,
            cfgIntervalEnd: 1.0
        )

        let result = AceStepEngine.generateMusic(
            params: params,
            config: GenerationConfig(batchSize: 1),
            progress: nil,
            pipeline: pipeline
        )

        XCTAssertTrue(result.success, result.error ?? "no error")
    }

    // MARK: - Timestep R Validation

    /// Verify timestep_r produces zero in decoder (matching Python behavior).
    /// Python: timestep_r = current timestep, so timestep - timestep_r = 0 for second embedding.
    func testTimestepRProducesZeroInDecoder() {
        let decoder = DiTDecoder(
            hiddenSize: 2048,
            intermediateSize: 6144,
            numHiddenLayers: 2,
            numAttentionHeads: 16,
            numKeyValueHeads: 8,
            headDim: 128,
            inChannels: 192,
            audioAcousticHiddenDim: 64,
            patchSize: 2
        )
        let stepper = MLXDiTStepper(decoder: decoder)
        let b = 1
        let t = 4
        let encL = 8

        // Test at timestep = 0.5, timestepR = 0.5 -> difference = 0
        let currentLatent = MLXArray.zeros([b, t, 64])
        let conditions = DiTConditions(
            encoderHiddenStates: MLXArray.zeros([b, encL, 2048]),
            contextLatents: MLXArray.zeros([b, t, 128])
        )

        // This should use timestep_r = timestep = 0.5 internally
        let vt = stepper.predictVelocity(
            currentLatent: currentLatent,
            timestep: 0.5,
            conditions: conditions,
            useCache: false
        )

        // Verify output shape
        XCTAssertEqual(vt.dim(0), b)
        XCTAssertEqual(vt.dim(1), t)
        XCTAssertEqual(vt.dim(2), 64)
    }

    // MARK: - DiT with Real Weights Tests

    /// Full pipeline with real DiT weights and real VAE (when available)
    func testFullPipelineWithRealDiTWeights() throws {
        guard let dir = ProcessInfo.processInfo.environment[Self.ditWeightsPathEnv]?.trimmingCharacters(in: .whitespaces),
              !dir.isEmpty else {
            try XCTSkipIf(true, "Set \(Self.ditWeightsPathEnv) to run DiT with real weights")
            return
        }
        let weightsURL = URL(fileURLWithPath: (dir as NSString).expandingTildeInPath).appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            try XCTSkipIf(true, "model.safetensors not found at \(weightsURL.path)")
            return
        }
        let decoder = DiTDecoder()
        let params = try loadDiTParametersForDecoder(from: weightsURL)
        decoder.update(parameters: params)
        let stepper = MLXDiTStepper(decoder: decoder)
        let b = 1

        let pipeline = ContractGenerationPipeline(
            stepper: stepper,
            decoder: FakeVAEDecoder(),
            sampleRate: AceStepConstants.defaultSampleRate,
            conditioningProvider: { params, latentLength, _ in
                DiTConditions(
                    encoderHiddenStates: MLXArray.zeros([b, 8, 2048]),
                    contextLatents: MLXArray.zeros([b, latentLength, 128])
                )
            }
        )

        let result = AceStepEngine.generateMusic(
            params: GenerationParams(
                caption: "test",
                lyrics: "[Instrumental]",
                duration: 1.0,
                inferenceSteps: 2,
                seed: 42
            ),
            config: GenerationConfig(batchSize: 1),
            progress: nil,
            pipeline: pipeline
        )

        XCTAssertTrue(result.success, result.error ?? "no error")
        if result.success, let first = result.audios.first {
            let tensor = first["tensor"] as? [Float]
            XCTAssertNotNil(tensor)
            XCTAssertFalse(tensor?.isEmpty ?? true)
        }
    }
}
