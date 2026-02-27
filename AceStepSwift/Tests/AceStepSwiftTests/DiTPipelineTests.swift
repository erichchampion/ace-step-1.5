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
            sampleRate: AceStepConstants.defaultSampleRate
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
}
