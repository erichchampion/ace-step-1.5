import XCTest
import MLX
@testable import AceStepSwift

final class GenerateMusicAPITests: XCTestCase {

    func testGenerateMusicWithoutEngineReturnsError() {
        let result = AceStepEngine.generateMusic(
            params: GenerationParams(caption: "pop", lyrics: "[Instrumental]"),
            config: GenerationConfig(),
            progress: nil,
            pipeline: nil
        )
        XCTAssertFalse(result.success)
        XCTAssertEqual(result.error, "Pipeline not initialized")
    }

    func testGenerateMusicWithStubPipelineReturnsSuccessWithEmptyAudios() {
        let stub = StubGenerationPipeline()
        let result = AceStepEngine.generateMusic(
            params: GenerationParams(caption: "jazz", lyrics: "[Verse 1] test"),
            config: GenerationConfig(batchSize: 1),
            progress: nil,
            pipeline: stub
        )
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.audios.isEmpty)
    }

    /// Invalid params (e.g. negative duration) do not crash; pipeline uses default latent length and returns success with stub.
    func testGenerateMusicWithInvalidParamsReturnsSuccessWithStub() {
        let stub = StubGenerationPipeline()
        let result = AceStepEngine.generateMusic(
            params: GenerationParams(
                caption: "test",
                lyrics: "[Instrumental]",
                duration: -1,
                inferenceSteps: 0,
                seed: -1
            ),
            config: GenerationConfig(batchSize: 1),
            progress: nil,
            pipeline: stub
        )
        XCTAssertTrue(result.success, result.error ?? "no error")
        XCTAssertTrue(result.audios.isEmpty)
    }

    func testLatentLengthFromDurationPositiveDuration() {
        let t = latentLengthFromDuration(durationSeconds: 30, sampleRate: 48000)
        let expected = Int(ceil(30 * 48000 / Double(vaeLatentToSamplesFactor)))
        XCTAssertEqual(t, max(1, expected))
    }

    func testLatentLengthFromDurationZeroReturnsDefault() {
        let t = latentLengthFromDuration(durationSeconds: 0, sampleRate: 48000)
        XCTAssertEqual(t, 100)
    }

    /// ContractGenerationPipeline with fake stepper/decoder returns success and non-empty audios.
    /// Requires Metal (MLX); may be skipped in environments without GPU.
    func testGenerateMusicWithContractPipelineReturnsSuccessWithAudios() {
        let pipeline = ContractGenerationPipeline(
            stepper: FakeDiffusionStepper(),
            decoder: FakeVAEDecoder(),
            sampleRate: AceStepConstants.defaultSampleRate
        )
        let result = AceStepEngine.generateMusic(
            params: GenerationParams(
                caption: "test",
                lyrics: "[Instrumental]",
                duration: 30,
                inferenceSteps: 4,
                seed: 42
            ),
            config: GenerationConfig(batchSize: 1),
            progress: nil,
            pipeline: pipeline
        )
        XCTAssertTrue(result.success, result.error ?? "no error")
        XCTAssertFalse(result.audios.isEmpty)
        guard let first = result.audios.first else { return }
        let tensor = first["tensor"] as? [Float]
        let sr = first["sample_rate"] as? Int
        XCTAssertNotNil(tensor)
        XCTAssertNotNil(sr)
        XCTAssertEqual(sr, AceStepConstants.defaultSampleRate)
        XCTAssertFalse(tensor?.isEmpty ?? true)
    }

    /// When conditioning provider supplies initialLatents (e.g. cover/repaint), pipeline starts from it instead of noise.
    func testPipelineUsesInitialLatentsWhenProvided() {
        let latentLen = latentLengthFromDuration(durationSeconds: 10, sampleRate: AceStepConstants.defaultSampleRate)
        let initialLatents = MLXArray.zeros([1, latentLen, 64])
        let pipeline = ContractGenerationPipeline(
            stepper: FakeDiffusionStepper(),
            decoder: FakeVAEDecoder(),
            sampleRate: AceStepConstants.defaultSampleRate,
            conditioningProvider: { _, _, _ in
                DiTConditions(initialLatents: initialLatents)
            }
        )
        let result = AceStepEngine.generateMusic(
            params: GenerationParams(
                caption: "cover",
                lyrics: "[Instrumental]",
                duration: 10,
                inferenceSteps: 2,
                seed: 1
            ),
            config: GenerationConfig(batchSize: 1),
            progress: nil,
            pipeline: pipeline
        )
        XCTAssertTrue(result.success, result.error ?? "no error")
        XCTAssertFalse(result.audios.isEmpty)
    }
}
