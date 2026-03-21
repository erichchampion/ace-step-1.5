import XCTest
@testable import AceStepSwift

final class ProgressiveGenerationOrchestratorTests: XCTestCase {

    // MARK: - Single Chunk (Passthrough)

    func testSingleChunk_PassesThrough() async throws {
        let plan = ProgressiveGenerationPlan.plan(
            totalDuration: 20,
            maxChunkDuration: 30,
            overlapSec: 10,
            crossfadeSec: 2,
            lyrics: "test",
            caption: "test"
        )
        XCTAssertFalse(plan.isProgressive)

        var generationCallCount = 0
        let params = GenerationParams(caption: "test", lyrics: "test", duration: 20, inferenceSteps: 2, seed: 42)
        let config = GenerationConfig(batchSize: 1)


        let result = try await ProgressiveGenerationOrchestrator.generate(
            params: params,
            config: config,
            plan: plan,
            generateChunk: { chunkParams, chunkConfig, chunkIndex, progress in
                generationCallCount += 1
                return GenerationResult(
                    audios: [[
                        "tensor": [Float](repeating: 0.5, count: 48000),
                        "sample_rate": 48000,
                        "channels": 1
                    ]],
                    statusMessage: "Done",
                    success: true
                )
            },
            progress: nil
        )

        XCTAssertTrue(result.success)
        XCTAssertEqual(generationCallCount, 1)
    }

    // MARK: - Multi-Chunk Progressive

    func testMultiChunk_CallsCorrectNumberOfTimes() async throws {
        let plan = ProgressiveGenerationPlan.plan(
            totalDuration: 60,
            maxChunkDuration: 30,
            overlapSec: 10,
            crossfadeSec: 2,
            lyrics: "[Instrumental]",
            caption: "test"
        )
        XCTAssertTrue(plan.isProgressive)
        let expectedChunks = plan.chunks.count

        var generationCalls: [Int] = []
        var taskTypes: [String] = []

        let params = GenerationParams(caption: "test", lyrics: "[Instrumental]", duration: 60, inferenceSteps: 2, seed: 42)
        let config = GenerationConfig(batchSize: 1)


        let result = try await ProgressiveGenerationOrchestrator.generate(
            params: params,
            config: config,
            plan: plan,
            generateChunk: { chunkParams, chunkConfig, chunkIndex, progress in
                generationCalls.append(chunkIndex)
                taskTypes.append(chunkParams.taskType)
                let sr = 48000
                let samples = Int(chunkParams.duration * Double(sr))
                return GenerationResult(
                    audios: [[
                        "tensor": [Float](repeating: 0.5, count: samples),
                        "sample_rate": sr,
                        "channels": 1
                    ]],
                    statusMessage: "Done",
                    success: true
                )
            },
            progress: nil
        )

        XCTAssertTrue(result.success)
        XCTAssertEqual(generationCalls.count, expectedChunks)
        // All chunks use text2music (crossfade stitching handles continuity)
        for i in 0..<taskTypes.count {
            XCTAssertEqual(taskTypes[i], "text2music", "Chunk \(i) should use text2music")
        }
    }

    // MARK: - Progress Callbacks

    func testProgressCallbacks_FireInOrder() async throws {
        let plan = ProgressiveGenerationPlan.plan(
            totalDuration: 60,
            maxChunkDuration: 30,
            overlapSec: 10,
            crossfadeSec: 2,
            lyrics: "[Instrumental]",
            caption: "test"
        )

        var progressUpdates: [ProgressiveProgressUpdate] = []
        let params = GenerationParams(caption: "test", lyrics: "[Instrumental]", duration: 60, inferenceSteps: 2, seed: 42)
        let config = GenerationConfig(batchSize: 1)


        _ = try await ProgressiveGenerationOrchestrator.generate(
            params: params,
            config: config,
            plan: plan,
            generateChunk: { chunkParams, chunkConfig, chunkIndex, progress in
                progress?(0.5, "Step 1/2")
                progress?(1.0, "Step 2/2")
                let sr = 48000
                let samples = Int(chunkParams.duration * Double(sr))
                return GenerationResult(
                    audios: [[
                        "tensor": [Float](repeating: 0.5, count: samples),
                        "sample_rate": sr,
                        "channels": 1
                    ]],
                    statusMessage: "Done",
                    success: true
                )
            },
            progress: { update in progressUpdates.append(update) }
        )

        XCTAssertFalse(progressUpdates.isEmpty)
        // Should have "Starting chunk" + per-step updates + "Stitching" + "complete"
        XCTAssertTrue(progressUpdates.last!.message.contains("complete") || progressUpdates.last!.message.contains("Stitching"))
    }

    // MARK: - Error Handling

    func testChunkFailure_ThrowsError() async {
        let plan = ProgressiveGenerationPlan.plan(
            totalDuration: 60,
            maxChunkDuration: 30,
            overlapSec: 10,
            crossfadeSec: 2,
            lyrics: "test",
            caption: "test"
        )

        let params = GenerationParams(caption: "test", lyrics: "test", duration: 60, inferenceSteps: 2, seed: 42)
        let config = GenerationConfig(batchSize: 1)


        do {
            _ = try await ProgressiveGenerationOrchestrator.generate(
                params: params,
                config: config,
                plan: plan,
                generateChunk: { chunkParams, chunkConfig, chunkIndex, progress in
                    if chunkIndex == 0 {
                        let sr = 48000
                        let samples = Int(chunkParams.duration * Double(sr))
                        return GenerationResult(
                            audios: [[
                                "tensor": [Float](repeating: 0.5, count: samples),
                                "sample_rate": sr,
                                "channels": 1
                            ]],
                            statusMessage: "Done",
                            success: true
                        )
                    }
                    return GenerationResult(success: false, error: "Metal out of memory")
                },
                progress: nil
            )
            XCTFail("Should have thrown")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("failed") || error.localizedDescription.contains("Metal"),
                          "Error should mention failure: \(error.localizedDescription)")
        }
    }
}
