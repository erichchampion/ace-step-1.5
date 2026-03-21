import XCTest
@testable import AceStepSwift

final class ProgressiveGenerationPlanTests: XCTestCase {

    // MARK: - Single Chunk (No Progressive)

    func testDurationWithinLimit_SingleChunk() {
        let plan = ProgressiveGenerationPlan.plan(
            totalDuration: 30,
            maxChunkDuration: 30,
            overlapSec: 10,
            crossfadeSec: 2,
            lyrics: "[Verse 1]\nHello world",
            caption: "test"
        )
        XCTAssertFalse(plan.isProgressive)
        XCTAssertEqual(plan.chunks.count, 1)
        XCTAssertEqual(plan.chunks[0].startSec, 0)
        XCTAssertEqual(plan.chunks[0].endSec, 30)
        XCTAssertTrue(plan.chunks[0].isFirstChunk)
        XCTAssertEqual(plan.chunks[0].overlapWithPreviousSec, 0)
    }

    func testDurationBelowLimit_SingleChunk() {
        let plan = ProgressiveGenerationPlan.plan(
            totalDuration: 20,
            maxChunkDuration: 30,
            overlapSec: 10,
            crossfadeSec: 2,
            lyrics: "test",
            caption: "test"
        )
        XCTAssertFalse(plan.isProgressive)
        XCTAssertEqual(plan.chunks.count, 1)
    }

    // MARK: - Multi-Chunk Progressive

    func testDoubleChunk_120sWithin30sLimit() {
        let plan = ProgressiveGenerationPlan.plan(
            totalDuration: 120,
            maxChunkDuration: 30,
            overlapSec: 10,
            crossfadeSec: 2,
            lyrics: "[Instrumental]",
            caption: "test"
        )
        XCTAssertTrue(plan.isProgressive)
        // advance per chunk = 30 - 10 = 20s
        // chunks: [0-30], [20-50], [40-70], [60-90], [80-110], [100-120]
        // But chunk at [100-120] has remaining=0 after it, so just 6 chunks
        XCTAssertGreaterThanOrEqual(plan.chunks.count, 2)

        // First chunk assertions
        XCTAssertTrue(plan.chunks[0].isFirstChunk)
        XCTAssertEqual(plan.chunks[0].startSec, 0)
        XCTAssertEqual(plan.chunks[0].overlapWithPreviousSec, 0)

        // Second chunk assertions
        XCTAssertFalse(plan.chunks[1].isFirstChunk)
        XCTAssertEqual(plan.chunks[1].overlapWithPreviousSec, 10)
        XCTAssertEqual(plan.chunks[1].startSec, 20)

        // Last chunk should end at totalDuration
        XCTAssertEqual(plan.chunks.last!.endSec, 120)
    }

    func testOverlapExceedsCrossfade_CrossfadeClamped() {
        let plan = ProgressiveGenerationPlan.plan(
            totalDuration: 60,
            maxChunkDuration: 30,
            overlapSec: 10,
            crossfadeSec: 15, // Exceeds overlap
            lyrics: "test",
            caption: "test"
        )
        // crossfadeSec should be clamped to min(15, 10) = 10
        XCTAssertEqual(plan.crossfadeSec, 10)
    }

    func testSmallRemainder_MergesIntoLastChunk() {
        // totalDuration=55, maxChunkDuration=30, overlap=10, advance=20
        // Chunks: [0–30], next would start at 20, end at 50, remaining = 5
        // 5 < overlap(10) + 5 = 15, so chunk [20–50] should extend to [20–55]
        let plan = ProgressiveGenerationPlan.plan(
            totalDuration: 55,
            maxChunkDuration: 30,
            overlapSec: 10,
            crossfadeSec: 2,
            lyrics: "test",
            caption: "test"
        )
        XCTAssertTrue(plan.isProgressive)
        XCTAssertEqual(plan.chunks.last!.endSec, 55)
    }

    func testOverlapEqualsMaxDuration_FallsBackToSingleChunk() {
        // overlap >= maxChunkDuration is invalid
        let plan = ProgressiveGenerationPlan.plan(
            totalDuration: 120,
            maxChunkDuration: 30,
            overlapSec: 30, // == maxChunkDuration
            crossfadeSec: 2,
            lyrics: "test",
            caption: "test"
        )
        XCTAssertFalse(plan.isProgressive)
        XCTAssertEqual(plan.chunks.count, 1)
    }

    // MARK: - Chunk Duration Property

    func testChunkDurationProperty() {
        let chunk = ProgressiveChunk(
            index: 0,
            startSec: 10, endSec: 40,
            overlapWithPreviousSec: 5,
            isFirstChunk: false,
            lyrics: "test"
        )
        XCTAssertEqual(chunk.durationSec, 30)
    }

    // MARK: - Exact Multiple Duration

    func testExactMultiple_60sIn30sChunks() {
        let plan = ProgressiveGenerationPlan.plan(
            totalDuration: 60,
            maxChunkDuration: 30,
            overlapSec: 10,
            crossfadeSec: 2,
            lyrics: "[Instrumental]",
            caption: "test"
        )
        XCTAssertTrue(plan.isProgressive)
        // 30-10=20 advance. Chunk 0: [0-30]. Chunk 1 starts at 20, naive end=50,
        // remaining=10 < overlap(10)+5=15, so tail-merge extends to [20-60] → 2 chunks
        XCTAssertEqual(plan.chunks.count, 2)
        XCTAssertEqual(plan.chunks.last!.endSec, 60)
    }
}
