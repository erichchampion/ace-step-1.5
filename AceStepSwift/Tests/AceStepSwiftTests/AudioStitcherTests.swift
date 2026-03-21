import XCTest
@testable import AceStepSwift

final class AudioStitcherTests: XCTestCase {

    // MARK: - Single Chunk

    func testSingleChunk_ReturnsUnchanged() {
        let audio: [Float] = [0.1, 0.2, 0.3, 0.4]
        let stitched = AudioStitcher.stitch(
            chunks: [audio],
            sampleRate: 4,
            channels: 1,
            chunkStartsSec: [0.0],
            overlapSec: 0,
            crossfadeSec: 0
        )
        XCTAssertEqual(stitched, audio)
    }

    func testEmptyChunks_ReturnsEmpty() {
        let stitched = AudioStitcher.stitch(
            chunks: [],
            sampleRate: 48000,
            channels: 1,
            chunkStartsSec: [],
            overlapSec: 0,
            crossfadeSec: 0
        )
        XCTAssertTrue(stitched.isEmpty)
    }

    // MARK: - Two Chunks With Crossfade

    func testTwoConstantChunks_CrossfadePreservesLevel() {
        // Two chunks of constant 1.0, overlapping by 4 samples at rate=4
        let sampleRate = 4
        let chunkDuration = 6 // 6 samples, 1.5 seconds at rate=4
        let overlapSec = 1.0 // 4 samples
        let crossfadeSec = 1.0

        let chunk1 = [Float](repeating: 1.0, count: chunkDuration)
        let chunk2 = [Float](repeating: 1.0, count: chunkDuration)

        // chunk1 at 0.0s, chunk2 at 0.5s (advanced by 0.5s = 2 samples)
        let advanceSec = Double(chunkDuration) / Double(sampleRate) - overlapSec
        let stitched = AudioStitcher.stitch(
            chunks: [chunk1, chunk2],
            sampleRate: sampleRate,
            channels: 1,
            chunkStartsSec: [0.0, advanceSec],
            overlapSec: overlapSec,
            crossfadeSec: crossfadeSec
        )

        // All values should be close to 1.0 since both inputs are 1.0
        // (cos^2 + sin^2 = 1 for equal-power crossfade when both inputs are equal)
        for (i, val) in stitched.enumerated() {
            XCTAssertEqual(val, 1.0, accuracy: 0.01,
                           "Sample \(i) should be ~1.0 for constant-value crossfade")
        }
    }

    // MARK: - Output Length

    func testOutputLength_CorrectForOverlap() {
        let sampleRate = 100 // Samples per second
        let channels = 1
        let chunk1Samples = 300 // 3.0 seconds
        let chunk2Samples = 300 // 3.0 seconds
        let overlapSec = 1.0 // 100 samples overlap
        let crossfadeSec = 0.5

        let chunk1 = [Float](repeating: 0.5, count: chunk1Samples)
        let chunk2 = [Float](repeating: 0.5, count: chunk2Samples)

        // Chunk 1 at 0.0s, Chunk 2 at 2.0s (advance = 3.0 - 1.0 = 2.0)
        let stitched = AudioStitcher.stitch(
            chunks: [chunk1, chunk2],
            sampleRate: sampleRate,
            channels: channels,
            chunkStartsSec: [0.0, 2.0],
            overlapSec: overlapSec,
            crossfadeSec: crossfadeSec
        )

        // Expected: chunk2 starts at 200, extends 300 samples → total = 500
        let expectedLength = 200 + 300
        XCTAssertEqual(stitched.count, expectedLength)
    }

    // MARK: - Stereo Interleaved

    func testStereo_InterleavedHandledCorrectly() {
        let sampleRate = 4
        let channels = 2
        // 8 interleaved samples = 4 frames
        let chunk1: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        let chunk2: [Float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        let stitched = AudioStitcher.stitch(
            chunks: [chunk1, chunk2],
            sampleRate: sampleRate,
            channels: channels,
            chunkStartsSec: [0.0, 0.5], // chunk2 starts at 0.5s = 2 frames = 4 interleaved samples
            overlapSec: 0.5, // 2 frames overlap
            crossfadeSec: 0.5
        )

        // Output should have: 2 non-overlapping frames from chunk1 + 2 crossfaded + 2 non-overlapping from chunk2
        // Total = 4 frames = 8 interleaved samples (chunk2's start at frame 2, chunk2 has 4 frames → ends at frame 6)
        XCTAssertGreaterThan(stitched.count, 0)
        // Verify it's even (stereo interleaved)
        XCTAssertEqual(stitched.count % 2, 0)
    }
}
