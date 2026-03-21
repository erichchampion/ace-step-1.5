import XCTest
@testable import AceStepSwift

final class LyricSegmenterTests: XCTestCase {

    // MARK: - Instrumental

    func testInstrumental_AllChunksGetInstrumental() {
        let chunks = makeChunks(count: 3, chunkDuration: 30, overlapSec: 10)
        let result = LyricSegmenter.segment(lyrics: "[Instrumental]", chunks: chunks, totalDuration: 70)
        XCTAssertEqual(result.count, 3)
        for r in result {
            XCTAssertEqual(r, "[Instrumental]")
        }
    }

    func testEmptyLyrics_AllChunksGetInstrumental() {
        let chunks = makeChunks(count: 2, chunkDuration: 30, overlapSec: 10)
        let result = LyricSegmenter.segment(lyrics: "", chunks: chunks, totalDuration: 50)
        XCTAssertEqual(result.count, 2)
        for r in result {
            XCTAssertEqual(r, "[Instrumental]")
        }
    }

    // MARK: - Single Chunk

    func testSingleChunk_ReturnsFullLyrics() {
        let lyrics = "[Verse 1]\nHello world\n[Chorus]\nLa la la"
        let chunks = makeChunks(count: 1, chunkDuration: 30, overlapSec: 0)
        let result = LyricSegmenter.segment(lyrics: lyrics, chunks: chunks, totalDuration: 30)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], lyrics)
    }

    // MARK: - Structured Lyrics

    func testStructuredLyrics_DistributedAcrossChunks() {
        let lyrics = """
        [Verse 1]
        Line A1
        Line A2
        [Chorus]
        Line B1
        Line B2
        [Verse 2]
        Line C1
        Line C2
        [Outro]
        Line D1
        """
        let chunks = makeChunks(count: 2, chunkDuration: 30, overlapSec: 10)
        let result = LyricSegmenter.segment(lyrics: lyrics, chunks: chunks, totalDuration: 50)
        XCTAssertEqual(result.count, 2)
        // First chunk should contain at least the Verse 1 and Chorus
        XCTAssertTrue(result[0].contains("Verse 1"), "First chunk should contain Verse 1")
        // Second chunk should contain the later sections
        XCTAssertTrue(result[1].contains("Line C") || result[1].contains("Line D") || result[1].contains("Verse 2") || result[1].contains("Outro"),
                       "Second chunk should contain later sections")
    }

    // MARK: - Unstructured Lyrics (Line-by-Line)

    func testUnstructuredLyrics_DistributedByLines() {
        let lyrics = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6"
        let chunks = makeChunks(count: 2, chunkDuration: 30, overlapSec: 10)
        let result = LyricSegmenter.segment(lyrics: lyrics, chunks: chunks, totalDuration: 50)
        XCTAssertEqual(result.count, 2)
        // First chunk should have some lines
        XCTAssertTrue(result[0].contains("Line 1"))
        // Second chunk should have later lines
        XCTAssertTrue(result[1].contains("Line 6") || result[1].contains("Line 5"))
    }

    // MARK: - Section Parsing

    func testParseSections_StructuredMarkers() {
        let lyrics = "[Verse 1]\nA\nB\n[Chorus]\nC\nD\n[Bridge]\nE"
        let sections = LyricSegmenter.parseSections(from: lyrics)
        XCTAssertEqual(sections.count, 3)
        XCTAssertEqual(sections[0].header, "[Verse 1]")
        XCTAssertEqual(sections[1].header, "[Chorus]")
        XCTAssertEqual(sections[2].header, "[Bridge]")
    }

    func testParseSections_CaseInsensitive() {
        let lyrics = "[verse 1]\nA\n[CHORUS]\nB"
        let sections = LyricSegmenter.parseSections(from: lyrics)
        XCTAssertEqual(sections.count, 2)
        XCTAssertEqual(sections[0].header, "[verse 1]")
        XCTAssertEqual(sections[1].header, "[CHORUS]")
    }

    func testParseSections_NoHeaders() {
        let lyrics = "Just some lyrics\nNo structure here"
        let sections = LyricSegmenter.parseSections(from: lyrics)
        XCTAssertEqual(sections.count, 1)
        XCTAssertNil(sections[0].header)
    }

    // MARK: - Context Lines

    func testContextLines_PresentAtBoundaries() {
        let lyrics = """
        [Verse 1]
        A line in verse
        [Verse 2]
        A line in verse 2
        [Chorus]
        A line in chorus
        [Outro]
        A line in outro
        """
        let chunks = makeChunks(count: 2, chunkDuration: 30, overlapSec: 10)
        let result = LyricSegmenter.segment(lyrics: lyrics, chunks: chunks, totalDuration: 50)
        XCTAssertEqual(result.count, 2)
        // Context markers should appear at transitions
        let firstContainsContinues = result[0].contains("(continues)")
        let secondContainsContinued = result[1].contains("(continued)")
        // At least one boundary should have context
        XCTAssertTrue(firstContainsContinues || secondContainsContinued,
                       "Should include context lines at chunk boundaries")
    }

    // MARK: - Helpers

    private func makeChunks(count: Int, chunkDuration: Double, overlapSec: Double) -> [ProgressiveChunk] {
        var chunks: [ProgressiveChunk] = []
        let advance = chunkDuration - overlapSec
        for i in 0..<count {
            let start = Double(i) * advance
            let end = start + chunkDuration
            chunks.append(ProgressiveChunk(
                index: i,
                startSec: start,
                endSec: end,
                overlapWithPreviousSec: i == 0 ? 0 : overlapSec,
                isFirstChunk: i == 0,
                lyrics: ""
            ))
        }
        return chunks
    }
}
