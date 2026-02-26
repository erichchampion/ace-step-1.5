import XCTest
@testable import AceStepSwift

final class FormatSampleParserTests: XCTestCase {

    func testParseMetadataFromThinkBlock() {
        let raw = """
        <think>
        bpm: 120
        caption: A dramatic Latin pop track.
        duration: 60
        keyscale: C major
        language: en
        timesignature: 4
        </think>
        """
        let (metadata, lyrics) = FormatSampleParser.parseLMOutput(raw)
        XCTAssertEqual(metadata["bpm"] as? Int, 120)
        XCTAssertEqual(metadata["caption"] as? String, "A dramatic Latin pop track.")
        XCTAssertEqual(metadata["duration"] as? Int, 60)
        XCTAssertEqual(metadata["keyscale"] as? String, "C major")
        XCTAssertEqual(metadata["language"] as? String, "en")
        XCTAssertEqual(metadata["timesignature"] as? String, "4")
        XCTAssertTrue(lyrics.isEmpty)
    }

    func testExtractLyricsAfterThinkEnd() {
        let raw = """
        <think>
        bpm: 100
        caption: Test.
        duration: 30
        keyscale: G major
        language: en
        timesignature: 4
        </think>

        # Lyric
        [Verse 1]
        Hello world
        """
        let lyrics = FormatSampleParser.extractLyricsFromOutput(raw)
        XCTAssertTrue(lyrics.contains("[Verse 1]"))
        XCTAssertTrue(lyrics.contains("Hello world"))
    }

    func testParseToFormatSampleResult() {
        let raw = """
        <think>
        bpm: 90
        caption: Chill beat.
        duration: 120
        keyscale: Am
        language: en
        timesignature: 4
        </think>
        """
        let result = FormatSampleParser.parseToFormatSampleResult(
            outputText: raw,
            fallbackLyrics: "[Instrumental]"
        )
        XCTAssertTrue(result.success)
        XCTAssertEqual(result.bpm, 90)
        XCTAssertEqual(result.caption, "Chill beat.")
        XCTAssertEqual(result.duration, 120)
        XCTAssertEqual(result.keyscale, "Am")
        XCTAssertEqual(result.language, "en")
        XCTAssertEqual(result.timesignature, "4")
        XCTAssertEqual(result.lyrics, "[Instrumental]")
    }
}
