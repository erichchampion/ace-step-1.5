import XCTest
@testable import AceStepSwift

final class CreateSampleAPITests: XCTestCase {

    func testCreateSampleWithoutProviderReturnsNotInitialized() {
        let result = AceStepEngine.createSample(
            query: "a soft love song",
            instrumental: false,
            vocalLanguage: nil,
            temperature: 0.85,
            llmProvider: nil
        )
        XCTAssertFalse(result.success)
        XCTAssertEqual(result.error, "LLM not initialized")
    }

    func testCreateSampleWithUninitializedProviderReturnsNotInitialized() {
        let provider = MLXLLMFormatProvider()
        let result = AceStepEngine.createSample(
            query: "a soft love song",
            instrumental: false,
            vocalLanguage: nil,
            temperature: 0.85,
            llmProvider: provider
        )
        XCTAssertFalse(result.success)
        XCTAssertEqual(result.error, "LLM not initialized")
    }

    func testCreateSampleWithMockProviderReturnsParsedResult() {
        let mockRaw = """
        <think>
        bpm: 100
        caption: A gentle acoustic ballad.
        duration: 120
        keyscale: G major
        language: en
        timesignature: 4
        </think>

        [Verse 1]
        Hello world
        """
        let mock = MockLLMFormatProvider(responseText: mockRaw)
        let result = AceStepEngine.createSample(
            query: "a soft love song",
            instrumental: false,
            vocalLanguage: nil,
            temperature: 0.85,
            llmProvider: mock
        )
        XCTAssertTrue(result.success)
        XCTAssertEqual(result.caption, "A gentle acoustic ballad.")
        XCTAssertEqual(result.bpm, 100)
        XCTAssertEqual(result.duration, 120)
        XCTAssertEqual(result.keyscale, "G major")
        XCTAssertEqual(result.language, "en")
        XCTAssertEqual(result.timesignature, "4")
        XCTAssertFalse(result.instrumental)
        XCTAssertTrue(result.lyrics.contains("Hello world"))
    }

    func testCreateSampleInstrumentalEchoedInResult() {
        let mockRaw = """
        <think>
        bpm: 90
        caption: Ambient piece.
        duration: 60
        keyscale: C major
        language: en
        timesignature: 4
        </think>

        [Instrumental]
        """
        let mock = MockLLMFormatProvider(responseText: mockRaw)
        let result = AceStepEngine.createSample(
            query: "ambient",
            instrumental: true,
            vocalLanguage: nil,
            temperature: 0.85,
            llmProvider: mock
        )
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.instrumental)
    }
}
