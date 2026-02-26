import XCTest
@testable import AceStepSwift

final class FormatSampleAPITests: XCTestCase {

    func testMLXLLMFormatProviderNotInitialized() {
        let provider = MLXLLMFormatProvider()
        XCTAssertFalse(provider.isInitialized)
    }

    func testMLXLLMFormatProviderGenerateFormatThrowsWhenNotLoaded() async {
        let provider = MLXLLMFormatProvider()
        do {
            _ = try await provider.generateFormat(
                caption: "pop",
                lyrics: "test",
                userMetadata: nil,
                temperature: 0.85
            )
            XCTFail("Expected MLXLLMFormatProviderError.notInitialized")
        } catch is MLXLLMFormatProviderError {
            // expected
        } catch {
            XCTFail("Expected MLXLLMFormatProviderError, got \(error)")
        }
    }

    func testFormatSampleWithoutProviderReturnsNotInitialized() {
        let result = AceStepEngine.formatSample(
            caption: "pop",
            lyrics: "[Verse 1] test",
            userMetadata: nil,
            temperature: 0.85,
            llmProvider: nil
        )
        XCTAssertFalse(result.success)
        XCTAssertEqual(result.error, "LLM not initialized")
    }

    func testFormatSampleWithMockProviderReturnsParsedResult() {
        let mockRaw = """
        <think>
        bpm: 110
        caption: Mock caption.
        duration: 45
        keyscale: F major
        language: en
        timesignature: 4
        </think>
        """
        let mock = MockLLMFormatProvider(responseText: mockRaw)
        let result = AceStepEngine.formatSample(
            caption: "pop",
            lyrics: "lyrics",
            userMetadata: nil,
            temperature: 0.85,
            llmProvider: mock
        )
        XCTAssertTrue(result.success)
        XCTAssertEqual(result.bpm, 110)
        XCTAssertEqual(result.caption, "Mock caption.")
        XCTAssertEqual(result.duration, 45)
        XCTAssertEqual(result.keyscale, "F major")
    }
}
