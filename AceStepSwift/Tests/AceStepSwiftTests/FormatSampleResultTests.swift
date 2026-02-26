import XCTest
@testable import AceStepSwift

final class FormatSampleResultTests: XCTestCase {

    func testSuccessResultCreation() {
        let result = FormatSampleResult(
            caption: "A dramatic Latin pop track",
            lyrics: "[Verse 1]\nHello world",
            bpm: 120,
            duration: 60.0,
            keyscale: "C major",
            language: "en",
            timesignature: "4",
            statusMessage: "OK",
            success: true,
            error: nil
        )
        XCTAssertTrue(result.success)
        XCTAssertEqual(result.caption, "A dramatic Latin pop track")
        XCTAssertEqual(result.bpm, 120)
        XCTAssertEqual(result.duration, 60.0)
        XCTAssertNil(result.error)
    }

    func testFailureResultCreation() {
        let result = FormatSampleResult(
            caption: "",
            lyrics: "",
            bpm: nil,
            duration: nil,
            keyscale: "",
            language: "",
            timesignature: "",
            statusMessage: "LLM not initialized",
            success: false,
            error: "LLM not initialized"
        )
        XCTAssertFalse(result.success)
        XCTAssertEqual(result.error, "LLM not initialized")
    }
}
