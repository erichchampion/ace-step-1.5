import XCTest
@testable import AceStepSwift

final class GenerationResultTests: XCTestCase {

    func testSuccessResultDefaults() {
        let result = GenerationResult()
        XCTAssertTrue(result.success)
        XCTAssertTrue(result.audios.isEmpty)
        XCTAssertNil(result.error)
    }

    func testFailureResult() {
        let result = GenerationResult(
            audios: [],
            statusMessage: "Model not loaded",
            extraOutputs: [:],
            success: false,
            error: "Model not loaded"
        )
        XCTAssertFalse(result.success)
        XCTAssertEqual(result.error, "Model not loaded")
    }
}
