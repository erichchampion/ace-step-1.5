import XCTest
@testable import AceStepSwift

final class CreateRandomSampleTests: XCTestCase {

    func testCreateRandomSampleSimpleModeReturnsSuccessAndNonEmptyCaption() {
        let result = AceStepEngine.createRandomSample(sampleType: "simple_mode")
        XCTAssertTrue(result.success, "create_random_sample simple_mode should succeed when examples exist")
        XCTAssertFalse(result.caption.isEmpty, "caption/description should be non-empty")
        XCTAssertNil(result.error)
    }

    func testCreateRandomSampleCustomModeReturnsSuccessAndExpectedFields() {
        let result = AceStepEngine.createRandomSample(sampleType: "custom_mode")
        XCTAssertTrue(result.success)
        XCTAssertFalse(result.caption.isEmpty)
        XCTAssertNotNil(result.bpm)
        XCTAssertNotNil(result.duration)
        XCTAssertFalse(result.keyscale.isEmpty)
        XCTAssertFalse(result.timesignature.isEmpty)
        XCTAssertNil(result.error)
    }

    func testCreateRandomSampleZeroExamplesReturnsFailure() {
        let result = RandomSampleStore.pickRandom(from: [], sampleTypeLabel: "empty_mode")
        XCTAssertFalse(result.success)
        XCTAssertNotNil(result.error)
        XCTAssertTrue(result.error?.contains("No example data available") ?? false)
    }

    func testCreateRandomSampleDefaultSampleTypeUsesSimpleMode() {
        let result = AceStepEngine.createRandomSample()
        XCTAssertTrue(result.success)
        XCTAssertFalse(result.caption.isEmpty)
    }

    func testCreateRandomSampleUnknownSampleTypeFallsBackToSimpleMode() {
        let result = AceStepEngine.createRandomSample(sampleType: "unknown_type")
        XCTAssertTrue(result.success)
        XCTAssertFalse(result.caption.isEmpty)
    }
}
