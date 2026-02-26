import XCTest
@testable import AceStepSwift

final class ConstantsTests: XCTestCase {

    func testBPMMinMax() {
        XCTAssertEqual(AceStepConstants.bpmMin, 30)
        XCTAssertEqual(AceStepConstants.bpmMax, 300)
    }

    func testDurationMinMax() {
        XCTAssertEqual(AceStepConstants.durationMin, 10)
        XCTAssertEqual(AceStepConstants.durationMax, 600)
    }

    func testValidTimeSignatures() {
        XCTAssertEqual(AceStepConstants.validTimeSignatures, [2, 3, 4, 6])
    }

    func testValidLanguagesContainsEnAndUnknown() {
        XCTAssertTrue(AceStepConstants.validLanguages.contains("en"))
        XCTAssertTrue(AceStepConstants.validLanguages.contains("unknown"))
    }
}
