import XCTest
@testable import AceStepSwift

final class RepaintSpanTests: XCTestCase {

    func testRepaintSpanMidRegion() {
        // 2 seconds to 5 seconds at 48000 Hz, max 250 latents
        let span = RepaintSpan(
            repaintingStartSec: 2.0,
            repaintingEndSec: 5.0,
            sampleRate: 48000,
            maxLatentLength: 250
        )
        XCTAssertTrue(span.isValid)
        // 2 * 48000 / 1920 = 50, 5 * 48000 / 1920 = 125
        XCTAssertEqual(span.startLatent, 50)
        XCTAssertEqual(span.endLatent, 125)
    }

    func testRepaintSpanClampedToMaxLatentLength() {
        let span = RepaintSpan(
            repaintingStartSec: 0,
            repaintingEndSec: 100,
            sampleRate: 48000,
            maxLatentLength: 100
        )
        XCTAssertTrue(span.isValid)
        XCTAssertEqual(span.startLatent, 0)
        XCTAssertEqual(span.endLatent, 100)
    }

    func testRepaintSpanUntilEndWhenEndNegative() {
        let span = RepaintSpan(
            repaintingStartSec: 1.0,
            repaintingEndSec: -1,
            sampleRate: 48000,
            maxLatentLength: 125
        )
        // end is derived from maxLatentLength: 125 * 1920 / 48000 = 5 sec
        XCTAssertTrue(span.isValid)
        XCTAssertEqual(span.startLatent, 25)  // 1 * 48000 / 1920
        XCTAssertEqual(span.endLatent, 125)
    }

    func testRepaintSpanInvalidWhenEndBeforeStart() {
        let span = RepaintSpan(
            repaintingStartSec: 5.0,
            repaintingEndSec: 2.0,
            sampleRate: 48000,
            maxLatentLength: 250
        )
        XCTAssertFalse(span.isValid, "end_sec < start_sec should yield invalid span")
    }

    func testRepaintSamplesPerLatentStepConstant() {
        XCTAssertEqual(repaintSamplesPerLatentStep, 1920)
    }
}
