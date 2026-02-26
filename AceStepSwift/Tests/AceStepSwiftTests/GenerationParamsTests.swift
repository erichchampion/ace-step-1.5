import XCTest
@testable import AceStepSwift

final class GenerationParamsTests: XCTestCase {

    func testDefaultTaskTypeIsTextToMusic() {
        let params = GenerationParams()
        XCTAssertEqual(params.taskType, "text2music")
    }

    func testDefaultLegoTrackIsNil() {
        let params = GenerationParams()
        XCTAssertNil(params.legoTrack)
    }

    func testDefaultCompleteTrackClassesIsNil() {
        let params = GenerationParams()
        XCTAssertNil(params.completeTrackClasses)
    }

    func testDefaultCaptionAndLyricsAreEmpty() {
        let params = GenerationParams()
        XCTAssertEqual(params.caption, "")
        XCTAssertEqual(params.lyrics, "")
    }

    func testDefaultDurationIsNegativeOne() {
        let params = GenerationParams()
        XCTAssertEqual(params.duration, -1.0)
    }

    func testDefaultInferenceStepsIs8() {
        let params = GenerationParams()
        XCTAssertEqual(params.inferenceSteps, 8)
    }

    func testDefaultSeedIsNegativeOne() {
        let params = GenerationParams()
        XCTAssertEqual(params.seed, -1)
    }

    func testParamsAcceptValidBPMRange() {
        var params = GenerationParams()
        params.bpm = 30
        XCTAssertEqual(params.bpm, 30)
        params.bpm = 300
        XCTAssertEqual(params.bpm, 300)
        params.bpm = 120
        XCTAssertEqual(params.bpm, 120)
    }

    func testParamsAcceptValidDurationRange() {
        var params = GenerationParams()
        params.duration = 10.0
        XCTAssertEqual(params.duration, 10.0)
        params.duration = 600.0
        XCTAssertEqual(params.duration, 600.0)
    }

    func testIsBPMValid() {
        XCTAssertTrue(GenerationParams.isBPMValid(30))
        XCTAssertTrue(GenerationParams.isBPMValid(300))
        XCTAssertTrue(GenerationParams.isBPMValid(120))
        XCTAssertFalse(GenerationParams.isBPMValid(29))
        XCTAssertFalse(GenerationParams.isBPMValid(301))
    }

    func testIsDurationValid() {
        XCTAssertTrue(GenerationParams.isDurationValid(10.0))
        XCTAssertTrue(GenerationParams.isDurationValid(600.0))
        XCTAssertTrue(GenerationParams.isDurationValid(60.0))
        XCTAssertFalse(GenerationParams.isDurationValid(9.9))
        XCTAssertFalse(GenerationParams.isDurationValid(600.1))
    }
}
