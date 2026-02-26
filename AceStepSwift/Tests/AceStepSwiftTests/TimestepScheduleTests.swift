import XCTest
@testable import AceStepSwift

final class TimestepScheduleTests: XCTestCase {

    func testShift1LegacySchedule() {
        let schedule = DiffusionSchedule.getTimestepSchedule(shift: 1.0, timesteps: nil, inferSteps: nil)
        let expected: [Double] = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
        XCTAssertEqual(schedule.count, expected.count)
        for (i, e) in expected.enumerated() {
            XCTAssertEqual(schedule[i], e, accuracy: 1e-10, "index \(i)")
        }
    }

    func testShift2LegacySchedule() {
        let schedule = DiffusionSchedule.getTimestepSchedule(shift: 2.0, timesteps: nil, inferSteps: nil)
        let expected: [Double] = [
            1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693,
            0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222
        ]
        XCTAssertEqual(schedule.count, expected.count)
        for (i, e) in expected.enumerated() {
            XCTAssertEqual(schedule[i], e, accuracy: 1e-10, "index \(i)")
        }
    }

    func testShift3LegacySchedule() {
        let schedule = DiffusionSchedule.getTimestepSchedule(shift: 3.0, timesteps: nil, inferSteps: nil)
        let expected: [Double] = [
            1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75,
            0.6428571428571429, 0.5, 0.3
        ]
        XCTAssertEqual(schedule.count, expected.count)
        for (i, e) in expected.enumerated() {
            XCTAssertEqual(schedule[i], e, accuracy: 1e-10, "index \(i)")
        }
    }

    func testInferStepsOverridesLegacy() {
        let schedule = DiffusionSchedule.getTimestepSchedule(shift: 3.0, timesteps: nil, inferSteps: 4)
        XCTAssertEqual(schedule.count, 4)
        // raw = [1.0 - i/4 for i in 0..<4] = [1.0, 0.75, 0.5, 0.25]; then t' = 3*t/(1+2t)
        XCTAssertEqual(schedule[0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(schedule[3], 0.5, accuracy: 1e-10)  // 0.25 -> 3*0.25/(1+0.5)=0.5
    }

    func testShift1WithInferSteps() {
        let schedule = DiffusionSchedule.getTimestepSchedule(shift: 1.0, timesteps: nil, inferSteps: 8)
        XCTAssertEqual(schedule.count, 8)
        XCTAssertEqual(schedule[0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(schedule[7], 0.125, accuracy: 1e-10)
    }

    func testCustomTimestepsMappedToValid() {
        let custom: [Double] = [1.0, 0.9, 0.5, 0.25]
        let schedule = DiffusionSchedule.getTimestepSchedule(shift: 2.0, timesteps: custom, inferSteps: nil)
        XCTAssertEqual(schedule.count, 4)
        // Each value should be snapped to nearest VALID_TIMESTEPS
        XCTAssertEqual(schedule[0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(schedule[1], 0.9, accuracy: 1e-10)
        XCTAssertEqual(schedule[2], 0.5, accuracy: 1e-10)
        XCTAssertEqual(schedule[3], 0.25, accuracy: 1e-10)
    }

    func testCustomTimestepsTrailingZerosRemoved() {
        let custom: [Double] = [1.0, 0.5, 0.0]
        let schedule = DiffusionSchedule.getTimestepSchedule(shift: 1.0, timesteps: custom, inferSteps: nil)
        XCTAssertEqual(schedule.count, 2)
        XCTAssertEqual(schedule[0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(schedule[1], 0.5, accuracy: 1e-10)
    }

    func testShiftRoundedToNearestValid() {
        // 2.5 rounds to 2.0 or 3.0 (nearest)
        let schedule = DiffusionSchedule.getTimestepSchedule(shift: 2.5, timesteps: nil, inferSteps: nil)
        XCTAssertEqual(schedule.count, 8)
    }
}
