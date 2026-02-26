/**
 Tests for diffusion step contract: DiffusionStepper and FakeDiffusionStepper.
 Requires MLX (Metal) to create arrays and assert shapes.
 */

import XCTest
import MLX
@testable import AceStepSwift

final class DiffusionContractTests: XCTestCase {

    /// FakeDiffusionStepper.step returns array with same shape as currentLatent.
    func testFakeDiffusionStepperReturnsSameShapeAsInput() {
        let stepper = FakeDiffusionStepper()
        let b = 2
        let t = 10
        let c = 64
        let currentLatent = MLXArray((0 ..< (b * t * c)).map { Float($0) }, [b, t, c])
        let conditions = DiTConditions()
        let next = stepper.step(currentLatent: currentLatent, timestep: 0.5, conditions: conditions, nextTimestep: nil)
        XCTAssertEqual(next.shape, [b, t, c])
    }
}
