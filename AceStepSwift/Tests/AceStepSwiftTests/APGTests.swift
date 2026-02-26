/**
 Unit tests for APG (Adaptive Projected Guidance). Requires MLX for array creation.
 */

import XCTest
import MLX
@testable import AceStepSwift

final class APGTests: XCTestCase {

    /// Output shape equals predCond shape [B, T, C].
    func testAPGForwardOutputShapeMatchesInput() {
        let b = 2
        let t = 5
        let c = 4
        let predCond = MLXArray((0 ..< (b * t * c)).map { Float($0) }, [b, t, c])
        let predUncond = MLXArray((0 ..< (b * t * c)).map { Float($0) }, [b, t, c])
        let out = apgForward(
            predCond: predCond,
            predUncond: predUncond,
            guidanceScale: 2.0,
            normThreshold: 2.5
        )
        XCTAssertEqual(out.shape, [b, t, c])
    }

    /// At guidanceScale == 1, output equals predCond (no guidance).
    func testAPGForwardAtGuidanceScaleOneReturnsPredCond() {
        let b = 1
        let t = 3
        let c = 4
        let predCond = MLXArray((0 ..< (b * t * c)).map { Float($0) }, [b, t, c])
        let predUncond = MLXArray((1 ..< (b * t * c + 1)).map { Float($0) }, [b, t, c])
        let out = apgForward(
            predCond: predCond,
            predUncond: predUncond,
            guidanceScale: 1.0,
            normThreshold: 2.5
        )
        let close = out.allClose(predCond).all().item(Bool.self)
        XCTAssertTrue(close, "At guidanceScale 1, output should equal predCond")
    }

    /// Overload without momentum produces same shape.
    func testAPGForwardWithoutMomentumSameShape() {
        let predCond = MLXArray((0 ..< 24).map { Float($0) }, [2, 3, 4])
        let predUncond = MLXArray((0 ..< 24).map { Float($0) }, [2, 3, 4])
        let out = apgForward(
            predCond: predCond,
            predUncond: predUncond,
            guidanceScale: 3.0,
            normThreshold: 2.5
        )
        XCTAssertEqual(out.shape, predCond.shape)
    }
}
