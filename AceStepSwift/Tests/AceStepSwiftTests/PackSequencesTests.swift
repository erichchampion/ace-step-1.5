/**
 Tests for packSequences: output shape, valid-first ordering, L1=0/L2=0 edges.
 Mirrors Python pack_sequences contract (concat, sort by mask descending, new_mask from lengths).
 */

import XCTest
import MLX
@testable import AceStepSwift

final class PackSequencesTests: XCTestCase {

    func testPackSequencesOutputShape() {
        let b = 2
        let L1 = 3
        let L2 = 2
        let d = 4
        let hidden1 = MLXArray.zeros([b, L1, d])
        let hidden2 = MLXArray.zeros([b, L2, d])
        let mask1 = MLXArray.ones([b, L1])
        let mask2 = MLXArray.ones([b, L2])
        let (packed, newMask) = packSequences(hidden1: hidden1, hidden2: hidden2, mask1: mask1, mask2: mask2)
        XCTAssertEqual(packed.dim(0), b)
        XCTAssertEqual(packed.dim(1), L1 + L2)
        XCTAssertEqual(packed.dim(2), d)
        XCTAssertEqual(newMask.dim(0), b)
        XCTAssertEqual(newMask.dim(1), L1 + L2)
    }

    func testPackSequencesValidFirstNewMask() {
        let b = 1
        let L1 = 3
        let L2 = 2
        let d = 2
        let hidden1 = MLXArray((0..<(b * L1 * d)).map { Float($0) }, [b, L1, d])
        let hidden2 = MLXArray((100..<(100 + b * L2 * d)).map { Float($0) }, [b, L2, d])
        let mask1 = MLXArray([Float(1), 1, 0], [b, L1])
        let mask2 = MLXArray([Float(1), 0], [b, L2])
        let (_, newMask) = packSequences(hidden1: hidden1, hidden2: hidden2, mask1: mask1, mask2: mask2)
        newMask.eval()
        let maskF = newMask.asType(.float32)
        // mask1 [1,1,0], mask2 [1,0] → 3 valid total; packed valid-first so newMask = [1,1,1,0,0].
        XCTAssertEqual(maskF[0..<1, 0..<1].squeezed(axis: 0).squeezed(axis: 0).item(Float.self), 1.0)
        XCTAssertEqual(maskF[0..<1, 1..<2].squeezed(axis: 0).squeezed(axis: 0).item(Float.self), 1.0)
        XCTAssertEqual(maskF[0..<1, 2..<3].squeezed(axis: 0).squeezed(axis: 0).item(Float.self), 1.0)
        XCTAssertEqual(maskF[0..<1, 3..<4].squeezed(axis: 0).squeezed(axis: 0).item(Float.self), 0.0)
        XCTAssertEqual(maskF[0..<1, 4..<5].squeezed(axis: 0).squeezed(axis: 0).item(Float.self), 0.0)
    }

    func testPackSequencesL1ZeroReturnsHidden2AndMask2() {
        let b = 2
        let L2 = 3
        let d = 4
        let hidden1 = MLXArray.zeros([b, 0, d])
        let hidden2 = MLXArray.ones([b, L2, d])
        let mask1 = MLXArray.zeros([b, 0])
        let mask2 = MLXArray.ones([b, L2])
        let (packed, newMask) = packSequences(hidden1: hidden1, hidden2: hidden2, mask1: mask1, mask2: mask2)
        XCTAssertEqual(packed.shape, hidden2.shape)
        XCTAssertEqual(newMask.shape, mask2.shape)
    }

    func testPackSequencesL2ZeroReturnsHidden1AndMask1() {
        let b = 1
        let L1 = 4
        let d = 2
        let hidden1 = MLXArray.ones([b, L1, d])
        let hidden2 = MLXArray.zeros([b, 0, d])
        let mask1 = MLXArray.ones([b, L1])
        let mask2 = MLXArray.zeros([b, 0])
        let (packed, newMask) = packSequences(hidden1: hidden1, hidden2: hidden2, mask1: mask1, mask2: mask2)
        XCTAssertEqual(packed.shape, hidden1.shape)
        XCTAssertEqual(newMask.shape, mask1.shape)
    }
}
