import XCTest
import MLX
@testable import AceStepSwift

final class ContiguityTests: XCTestCase {
    func testArrayContiguity() {
        // [2, 3] layout
        let a = MLXArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        // Transpose to [3, 2]
        let b = a.transposed(axes: [1, 0])
        let c = b.contiguous()
        
        // No MLX.eval(c) called!
        let arrC = c.asArray(Float.self)
        let arrB = b.asArray(Float.self)
        
        print("arrC_Contiguous: \(arrC)")
        print("arrB_Transposed: \(arrB)")
    }
}
