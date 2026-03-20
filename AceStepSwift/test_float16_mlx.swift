import Foundation
import CoreML

func run() {
    let shape: [NSNumber] = [1, 2, 3]
    let m = try! MLMultiArray(shape: shape, dataType: .float16)
    
    let count = m.count
    let mShape = m.shape.map { $0.intValue }
    let pointer = m.dataPointer.bindMemory(to: Float16.self, capacity: count)
    let buffer = UnsafeBufferPointer(start: pointer, count: count)
    let array = Array(buffer)
    print("Test passed: shape \(mShape), array count \(array.count)")
}
run()
