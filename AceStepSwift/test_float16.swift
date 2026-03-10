import Foundation
import CoreML

@available(macOS 11.0, *)
func test(multiArray: MLMultiArray) {
    let count = multiArray.count
    let shape = multiArray.shape.map { $0.intValue }
    let pointer = multiArray.dataPointer.bindMemory(to: Float16.self, capacity: count)
    let buffer = UnsafeBufferPointer(start: pointer, count: count)
    let array = Array(buffer)
    print(array.count)
}
