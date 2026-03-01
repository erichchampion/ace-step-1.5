import XCTest
import MLX
@testable import AceStepSwift

final class MLXAutoEncoderOobleckTests: XCTestCase {

    override func setUp() {
        super.setUp()
        MLX.GPU.set(cacheLimit: 0)
    }

    func testEncodeMean() throws {
        let encoder = MLXAutoEncoderOobleck()
        // Default downsampling ratios are [2, 4, 4, 6, 10], which multiply to 1920.
        // We use a length of 3840 to ensure it divides evenly and produces an output length of 2.
        let input = MLXArray.zeros([1, 3840, 2])
        
        let mean = encoder.encodeMean(audioNLC: input)
        // With default parameters, decoderInputChannels is 64.
        XCTAssertEqual(mean.shape, [1, 2, 64])
    }

    func testEncodeAndSample() throws {
        let encoder = MLXAutoEncoderOobleck()
        let input = MLXArray.zeros([1, 3840, 2])
        
        let sample = encoder.encodeAndSample(audioNLC: input)
        // The sampled output should have the same shape as the mean.
        XCTAssertEqual(sample.shape, [1, 2, 64])
    }
}
