/**
 Tests for VAE decode contract: VAEDecoder and FakeVAEDecoder.
 Requires MLX (Metal) to create arrays and assert shapes.
 */

import XCTest
import MLX
@testable import AceStepSwift

final class VAEDecodeContractTests: XCTestCase {

    /// FakeVAEDecoder.decode returns [B, T*2048] for latent [B, T, 64].
    func testFakeVAEDecoderOutputShape() {
        let decoder = FakeVAEDecoder()
        let b = 1
        let t = 5
        let c = 64
        let latent = MLXArray((0 ..< (b * t * c)).map { Float($0) }, [b, t, c])
        let audio = decoder.decode(latent: latent)
        XCTAssertEqual(audio.shape[0], b)
        XCTAssertEqual(audio.shape[1], t * vaeLatentToSamplesFactor)
    }
}
