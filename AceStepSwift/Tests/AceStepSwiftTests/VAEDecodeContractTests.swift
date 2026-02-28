/**
 Tests for VAE decode contract: VAEDecoder and FakeVAEDecoder.
 Requires MLX (Metal) to create arrays and assert shapes.
 */

import XCTest
import MLX
@testable import AceStepSwift

final class VAEDecodeContractTests: XCTestCase {

    private static let vaeWeightsPathEnv = "VAE_WEIGHTS_PATH"

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

    /// Test MLXVAEDecoder with real weights loads and decodes correctly.
    func testMLXVAEDecoderWithRealWeights() throws {
        guard let path = ProcessInfo.processInfo.environment[Self.vaeWeightsPathEnv]?.trimmingCharacters(in: .whitespaces),
              !path.isEmpty else {
            try XCTSkipIf(true, "Set \(Self.vaeWeightsPathEnv) to run VAE with real weights")
            return
        }
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            try XCTSkipIf(true, "VAE weights not found at \(url.path)")
            return
        }

        let decoder = MLXVAEDecoder()
        try decoder.loadWeights(from: url)

        // Test decode with known input shape
        let b = 1
        let t = 10
        let c = 64
        let latent = MLXArray.zeros([b, t, c])
        let audio = decoder.decode(latent: latent)

        // VAE upscales by 2048 factor
        XCTAssertEqual(audio.dim(0), b)
        XCTAssertEqual(audio.dim(1), t * vaeLatentToSamplesFactor)
    }

    /// Test MLXVAEDecoder output shape is deterministic with same input.
    func testMLXVAEDecoderDeterministic() throws {
        guard let path = ProcessInfo.processInfo.environment[Self.vaeWeightsPathEnv]?.trimmingCharacters(in: .whitespaces),
              !path.isEmpty else {
            try XCTSkipIf(true, "Set \(Self.vaeWeightsPathEnv) to test VAE determinism")
            return
        }
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            try XCTSkipIf(true, "VAE weights not found")
            return
        }

        let decoder = MLXVAEDecoder()
        try decoder.loadWeights(from: url)

        let b = 1
        let t = 5
        let latent = MLXArray.zeros([b, t, 64])

        let audio1 = decoder.decode(latent: latent)
        let audio2 = decoder.decode(latent: latent)

        let close = audio1.allClose(audio2).all().item(Bool.self)
        XCTAssertTrue(close, "VAE decode should be deterministic")
    }
}
