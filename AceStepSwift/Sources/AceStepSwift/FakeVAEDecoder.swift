/**
 Fake VAE decoder that returns zeros with shape [B, T*2048] from latent [B, T, 64].
 Used to satisfy the VAEDecoder contract in tests without a real VAE model.
 */

import Foundation
import MLX

/// Fake decoder returning zeros of the correct output shape.
public final class FakeVAEDecoder: VAEDecoder {
    public init() {}

    public func decode(latent: MLXArray) -> MLXArray {
        let shape = latent.shape
        guard shape.count >= 2 else {
            return MLXArray([Float(0)], [1])
        }
        let b = shape[0]
        let t = shape[1]
        let samples = t * vaeLatentToSamplesFactor
        let count = b * samples
        return MLXArray([Float](repeating: 0, count: count), [b, samples])
    }
}
