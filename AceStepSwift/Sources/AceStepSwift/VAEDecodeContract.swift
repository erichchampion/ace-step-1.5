/**
 VAE decode contract: latent [B, T, 64] → audio waveform.
 Python uses downsampling 2×4×4×8×8 = 2048, so audio length = T * 2048.
 */

import Foundation
import MLX

/// Downsampling factor from latent time steps to audio samples (2*4*4*8*8).
public let vaeLatentToSamplesFactor = 2048

/// Protocol for VAE decode: latents [B, T, C] (C=64) → audio [B, T*2048].
public protocol VAEDecoder: AnyObject {
    func decode(latent: MLXArray) -> MLXArray
}
