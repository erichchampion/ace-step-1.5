/**
 VAE decode contract: latent [B, T, 64] → audio waveform.
 Python uses downsampling 2×4×4×8×8 = 2048, so audio length = T * 2048.
 */

import Foundation
import MLX

/// Downsampling factor from latent time steps to audio samples (2*4*4*5*6).
public let vaeLatentToSamplesFactor = 1920

/// Protocol for VAE decode: latents [B, T, C] (C=64) → audio [B, T*1920].
public protocol VAEDecoder: AnyObject {
    func decode(latent: MLXArray) -> MLXArray
}
