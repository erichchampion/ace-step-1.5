/**
 Real VAE decoder conforming to VAEDecoder. Load weights from safetensors (Python vae_convert output).
 */

import Foundation
import MLX
import MLXNN

/// Real VAE decoder: loads weights and decodes latent [B, T, 64] -> audio [B, L_audio, C_audio].
public final class MLXVAEDecoder: VAEDecoder {
    private let model: MLXAutoEncoderOobleck
    private var loaded: Bool = false

    public init(
        decoderChannels: Int = 128,
        decoderInputChannels: Int = 64,
        audioChannels: Int = 2,
        downsamplingRatios: [Int]? = nil,
        channelMultiples: [Int]? = nil
    ) {
        self.model = MLXAutoEncoderOobleck(
            encoderHiddenSize: 128,
            decoderChannels: decoderChannels,
            decoderInputChannels: decoderInputChannels,
            audioChannels: audioChannels,
            downsamplingRatios: downsamplingRatios,
            channelMultiples: channelMultiples
        )
    }

    /// Load decoder weights from a safetensors file (decoder.* keys). Use Python vae_convert to produce the file.
    public func loadWeights(from url: URL) throws {
        let params = try loadParameters(from: url)
        model.update(parameters: params)
        loaded = true
    }

    public func decode(latent: MLXArray) -> MLXArray {
        model.decode(latentsNLC: latent)
    }
}
