/**
 VAE encode/decode. Mirrors vae_model.MLXAutoEncoderOobleck.
 Encoder output is 2*decoderInputChannels (mean + log_scale); decode path unchanged.
 */

import Foundation
import MLX
import MLXNN

public class MLXAutoEncoderOobleck: Module {
    public let encoder: VAEOobleckEncoder
    public let decoder: VAEOobleckDecoder

    public init(
        encoderHiddenSize: Int = 128,
        decoderChannels: Int = 128,
        decoderInputChannels: Int = 64,
        audioChannels: Int = 2,
        downsamplingRatios: [Int]? = nil,
        channelMultiples: [Int]? = nil
    ) {
        let ratios = downsamplingRatios ?? [2, 4, 4, 6, 10]
        let multiples = channelMultiples ?? [1, 2, 4, 8, 16]
        self.encoder = VAEOobleckEncoder(
            encoderHiddenSize: encoderHiddenSize,
            audioChannels: audioChannels,
            downsamplingRatios: ratios,
            channelMultiples: multiples
        )
        self.decoder = VAEOobleckDecoder(
            channels: decoderChannels,
            inputChannels: decoderInputChannels,
            audioChannels: audioChannels,
            upsamplingRatios: ratios.reversed(),
            channelMultiples: multiples
        )
        super.init()
    }

    private func getMeanAndScale(audioNLC: MLXArray) -> (MLXArray, MLXArray) {
        let h = encoder(audioNLC)
        let parts = split(h, parts: 2, axis: -1)
        return (parts[0], parts[1])
    }

    /// Encode audio -> mean latent (no sampling). [B, L_audio, C_audio] -> [B, L_latent, C_latent]. NLC.
    public func encodeMean(audioNLC: MLXArray) -> MLXArray {
        return getMeanAndScale(audioNLC: audioNLC).0
    }

    /// Encode audio -> sample latent (mean + std * noise). [B, L_audio, C_audio] -> [B, L_latent, C_latent]. NLC.
    public func encodeAndSample(audioNLC: MLXArray) -> MLXArray {
        let (mean, scale) = getMeanAndScale(audioNLC: audioNLC)
        let logStdClampThreshold: Float = 20.0
        let epsilon: Float = 1e-4
        let std = MLX.where(scale .> logStdClampThreshold, scale, MLX.log(1 + MLX.exp(scale)) + epsilon)
        let noise = MLXRandom.normal(mean.shape)
        return mean + std * noise
    }

    /// Decode latents [B, L_latent, C_latent] -> audio [B, L_audio, C_audio]. NLC format.
    public func decode(latentsNLC: MLXArray) -> MLXArray {
        decoder(latentsNLC)
    }
}
