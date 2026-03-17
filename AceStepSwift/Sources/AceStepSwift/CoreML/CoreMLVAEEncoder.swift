import Foundation
import CoreML
import MLX


/// CoreML implementation of VAE Encoder that runs the .mlpackage VAE encoder models.
/// Converts raw stereo audio → latent space representation for reference/source audio encoding.
public final class CoreMLVAEEncoder {
    private let model: MLModel

    public init(modelURL: URL) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        let compiledURL = try await CoreMLHelper.compileIfNeeded(modelURL: modelURL)
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
    }

    /// Encode raw stereo audio into latent space.
    /// - Parameter audio: MLXArray of shape `[1, 2, T]` (batch=1, stereo, time samples at 48kHz)
    /// - Returns: MLXArray of shape `[1, T', 64]` (NLC format: batch, latent frames, channels)
    public func encode(audio: MLXArray) -> MLXArray {
        // CoreML encoder expects: audio [1, 2, T]
        // Output: latent_mean [1, 64, T']

        let batch = audio.dim(0)
        let channels = audio.dim(1)
        let samples = audio.dim(2)

        guard batch == 1 else {
            fatalError("CoreMLVAEEncoder only supports batch size 1")
        }
        guard channels == 2 else {
            fatalError("CoreMLVAEEncoder expects stereo audio (2 channels), got \(channels)")
        }

        do {
            // Audio is already in NCL [1, 2, T] format for CoreML
            let audioContiguous = audio.contiguous()
            let audioData = audioContiguous.asArray(Float.self)

            let audioShaped = MLShapedArray<Float>(scalars: audioData, shape: [1, 2, samples])

            // Prepare inputs
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
                "audio": MLMultiArray(audioShaped)
            ])

            // Run prediction
            let output = try model.prediction(from: inputProvider)

            // Extract latent_mean
            guard let latentMultiArray = output.featureValue(for: "latent_mean")?.multiArrayValue else {
                fatalError("Failed to extract latent_mean from Core ML output")
            }

            // Convert back to MLXArray
            let shape = latentMultiArray.shape.map { $0.intValue }  // [1, 64, T']

            let mlxLatent: MLXArray
            if latentMultiArray.dataType == .float16 {
                let count = latentMultiArray.count
                var floats = [Float](repeating: 0, count: count)
                let f16Ptr = latentMultiArray.dataPointer.bindMemory(to: Float16.self, capacity: count)
                for i in 0..<count {
                    floats[i] = Float(f16Ptr[i])
                }
                mlxLatent = MLXArray(floats, shape)
            } else {
                let latentShaped = MLShapedArray<Float>(latentMultiArray)
                mlxLatent = MLXArray(latentShaped.scalars, shape)
            }

            // CoreML output is NCL [1, 64, T'] → transpose to NLC [1, T', 64]
            let result = mlxLatent.transposed(axes: [0, 2, 1]).contiguous()
            let mean = result.mean().item(Float.self)
            print("[CoreMLVAEEncoder] ✅ Encoded: shape=\(result.shape), mean=\(mean)")
            return result

        } catch {
            print("[CoreMLVAEEncoder] ❌ Prediction FAILED: \(error)")
            print("[CoreMLVAEEncoder] Input shape: [1, 2, \(samples)] — returning zeros as fallback")
            // Return zeros as fallback
            let latentFrames = samples / 1920  // Approximate downsampling ratio
            return MLXArray.zeros([batch, max(1, latentFrames), 64])
        }
    }
}
