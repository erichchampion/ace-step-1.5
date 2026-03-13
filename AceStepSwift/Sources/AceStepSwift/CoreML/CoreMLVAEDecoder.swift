import Foundation
import CoreML
import MLX


/// CoreML implementation of VAEDecoder that runs the .mlpackage VAE models.
public final class CoreMLVAEDecoder: VAEDecoder {
    private let model: MLModel

    public init(modelURL: URL) async throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU // Bypass ANE for VAE due to compilation hang with conv_transpose respecialization
        let compiledURL = try await CoreMLHelper.compileIfNeeded(modelURL: modelURL)
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
    }

    public func decode(latent: MLXArray) -> MLXArray {
        // Core ML VAE expects:
        // latents: [1, 64, seq]
        // Output: audio [1, 2, seq*2048]
        
        let batch = latent.dim(0)
        let seq = latent.dim(1)
        let channels = latent.dim(2)
        
        guard batch == 1 else {
            fatalError("CoreMLVAEDecoder only supports batch size 1")
        }
        guard channels == 64 else {
            fatalError("CoreMLVAEDecoder expects 64 latent channels")
        }
        
        do {
            // MLX NLC [1, T, 64] -> CoreML NCL [1, 64, T]
            // We can transpose in MLX first for efficiency.
            // IMPORTANT: Ensure the array is contiguous before calling asArray.
            let latentNCL = latent.transposed(axes: [0, 2, 1]).contiguous()
            let latentData = latentNCL.asArray(Float.self)
            
            let latentShaped = MLShapedArray<Float>(scalars: latentData, shape: [1, 64, seq])
            
            // Prepare inputs
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
                "latents": MLMultiArray(latentShaped)
            ])
            
            // Run prediction
            let output = try model.prediction(from: inputProvider)
            
            // Extract audio
            guard let audioMultiArray = output.featureValue(for: "audio")?.multiArrayValue else {
                fatalError("Failed to extract audio from Core ML output")
            }
            
            // Convert back to MLXArray [B, L, C] (matches MLX pipeline expectation usually [1, L, 2])
            let shape = audioMultiArray.shape.map { $0.intValue }
            
            let mlxWav: MLXArray
            if audioMultiArray.dataType == .float16 {
                let count = audioMultiArray.count
                var floats = [Float](repeating: 0, count: count)
                let f16Ptr = audioMultiArray.dataPointer.bindMemory(to: Float16.self, capacity: count)
                for i in 0..<count {
                    floats[i] = Float(f16Ptr[i])
                }
                mlxWav = MLXArray(floats, shape)
            } else {
                let audioShaped = MLShapedArray<Float>(audioMultiArray)
                mlxWav = MLXArray(audioShaped.scalars, shape)
            }
            
            // If output is [1, 2, L], we want [1, L, 2] usually for audio players
            return mlxWav.transposed(axes: [0, 2, 1]).contiguous()
            
        } catch {
            print("[CoreMLVAEDecoder] Prediction failed: \(error)")
            return MLXArray.zeros([batch, seq * 1920, 2]) // Fallback
        }
    }
}
