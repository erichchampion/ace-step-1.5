import Foundation
import CoreML
import MLX


/// CoreML implementation of DiffusionStepper that runs the .mlpackage DiT models.
public final class CoreMLDiTStepper: DiffusionStepper {
    private let model: MLModel
    private let d: Int // audio_acoustic_hidden_dim, usually 64

    // Cached per-sequence-length prediction arrays to avoid re-allocating every step
    private var cachedSeqLen: Int = 0
    private var cachedMaskShaped: MLShapedArray<Float>?
    private var cachedPosShaped: MLShapedArray<Int32>?
    private var cachedCacheShaped: MLShapedArray<Int32>?
    private var cachedSWMaskShaped: MLShapedArray<Float>?

    public init(modelURL: URL, d: Int = 64) async throws {
        let config = CoreMLConfigFactory.makeConfig(computeUnits: .cpuAndGPU) // Bypass ANE due to ios17.mul broadcasting crash with RangeDim
        let compiledURL = try await CoreMLHelper.compileIfNeeded(modelURL: modelURL)
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        self.d = d
    }

    /// Clear cached prediction arrays to free memory after generation.
    public func clearPredictionCache() {
        cachedSeqLen = 0
        cachedMaskShaped = nil
        cachedPosShaped = nil
        cachedCacheShaped = nil
        cachedSWMaskShaped = nil
    }

    /// Build or retrieve cached per-sequence-length arrays.
    private func ensureCachedArrays(batch: Int, seq: Int) {
        guard seq != cachedSeqLen else { return }
        cachedSeqLen = seq

        cachedMaskShaped = MLShapedArray<Float>(scalars: [Float](repeating: 1.0, count: seq), shape: [batch, seq])

        let downSeq = seq / 2
        let positionsData = (0..<downSeq).map { Int32($0) }
        cachedPosShaped = MLShapedArray<Int32>(scalars: positionsData, shape: [batch, downSeq])
        cachedCacheShaped = MLShapedArray<Int32>(scalars: positionsData, shape: [downSeq])

        // Build sliding window mask [1, 1, downSeq, downSeq]
        let slidingWindow = 128
        var swMaskData = [Float](repeating: 0, count: downSeq * downSeq)
        for i in 0..<downSeq {
            for j in 0..<downSeq {
                swMaskData[i * downSeq + j] = abs(i - j) <= slidingWindow ? 0.0 : -65500.0
            }
        }
        cachedSWMaskShaped = MLShapedArray<Float>(scalars: swMaskData, shape: [1, 1, downSeq, downSeq])
    }

    public func predictVelocity(currentLatent: MLXArray, timestep: Float, conditions: DiTConditions, useCache: Bool) -> MLXArray {
        let batch = currentLatent.dim(0)
        let seq = currentLatent.dim(1)
        
        guard batch == 1 else {
            fatalError("CoreMLDiTStepper only supports batch size 1")
        }
        
        do {
            let latentData = currentLatent.contiguous().asArray(Float.self)
            let latentShaped = MLShapedArray<Float>(scalars: latentData, shape: [batch, seq, d])
            
            ensureCachedArrays(batch: batch, seq: seq)
            
            let tShaped = MLShapedArray<Float>(scalars: [timestep], shape: [1])
            let trShaped = MLShapedArray<Float>(scalars: [timestep], shape: [1])
            
            guard let encHidden = conditions.encoderHiddenStates else {
                fatalError("CoreMLDiTStepper requires encoderHiddenStates to be populated")
            }
            let encSeq = encHidden.dim(1)
            let encW = encHidden.dim(2)
            let encData = encHidden.contiguous().asArray(Float.self)
            let encMean = encData.reduce(0, +) / Float(encData.count)
            debugPrint("[CoreMLDiTStepper] predictVelocity t=\(timestep) encMean=\(encMean) shape=\(encHidden.shape)")
            let encShaped = MLShapedArray<Float>(scalars: encData, shape: [batch, encSeq, encW])
            
            let encMaskData: [Float]
            if let realEncMask = conditions.encoderAttentionMask {
                encMaskData = realEncMask.contiguous().asArray(Float.self)
            } else {
                encMaskData = [Float](repeating: 1.0, count: encSeq)
            }
            let encMaskShaped = MLShapedArray<Float>(scalars: encMaskData, shape: [batch, encSeq])
            
            let ctxLatents = conditions.contextLatents ?? MLXArray.zeros([batch, seq, d * 2], type: Float.self)
            let ctxW = ctxLatents.dim(2)
            let ctxData = ctxLatents.contiguous().asArray(Float.self)
            let ctxShaped = MLShapedArray<Float>(scalars: ctxData, shape: [batch, seq, ctxW])
            
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
                "hidden_states": MLMultiArray(latentShaped),
                "timestep": MLMultiArray(tShaped),
                "timestep_r": MLMultiArray(trShaped),
                "attention_mask": MLMultiArray(cachedMaskShaped!),
                "encoder_hidden_states": MLMultiArray(encShaped),
                "encoder_attention_mask": MLMultiArray(encMaskShaped),
                "context_latents": MLMultiArray(ctxShaped),
                "position_ids": MLMultiArray(cachedPosShaped!),
                "cache_position": MLMultiArray(cachedCacheShaped!),
                "sliding_window_mask": MLMultiArray(cachedSWMaskShaped!)
            ])
            
            let output = try model.prediction(from: inputProvider)
            
            guard let velocityMultiArray = output.featureValue(for: "velocity")?.multiArrayValue else {
                fatalError("Failed to extract velocity from Core ML output")
            }
            
            let vt: MLXArray
            if velocityMultiArray.dataType == .float16 {
                let count = velocityMultiArray.count
                var floats = [Float](repeating: 0, count: count)
                let f16Ptr = velocityMultiArray.dataPointer.bindMemory(to: Float16.self, capacity: count)
                for i in 0..<count {
                    floats[i] = Float(f16Ptr[i])
                }
                vt = MLXArray(floats, [batch, seq, d])
            } else {
                let velocityShaped = MLShapedArray<Float>(velocityMultiArray)
                vt = MLXArray(velocityShaped.scalars, [batch, seq, d])
            }
            
            let vtMean = vt.mean().item(Float.self)
            print("[CoreMLDiTStepper] predictVelocity t=\(timestep) mean=\(vtMean)")
            return vt
            
        } catch {
            print("[CoreMLDiTStepper] Prediction failed: \(error)")
            return MLXArray.zeros([batch, seq, d]) // Fallback
        }
    }

    public func step(currentLatent: MLXArray, timestep: Float, conditions: DiTConditions, nextTimestep: Float?) -> MLXArray {
        let vt = predictVelocity(
            currentLatent: currentLatent,
            timestep: timestep,
            conditions: conditions,
            useCache: true
        )
        
        // Apply ODE step (same as MLXDiTStepper)
        if let nextT = nextTimestep {
            let dt = timestep - nextT
            return currentLatent - vt * dt
        } else {
            return currentLatent - vt * timestep
        }
    }
}
