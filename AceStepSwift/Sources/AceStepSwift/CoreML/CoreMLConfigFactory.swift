import Foundation
import CoreML

/// Centralized factory for CoreML model configurations with optimization hints.
///
/// All CoreML models in the pipeline should use this factory instead of creating
/// bare `MLModelConfiguration()` instances, ensuring consistent optimization hints
/// (e.g., `reshapeFrequency`, `specializationStrategy`) are applied across the
/// entire inference pipeline.
public enum CoreMLConfigFactory {

    /// Build a standard `MLModelConfiguration` for inference models.
    ///
    /// - Parameters:
    ///   - computeUnits: Compute units to use. Default is `.cpuAndGPU`, bypassing
    ///     the ANE due to known `mul` broadcasting crashes with `RangeDim` shapes.
    ///     Use `.cpuOnly` for models with MLIR pass-manager issues (e.g., text encoder/LLM).
    ///   - reshapeFrequency: How often the model's input shapes are expected to change.
    ///     `.infrequent` tells CoreML to re-optimize internal compute plans for each new
    ///     shape, yielding faster subsequent predictions — ideal for diffusion loops and
    ///     single-shot encode/decode calls. `.frequent` (the CoreML default) avoids
    ///     shape-specific optimizations.
    /// - Returns: A configured `MLModelConfiguration` with optimization hints set.
    public static func makeConfig(
        computeUnits: MLComputeUnits = .cpuAndGPU,
        reshapeFrequency: ReshapeHint = .infrequent
    ) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        var hints = MLOptimizationHints()
        switch reshapeFrequency {
        case .infrequent:
            hints.reshapeFrequency = .infrequent
        case .frequent:
            hints.reshapeFrequency = .frequent
        }
        config.optimizationHints = hints

        return config
    }

    /// Expected frequency of input shape changes.
    public enum ReshapeHint {
        /// Input shapes are stable across predictions (e.g., diffusion steps, single-shot encode).
        /// CoreML will spend extra time optimizing for the shape, yielding faster predictions.
        case infrequent
        /// Input shapes change frequently between predictions.
        /// CoreML will avoid shape-specific optimizations.
        case frequent
    }
}
