/**
 Diagnostic helpers for array math and formatting.
 Extracted from ContractGenerationPipeline for DRY adherence and reuse across ML modules.
 */

import Foundation
#if canImport(MLX)
import MLX
#endif

public enum DiagnosticLogHelpers {

    /// Returns mean and standard deviation for a flattened swift array of Floats.
    public static func stats(_ x: [Float]) -> (mean: Float, std: Float) {
        guard !x.isEmpty else { return (0, 0) }
        let mean = x.reduce(0, +) / Float(x.count)
        let variance = x.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(x.count)
        return (mean, sqrt(max(variance, 0.0)))
    }

    /// String formats a float metric.
    public static func formatStat(_ value: Float) -> String {
        String(format: "%.6f", value)
    }

    #if canImport(MLX)
    /// Returns mean and standard deviation of an MLX array as Floats.
    public static func tensorMeanStd(_ x: MLXArray) -> (mean: Float, std: Float) {
        let mean = x.mean().item(Float.self)
        let centered = x - MLXArray(mean)
        let variance = (centered * centered).mean().item(Float.self)
        return (mean, sqrt(max(variance, 0.0)))
    }
    #endif
}
