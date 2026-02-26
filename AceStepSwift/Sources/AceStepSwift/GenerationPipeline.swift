/**
 Protocol for the music generation pipeline (DiT + VAE). Allows stub for tests and real implementation later.
 */

import Foundation

public protocol GenerationPipeline: AnyObject {
    var isInitialized: Bool { get }
    func run(params: GenerationParams, config: GenerationConfig, progress: ((Double, String) -> Void)?) async throws -> GenerationResult
}
