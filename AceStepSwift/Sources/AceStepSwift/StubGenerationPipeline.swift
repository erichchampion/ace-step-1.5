/**
 Stub pipeline that returns success with no audio. Used for tests and when DiT/VAE are not yet loaded.
 */

import Foundation

public final class StubGenerationPipeline: GenerationPipeline {
    public var isInitialized: Bool { true }

    public init() {}

    public func run(params: GenerationParams, config: GenerationConfig, progress: ((Double, String) -> Void)?) async throws -> GenerationResult {
        progress?(1.0, "Stub complete")
        return GenerationResult(
            audios: [],
            statusMessage: "Stub pipeline (DiT/VAE not implemented)",
            extraOutputs: [:],
            success: true,
            error: nil
        )
    }
}
