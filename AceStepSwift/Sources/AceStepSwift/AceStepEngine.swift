/**
 Top-level API for format_sample and generate_music. Cadenza app calls these.
 */

import Foundation
#if canImport(MLX)
import MLX
#endif
public enum AceStepEngine {

    /// Configure MLX Metal Memory Limits (in bytes). Call this once at app startup.
    /// - Parameters:
    ///   - memoryLimit: Maximum memory MLX is allowed to allocate on the GPU (e.g. 4 * 1024 * 1024 * 1024 for 4GB).
    ///   - cacheLimit: Maximum memory MLX will keep in its Metal buffer cache (e.g. 1 * 1024 * 1024 * 1024 for 1GB).
    public static func configureMLXMemory(memoryLimit: Int? = nil, cacheLimit: Int? = nil) {
        #if canImport(MLX)
        if let limit = memoryLimit {
            MLX.GPU.set(memoryLimit: limit)
        }
        if let limit = cacheLimit {
            MLX.GPU.set(cacheLimit: limit)
        }
        #endif
    }

    /// Generate music asynchronously. Preferred over the synchronous `generateMusic` for UI contexts
    /// because it does not block the calling thread with `DispatchGroup.wait()`.
    /// Supports Swift cooperative cancellation — cancel the enclosing `Task` to abort between diffusion steps.
    public static func generateMusicAsync(
        params: GenerationParams,
        config: GenerationConfig,
        progress: ((Double, String) -> Void)?,
        pipeline: GenerationPipeline?
    ) async -> GenerationResult {
        guard let pipe = pipeline, pipe.isInitialized else {
            return GenerationResult(
                audios: [],
                statusMessage: "Pipeline not initialized",
                extraOutputs: [:],
                success: false,
                error: "Pipeline not initialized"
            )
        }
        do {
            return try await pipe.run(params: params, config: config, progress: progress)
        } catch is CancellationError {
            return GenerationResult(
                audios: [],
                statusMessage: "Generation cancelled",
                extraOutputs: [:],
                success: false,
                error: "cancelled"
            )
        } catch {
            return GenerationResult(
                audios: [],
                statusMessage: "\(error)",
                extraOutputs: [:],
                success: false,
                error: "\(error)"
            )
        }
    }

    /// A thread-safe box for retrieving the result of an unstructured task.
    private final class Box<T>: @unchecked Sendable {
        var value: T?
    }

    /// Helper to run an async operation synchronously without strict concurrency warnings.
    private static func runSynchronousTask<T>(operation: @escaping () async throws -> T) -> Result<T, Error> {
        let box = Box<Result<T, Error>>()
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            do {
                let res = try await operation()
                box.value = .success(res)
            } catch {
                box.value = .failure(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        return box.value ?? .failure(NSError(domain: "AceStepEngine", code: -1, userInfo: [NSLocalizedDescriptionKey: "Task completed without result"]))
    }

    /// Generate music from params. Returns GenerationResult with audios or error.
    /// - Note: This synchronous variant blocks the calling thread via `DispatchSemaphore.wait()`.
    ///   For UI contexts, use `generateMusicAsync` instead to keep the main thread responsive.
    public static func generateMusic(
        params: GenerationParams,
        config: GenerationConfig,
        progress: ((Double, String) -> Void)?,
        pipeline: GenerationPipeline?
    ) -> GenerationResult {
        guard let pipe = pipeline, pipe.isInitialized else {
            return GenerationResult(
                audios: [],
                statusMessage: "Pipeline not initialized",
                extraOutputs: [:],
                success: false,
                error: "Pipeline not initialized"
            )
        }
        
        let result = runSynchronousTask {
            try await pipe.run(params: params, config: config, progress: progress)
        }
        
        switch result {
        case .success(let genResult):
            return genResult
        case .failure(let error):
            return GenerationResult(
                audios: [],
                statusMessage: "\(error)",
                extraOutputs: [:],
                success: false,
                error: "\(error)"
            )
        }
    }

    /// Format caption and lyrics via LLM into structured metadata. Returns FormatSampleResult.
    public static func formatSample(
        caption: String,
        lyrics: String,
        userMetadata: [String: Any]? = nil,
        temperature: Double = 0.85,
        llmProvider: LLMFormatProvider?
    ) -> FormatSampleResult {
        guard let provider = llmProvider, provider.isInitialized else {
            return FormatSampleResult(
                caption: "",
                lyrics: lyrics,
                bpm: nil,
                duration: nil,
                keyscale: "",
                language: "",
                timesignature: "",
                statusMessage: "LLM not initialized",
                success: false,
                error: "LLM not initialized"
            )
        }
        
        let cap = caption.trimmingCharacters(in: .whitespaces).isEmpty ? "NO USER INPUT" : caption
        let lyr = lyrics.trimmingCharacters(in: .whitespaces).isEmpty ? "[Instrumental]" : lyrics
        
        let result = runSynchronousTask {
            try await provider.generateFormat(caption: cap, lyrics: lyr, userMetadata: userMetadata, temperature: temperature)
        }
        
        guard case .success(let outputText) = result, !outputText.isEmpty else {
            return FormatSampleResult(
                caption: "",
                lyrics: lyr,
                bpm: nil,
                duration: nil,
                keyscale: "",
                language: "",
                timesignature: "",
                statusMessage: "Format generation failed",
                success: false,
                error: "Format generation failed"
            )
        }
        
        return FormatSampleParser.parseToFormatSampleResult(outputText: outputText, fallbackLyrics: lyr)
    }

    /// Returns a random preset for form filling. sampleType: "simple_mode" or "custom_mode" (defaults to "simple_mode").
    /// Does not call the LLM. Optional follow-up: pass result caption/lyrics to formatSample for enhancement.
    public static func createRandomSample(sampleType: String = "simple_mode") -> RandomSampleResult {
        RandomSampleStore.pickRandom(sampleType: sampleType)
    }

    /// Create a music sample from a natural language query via the LLM. Returns CreateSampleResult (caption, lyrics, metadata).
    public static func createSample(
        query: String,
        instrumental: Bool = false,
        vocalLanguage: String? = nil,
        temperature: Double = 0.85,
        llmProvider: LLMFormatProvider?
    ) -> CreateSampleResult {
        guard let provider = llmProvider, provider.isInitialized else {
            return CreateSampleResult(
                statusMessage: "LLM not initialized",
                success: false,
                error: "LLM not initialized"
            )
        }
        
        let result = runSynchronousTask {
            try await provider.generateFromQuery(query: query, instrumental: instrumental, vocalLanguage: vocalLanguage, temperature: temperature)
        }
        
        guard case .success(let outputText) = result, !outputText.isEmpty else {
            return CreateSampleResult(
                statusMessage: "Create sample failed",
                success: false,
                error: "Create sample failed"
            )
        }
        
        return FormatSampleParser.parseToCreateSampleResult(outputText: outputText, instrumental: instrumental)
    }
}
