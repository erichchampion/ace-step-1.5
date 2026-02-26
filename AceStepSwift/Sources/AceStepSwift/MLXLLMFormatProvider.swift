/**
 LLMFormatProvider that loads and runs a model via MLXLMCommon (mlx-swift-lm). Uses ChatSession for generation.
 */

import Foundation
import MLXLMCommon

public final class MLXLLMFormatProvider: LLMFormatProvider {

    private var container: ModelContainer?

    public var isInitialized: Bool { container != nil }

    public init() {
        self.container = nil
    }

    /// Load model from a local directory (config.json + weights).
    public func load(from directory: URL) async throws {
        container = try await loadModelContainer(
            directory: directory,
            progressHandler: { _ in }
        )
    }

    /// Load model from Hugging Face by id (e.g. "mlx-community/Qwen3-4B-4bit").
    public func load(id: String) async throws {
        container = try await loadModelContainer(
            id: id,
            progressHandler: { _ in }
        )
    }

    /// Unload the model to free memory.
    public func unload() {
        container = nil
    }

    public func generateFormat(
        caption: String,
        lyrics: String,
        userMetadata: [String: Any]?,
        temperature: Double
    ) async throws -> String {
        guard let container else {
            throw MLXLLMFormatProviderError.notInitialized
        }
        let params = GenerateParameters(
            maxTokens: 2048,
            temperature: Float(temperature),
            topP: 0.9
        )
        let prompt = buildFormatPrompt(caption: caption, lyrics: lyrics)
        let session = ChatSession(container, generateParameters: params)
        return try await session.respond(to: prompt)
    }

    public func generateFromQuery(
        query: String,
        instrumental: Bool,
        vocalLanguage: String?,
        temperature: Double
    ) async throws -> String {
        guard let container else {
            throw MLXLLMFormatProviderError.notInitialized
        }
        let params = GenerateParameters(
            maxTokens: 2048,
            temperature: Float(temperature),
            topP: 0.9
        )
        let prompt = buildInspirationPrompt(query: query, instrumental: instrumental)
        let session = ChatSession(container, generateParameters: params)
        return try await session.respond(to: prompt)
    }

    private func buildInspirationPrompt(query: String, instrumental: Bool) -> String {
        let q = query.trimmingCharacters(in: .whitespaces).isEmpty ? "NO USER INPUT" : query
        let inst = instrumental ? "true" : "false"
        return """
        # Instruction
        Expand the user's input into a more detailed and specific musical description.
        Output exactly in this format inside a <think> block:
        <think>
        bpm: [30-300]
        caption: [one sentence description ending with a period]
        duration: [10-600]
        keyscale: [e.g. C major or Am]
        language: [e.g. en, zh, ja]
        timesignature: [2, 3, 4, or 6]
        </think>

        \(q)

        instrumental: \(inst)
        """
    }

    private func buildFormatPrompt(caption: String, lyrics: String) -> String {
        """
        Format the following music caption and lyrics into structured metadata.
        Output exactly in this format inside a <think> block:
        <think>
        bpm: [30-300]
        caption: [one sentence description ending with a period]
        duration: [10-600]
        keyscale: [e.g. C major or Am]
        language: [e.g. en, zh, ja]
        timesignature: [2, 3, 4, or 6]
        </think>

        Caption: \(caption)
        Lyrics:
        \(lyrics)
        """
    }
}

public enum MLXLLMFormatProviderError: Error {
    case notInitialized
}
