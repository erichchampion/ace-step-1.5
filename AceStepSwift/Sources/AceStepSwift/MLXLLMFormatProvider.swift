/**
 LLMFormatProvider that loads and runs a model via MLXLMCommon (mlx-swift-lm). Uses ChatSession for generation.
 */

import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM
import Tokenizers

public final class MLXLLMFormatProvider: LLMFormatProvider {

    private var container: ModelContainer?

    public var isInitialized: Bool { container != nil }

    public init() {
        self.container = nil
    }

    /// Load model from a local directory (config.json + weights).
    /// Falls back to manual prefix-mapped loading when the safetensors keys
    /// lack the `model.` prefix expected by the Swift Qwen3Model struct.
    public func load(from directory: URL) async throws {
        do {
            container = try await loadModelContainer(
                directory: directory,
                progressHandler: { _ in }
            )
        } catch {
            // Standard loadModelContainer fails for ACE-Step LM weights because
            // the safetensors keys (e.g. `norm.weight`, `layers.0.…`) lack the
            // `model.` prefix that the Swift Qwen3Model struct expects.
            // Fix: create a temp directory with prefix-mapped weights, then retry.
            let prefixedDir = try Self.createPrefixedWeightsDirectory(from: directory)
            container = try await loadModelContainer(
                directory: prefixedDir,
                progressHandler: { _ in }
            )
        }
    }

    /// Creates a temporary directory containing model.safetensors with all keys
    /// prepended with "model.", plus copies of config.json and tokenizer files.
    private static func createPrefixedWeightsDirectory(from directory: URL) throws -> URL {
        let fm = FileManager.default
        let tmpDir = fm.temporaryDirectory.appendingPathComponent("mlx-llm-prefixed-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: tmpDir, withIntermediateDirectories: true)

        // Copy config and tokenizer files
        for filename in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            let src = directory.appendingPathComponent(filename)
            if fm.fileExists(atPath: src.path) {
                try fm.copyItem(at: src, to: tmpDir.appendingPathComponent(filename))
            }
        }

        // Load weights, add "model." prefix, and write back
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        let rawWeights = try loadArrays(url: weightsURL)
        let prefixed: [String: MLXArray] = Dictionary(uniqueKeysWithValues:
            rawWeights.map { (key, value) in
                let newKey = key.hasPrefix("model.") ? key : "model.\(key)"
                return (newKey, value)
            }
        )
        let outURL = tmpDir.appendingPathComponent("model.safetensors")
        try save(arrays: prefixed, url: outURL)
        return tmpDir
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

    /// Python: DEFAULT_LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"
    private static let formatSystemPrompt = "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"

    /// Python: DEFAULT_LM_INSPIRED_INSTRUCTION = "Expand the user's input into a more detailed and specific musical description:"
    private static let inspirationSystemPrompt = "# Instruction\nExpand the user's input into a more detailed and specific musical description:\n\n"

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
        let session = ChatSession(container, instructions: Self.formatSystemPrompt, generateParameters: params)
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
        let session = ChatSession(container, instructions: Self.inspirationSystemPrompt, generateParameters: params)
        return try await session.respond(to: prompt)
    }

    /// User message for inspiration mode.
    /// Python: f"{query}\n\ninstrumental: {instrumental_str}"
    private func buildInspirationPrompt(query: String, instrumental: Bool) -> String {
        let q = query.trimmingCharacters(in: .whitespaces).isEmpty ? "NO USER INPUT" : query
        let inst = instrumental ? "true" : "false"
        return "\(q)\n\ninstrumental: \(inst)"
    }

    /// User message for format mode.
    /// Python: f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
    private func buildFormatPrompt(caption: String, lyrics: String) -> String {
        "# Caption\n\(caption)\n\n# Lyric\n\(lyrics)\n"
    }
}

public enum MLXLLMFormatProviderError: Error {
    case notInitialized
}
