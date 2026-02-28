import Foundation
import MLX
import MLXLMCommon
import MLXLLM

public protocol TextHiddenStateProvider: AnyObject {
    func encodeHiddenStates(text: String, maxLength: Int) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray)
    /// Lyric branch states for ConditionEncoder.lyricEncoder path.
    /// Default implementation can mirror encodeHiddenStates; providers may override.
    func encodeLyricHiddenStates(text: String, maxLength: Int) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray)
    /// Token embedding branch used when a provider can expose embed_tokens directly.
    /// Default implementation falls back to encodeLyricHiddenStates.
    func encodeTokenEmbeddings(text: String, maxLength: Int) throws -> (embeddings: MLXArray, attentionMask: MLXArray)
}

public enum QwenTextHiddenStateProviderError: Error {
    case unsupportedModelType(String)
}

/// Produces text-model hidden states using Qwen3 embedding weights/tokenizer.
/// This mirrors the Python text encoder path at a high level: tokenize -> model -> last hidden states.
/// When lyricEmbeddingLoader is set, encodeTokenEmbeddings uses raw token embeddings (matches Python embed_tokens).
public final class QwenTextHiddenStateProvider: TextHiddenStateProvider {
    private let model: Qwen3Model
    private let encodeTokens: (String) -> [Int]
    private let lock = NSLock()
    private var lyricEmbeddingLoader: LyricTokenEmbeddingLoader?

    private init(model: Qwen3Model, encodeTokens: @escaping (String) -> [Int], lyricEmbeddingLoader: LyricTokenEmbeddingLoader? = nil) {
        self.model = model
        self.encodeTokens = encodeTokens
        self.lyricEmbeddingLoader = lyricEmbeddingLoader
    }

    public static func load(directory: URL? = nil) async throws -> QwenTextHiddenStateProvider {
        let context: ModelContext
        if let directory {
            context = try await loadModel(directory: directory)
        } else {
            context = try await loadModel(id: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
        }
        guard let qwenModel = context.model as? Qwen3Model else {
            throw QwenTextHiddenStateProviderError.unsupportedModelType(String(describing: type(of: context.model)))
        }
        var lyricLoader: LyricTokenEmbeddingLoader?
        if let directory {
            lyricLoader = try? LyricTokenEmbeddingLoader.load(fromDirectory: directory)
        }
        return QwenTextHiddenStateProvider(
            model: qwenModel,
            encodeTokens: { context.tokenizer.encode(text: $0, addSpecialTokens: true) },
            lyricEmbeddingLoader: lyricLoader
        )
    }

    public func encodeHiddenStates(text: String, maxLength: Int = 256) throws -> (
        hiddenStates: MLXArray, attentionMask: MLXArray
    ) {
        var tokenIDs = encodeTokens(text)
        if tokenIDs.isEmpty {
            tokenIDs = [0]
        }
        if tokenIDs.count > maxLength {
            tokenIDs = Array(tokenIDs.prefix(maxLength))
        }

        let inputIDs = MLXArray(tokenIDs, [1, tokenIDs.count])
        let attentionMask = MLXArray.ones([1, tokenIDs.count])

        let hiddenStates = lock.withLock {
            // Use the transformer body output (not lm_head logits) as text hidden states.
            model.model(inputIDs, cache: nil)
        }
        hiddenStates.eval()
        return (hiddenStates, attentionMask)
    }

    public func encodeLyricHiddenStates(text: String, maxLength: Int = 2048) throws -> (
        hiddenStates: MLXArray, attentionMask: MLXArray
    ) {
        try encodeHiddenStates(text: text, maxLength: maxLength)
    }

    public func encodeTokenEmbeddings(text: String, maxLength: Int = 2048) throws -> (
        embeddings: MLXArray, attentionMask: MLXArray
    ) {
        if let loader = lyricEmbeddingLoader {
            var tokenIDs = encodeTokens(text)
            if tokenIDs.isEmpty { tokenIDs = [0] }
            if tokenIDs.count > maxLength { tokenIDs = Array(tokenIDs.prefix(maxLength)) }
            let embeddings = loader.embed(tokenIDs: tokenIDs)
            embeddings.eval()
            let mask = MLXArray.ones([1, tokenIDs.count])
            return (embeddings, mask)
        }
        let lyric = try encodeLyricHiddenStates(text: text, maxLength: maxLength)
        return (lyric.hiddenStates, lyric.attentionMask)
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}

public extension TextHiddenStateProvider {
    func encodeLyricHiddenStates(text: String, maxLength: Int) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray) {
        try encodeHiddenStates(text: text, maxLength: maxLength)
    }

    func encodeTokenEmbeddings(text: String, maxLength: Int) throws -> (embeddings: MLXArray, attentionMask: MLXArray) {
        let lyric = try encodeLyricHiddenStates(text: text, maxLength: maxLength)
        return (lyric.hiddenStates, lyric.attentionMask)
    }
}
