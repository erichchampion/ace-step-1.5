import Foundation
import MLX
import MLXLMCommon
import MLXLLM

public protocol TextHiddenStateProvider: AnyObject {
    func encodeHiddenStates(text: String, maxLength: Int) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray)
    func encodeLyricHiddenStates(text: String, maxLength: Int) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray)
    func encodeTokenEmbeddings(text: String, maxLength: Int) throws -> (embeddings: MLXArray, attentionMask: MLXArray)
}

public enum QwenTextHiddenStateProviderError: Error {
    case unsupportedModelType(String)
}

/// Produces text-model hidden states using Qwen3 embedding weights/tokenizer.
/// This mirrors the Python text encoder path at a high level: tokenize -> model -> last hidden states.
/// When lyricEmbeddingLoader is set, encodeTokenEmbeddings uses raw token embeddings (matches Python embed_tokens).
/// If not, falls back to using model's embedTokens directly (available on Qwen3Model).
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
        var lyricLoader: LyricTokenEmbeddingLoader? = nil
        
        if let directory {
            do {
                print("[QwenTextHiddenStateProvider] Attempting to load model from: \(directory.path)")
                context = try await loadModel(directory: directory)
                print("[QwenTextHiddenStateProvider] Local model loaded successfully!")
                
                // Try to load embed_tokens from local directory
                lyricLoader = try? LyricTokenEmbeddingLoader.load(fromDirectory: directory)
                if lyricLoader != nil {
                    print("[QwenTextHiddenStateProvider] LyricTokenEmbeddingLoader loaded successfully from local model")
                }
            } catch {
                print("[QwenTextHiddenStateProvider] Local model load failed: \(error.localizedDescription)")
                print("[QwenTextHiddenStateProvider] Falling back to remote...")
                context = try await loadModel(id: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
            }
        } else {
            context = try await loadModel(id: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
        }
        
        guard let qwenModel = context.model as? Qwen3Model else {
            throw QwenTextHiddenStateProviderError.unsupportedModelType(String(describing: type(of: context.model)))
        }
        
        if lyricLoader == nil {
            print("[QwenTextHiddenStateProvider] WARNING: Using full model forward for lyrics (no embed_tokens)")
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
        var tokenIDs = encodeTokens(text)
        if tokenIDs.isEmpty { tokenIDs = [0] }
        if tokenIDs.count > maxLength { tokenIDs = Array(tokenIDs.prefix(maxLength)) }
        
        if let loader = lyricEmbeddingLoader {
            let embeddings = loader.embed(tokenIDs: tokenIDs)
            embeddings.eval()
            print("[QwenTextHiddenStateProvider] Using embed_tokens for lyrics")
            let mask = MLXArray.ones([1, tokenIDs.count])
            return (embeddings, mask)
        }
        
        print("[QwenTextHiddenStateProvider] WARNING: Using full model forward for lyrics (no embed_tokens)")
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
