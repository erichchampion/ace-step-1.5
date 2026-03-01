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
        
        // Load the model (either local or remote)
        if let directory {
            do {
                print("[QwenTextHiddenStateProvider] Attempting to load model from: \(directory.path)")
                context = try await loadModel(directory: directory)
                print("[QwenTextHiddenStateProvider] Local model loaded successfully!")
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
        
        // Note: We don't use LyricTokenEmbeddingLoader with the quantized model because
        // the embed_tokens weights are stored in quantized format in safetensors.
        // Instead, we use the full model forward for lyrics (encodeLyricHiddenStates).
        print("[QwenTextHiddenStateProvider] Using full model forward for lyrics (quantized model)")
        
        return QwenTextHiddenStateProvider(
            model: qwenModel,
            encodeTokens: { context.tokenizer.encode(text: $0, addSpecialTokens: true) },
            lyricEmbeddingLoader: nil
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
        
        print("[QwenTextHiddenStateProvider] Using model.flattenedParameters() for lyrics")
        let inputIDs = MLXArray(tokenIDs, [1, tokenIDs.count])
        let embeddings: MLXArray
        lock.lock()
        defer { lock.unlock() }
        let flat = model.parameters().flattened()
        if let match = flat.first(where: { $0.0 == "model.embed_tokens.weight" || $0.0 == "embed_tokens.weight" }) {
            // Manual embedding lookup: shape [vocab_size, hidden_dim]
            let weight = match.1
            embeddings = weight[inputIDs]
        } else {
            print("[QwenTextHiddenStateProvider] FAIL: Could not find embed_tokens.weight, falling back to full forward")
            lock.unlock()
            let lyric = try encodeLyricHiddenStates(text: text, maxLength: maxLength)
            return (lyric.hiddenStates, lyric.attentionMask)
        }
        embeddings.eval()
        let mask = MLXArray.ones([1, tokenIDs.count])
        return (embeddings, mask)
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
