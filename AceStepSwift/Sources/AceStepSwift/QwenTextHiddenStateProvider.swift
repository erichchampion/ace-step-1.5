import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM
import Tokenizers

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

    private static let defaultQuantizedModelID = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

    /// Load from a quantized MLX-community model directory (uses MLXLLM's loadModel).
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
                context = try await loadModel(id: defaultQuantizedModelID)
            }
        } else {
            context = try await loadModel(id: defaultQuantizedModelID)
        }
        
        guard let qwenModel = context.model as? Qwen3Model else {
            throw QwenTextHiddenStateProviderError.unsupportedModelType(String(describing: type(of: context.model)))
        }
        
        print("[QwenTextHiddenStateProvider] Using model.modules() Embedding for lyrics (quantized model)")
        
        return QwenTextHiddenStateProvider(
            model: qwenModel,
            encodeTokens: { context.tokenizer.encode(text: $0, addSpecialTokens: true) },
            lyricEmbeddingLoader: nil
        )
    }

    /// Load from a full-precision HuggingFace Qwen3-Embedding model directory (PyTorch safetensors format).
    /// This is the preferred path: full-precision BF16 weights produce correct embedding magnitudes
    /// matching the Python pipeline's text_encoder.embed_tokens() output.
    public static func loadFullPrecision(directory: URL) async throws -> QwenTextHiddenStateProvider {
        print("[QwenTextHiddenStateProvider] Loading full-precision model from: \(directory.path)")

        // 1. Build Qwen3Configuration via JSON decoding (the memberwise init is internal to MLXLLM).
        //    The full-precision model's config.json is a sentence-transformers pooling config,
        //    not the Qwen architecture config, so we encode the known values manually.
        let configJSON = """
        {
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "rms_norm_eps": 1e-6,
            "vocab_size": 151669,
            "num_key_value_heads": 8,
            "rope_theta": 1000000,
            "head_dim": 128,
            "tie_word_embeddings": true,
            "max_position_embeddings": 32768
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(Qwen3Configuration.self, from: configJSON)

        // 2. Create model
        let model = Qwen3Model(config)

        // 3. Load safetensors weights; add "model." prefix since the PyTorch safetensors
        //    uses flat keys (embed_tokens.weight, layers.0...) but Qwen3Model wraps them
        //    under its `model` (Qwen3ModelInner) property.
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        let rawWeights = try loadArrays(url: weightsURL)
        let sanitized = model.sanitize(weights: rawWeights)
        let prefixed: [String: MLXArray] = Dictionary(uniqueKeysWithValues:
            sanitized.map { (key, value) in
                let newKey = "model.\(key)"
                return (newKey, value)
            }
        )
        let params = ModuleParameters.unflattened(prefixed)
        model.update(parameters: params)
        eval(model.parameters())
        print("[QwenTextHiddenStateProvider] Full-precision model weights loaded (\(rawWeights.count) tensors)")

        // 4. Load tokenizer
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw QwenTextHiddenStateProviderError.unsupportedModelType("tokenizer.json not found in \(directory.path)")
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)

        print("[QwenTextHiddenStateProvider] Full-precision Qwen3 ready (embed_tokens is unquantized Embedding)")

        return QwenTextHiddenStateProvider(
            model: model,
            encodeTokens: { tokenizer.encode(text: $0, addSpecialTokens: true) },
            lyricEmbeddingLoader: nil
        )
    }

    /// Lock-free model forward pass. Callers MUST hold `lock`.
    private func _forwardUnlocked(inputIDs: MLXArray) -> MLXArray {
        model.model(inputIDs, cache: nil)
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
            _forwardUnlocked(inputIDs: inputIDs)
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
        
        print("[QwenTextHiddenStateProvider] Using model.modules() to find Embedding for lyrics")
        let inputIDs = MLXArray(tokenIDs, [1, tokenIDs.count])
        let embeddings: MLXArray
        lock.lock()
        defer { lock.unlock() }
        
        if let embeddingLayer = model.model.modules().compactMap({ $0 as? Embedding }).first {
            embeddings = embeddingLayer(inputIDs)
        } else {
            // Fallback: full forward pass. Lock is already held, so call _forwardUnlocked directly.
            print("[QwenTextHiddenStateProvider] FAIL: Could not find Embedding module, falling back to full forward")
            let hiddenStates = _forwardUnlocked(inputIDs: inputIDs)
            hiddenStates.eval()
            let mask = MLXArray.ones([1, tokenIDs.count])
            return (hiddenStates, mask)
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
