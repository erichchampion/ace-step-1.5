import Foundation
import MLX
import MLXLMCommon
import MLXLLM

public protocol TextHiddenStateProvider: AnyObject {
    func encodeHiddenStates(text: String, maxLength: Int) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray)
}

public enum QwenTextHiddenStateProviderError: Error {
    case unsupportedModelType(String)
}

/// Produces text-model hidden states using Qwen3 embedding weights/tokenizer.
/// This mirrors the Python text encoder path at a high level: tokenize -> model -> last hidden states.
public final class QwenTextHiddenStateProvider: TextHiddenStateProvider {
    private let model: Qwen3Model
    private let encodeTokens: (String) -> [Int]
    private let lock = NSLock()

    private init(model: Qwen3Model, encodeTokens: @escaping (String) -> [Int]) {
        self.model = model
        self.encodeTokens = encodeTokens
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
        return QwenTextHiddenStateProvider(
            model: qwenModel,
            encodeTokens: { context.tokenizer.encode(text: $0, addSpecialTokens: true) }
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
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
