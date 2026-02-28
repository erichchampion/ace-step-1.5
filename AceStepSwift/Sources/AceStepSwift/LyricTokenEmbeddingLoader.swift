/**
 Loads the token embedding matrix from a Qwen checkpoint (safetensors) for the lyric branch.
 Python uses text_encoder.embed_tokens(lyric_token_ids); this provides the same input to ConditionEncoder.lyricEncoder.
 */

import Foundation
import MLX

/// Embedding matrix [vocabSize, hiddenSize] and lookup for lyric token IDs.
/// Load from the same Qwen checkpoint directory used for the text encoder.
public final class LyricTokenEmbeddingLoader {
    /// Shape: [vocabSize, hiddenSize] (e.g. [151936, 1024]). Rows are token embeddings.
    public let embeddingWeight: MLXArray
    public let hiddenSize: Int
    public let vocabSize: Int

    private let lock = NSLock()

    public init(embeddingWeight: MLXArray) {
        precondition(embeddingWeight.ndim == 2, "embeddingWeight must be 2D [vocabSize, hiddenSize]")
        self.embeddingWeight = embeddingWeight
        self.vocabSize = embeddingWeight.dim(0)
        self.hiddenSize = embeddingWeight.dim(1)
    }

    /// Returns token embeddings for the given token IDs. Output shape [1, tokenIDs.count, hiddenSize].
    public func embed(tokenIDs: [Int]) -> MLXArray {
        guard !tokenIDs.isEmpty else {
            return MLXArray.zeros([1, 0, hiddenSize])
        }
        return lock.withLock {
            let rows = tokenIDs.map { id -> MLXArray in
                let clamped = min(max(id, 0), vocabSize - 1)
                return embeddingWeight[clamped..<(clamped + 1), 0..<hiddenSize]
            }
            let stacked = concatenated(rows, axis: 0)
            return stacked.expandedDimensions(axis: 0)
        }
    }

    /// Load embedding matrix from a safetensors file. Tries common Qwen/transformers key names.
    public static func load(from url: URL) throws -> LyricTokenEmbeddingLoader {
        let flat = try loadArrays(url: url)
        let keys = [
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "model.model.embed_tokens.weight"
        ]
        for key in keys {
            if let weight = flat[key] {
                guard weight.ndim == 2 else { continue }
                return LyricTokenEmbeddingLoader(embeddingWeight: weight)
            }
        }
        throw LyricTokenEmbeddingLoaderError.embeddingKeyNotFound(tried: keys, available: Array(flat.keys.prefix(20)))
    }

    /// Load from a directory containing model.safetensors (or the single safetensors file path).
    public static func load(fromDirectory directory: URL) throws -> LyricTokenEmbeddingLoader {
        let single = directory.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: single.path) {
            return try load(from: single)
        }
        if directory.pathExtension == "safetensors" && FileManager.default.fileExists(atPath: directory.path) {
            return try load(from: directory)
        }
        throw LyricTokenEmbeddingLoaderError.fileNotFound(directory: directory)
    }
}

public enum LyricTokenEmbeddingLoaderError: Error {
    case embeddingKeyNotFound(tried: [String], available: [String])
    case fileNotFound(directory: URL)
}
