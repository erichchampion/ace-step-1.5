/**
 KV cache for cross-attention: K/V from encoder hidden states are computed once and reused. Mirrors dit_model.MLXCrossAttentionCache.
 */

import Foundation
import MLX

public final class DiTCrossAttentionCache {
    private var keys: [Int: MLXArray] = [:]
    private var values: [Int: MLXArray] = [:]
    private var updated: Set<Int> = []

    public init() {}

    public func update(key: MLXArray, value: MLXArray, layerIdx: Int) {
        keys[layerIdx] = key
        values[layerIdx] = value
        updated.insert(layerIdx)
    }

    public func isUpdated(layerIdx: Int) -> Bool {
        updated.contains(layerIdx)
    }

    public func get(layerIdx: Int) -> (MLXArray, MLXArray) {
        (keys[layerIdx]!, values[layerIdx]!)
    }
}
