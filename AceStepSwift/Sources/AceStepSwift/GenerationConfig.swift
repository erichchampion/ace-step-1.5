/**
 Generation run configuration. Mirrors acestep/inference.py GenerationConfig.
 */

public struct GenerationConfig {
    public var batchSize: Int
    public var allowLmBatch: Bool
    public var useRandomSeed: Bool
    public var seeds: [Int]?
    public var lmBatchChunkSize: Int
    public var constrainedDecodingDebug: Bool
    public var audioFormat: String

    public init(
        batchSize: Int = 2,
        allowLmBatch: Bool = false,
        useRandomSeed: Bool = true,
        seeds: [Int]? = nil,
        lmBatchChunkSize: Int = 8,
        constrainedDecodingDebug: Bool = false,
        audioFormat: String = "flac"
    ) {
        self.batchSize = batchSize
        self.allowLmBatch = allowLmBatch
        self.useRandomSeed = useRandomSeed
        self.seeds = seeds
        self.lmBatchChunkSize = lmBatchChunkSize
        self.constrainedDecodingDebug = constrainedDecodingDebug
        self.audioFormat = audioFormat
    }
}
