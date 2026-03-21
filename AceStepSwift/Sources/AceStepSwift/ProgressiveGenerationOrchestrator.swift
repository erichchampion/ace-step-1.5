/**
 Orchestrates multi-chunk progressive song generation.
 
 Given a `ProgressiveGenerationPlan`, iterates through chunks in order.
 All chunks use text2music with full inference steps and shared conditioning
 (same genre, caption, style). Musical coherence between chunks is maintained
 through equal-power crossfade at overlap boundaries during stitching, plus
 distributed lyrics that preserve narrative flow.
 
 The caller provides a `generateChunk` closure that runs a single generation
 (with the appropriate pipeline and memory management).
 */

import Foundation

public struct ProgressiveProgressUpdate {
    /// Zero-based index of current chunk.
    public let chunkIndex: Int
    /// Total number of chunks.
    public let totalChunks: Int
    /// Progress within current chunk (0.0–1.0).
    public let chunkProgress: Double
    /// Overall progress across all chunks (0.0–1.0).
    public let overallProgress: Double
    /// Human-readable message.
    public let message: String

    public init(chunkIndex: Int, totalChunks: Int, chunkProgress: Double, overallProgress: Double, message: String) {
        self.chunkIndex = chunkIndex
        self.totalChunks = totalChunks
        self.chunkProgress = chunkProgress
        self.overallProgress = overallProgress
        self.message = message
    }
}

public enum ProgressiveGenerationOrchestrator {

    public enum OrchestratorError: LocalizedError {
        case chunkGenerationFailed(index: Int, message: String)
        case noAudioInChunkResult(index: Int)

        public var errorDescription: String? {
            switch self {
            case .chunkGenerationFailed(let idx, let msg):
                return "Chunk \(idx + 1) generation failed: \(msg)"
            case .noAudioInChunkResult(let idx):
                return "Chunk \(idx + 1) produced no audio output"
            }
        }
    }

    /// Closure type for generating a single chunk.
    /// - Parameters:
    ///   - params: Generation params for this chunk (lyrics, duration, task type, src audio, etc.)
    ///   - config: Generation config (batch size, format)
    ///   - chunkIndex: Zero-based chunk index
    ///   - progress: Per-step progress callback
    /// - Returns: GenerationResult for this chunk
    public typealias ChunkGenerator = (
        _ params: GenerationParams,
        _ config: GenerationConfig,
        _ chunkIndex: Int,
        _ progress: ((Double, String) -> Void)?
    ) async throws -> GenerationResult

    /// Generates a long song progressively.
    ///
    /// - Parameters:
    ///   - params: Original user params (used as template; lyrics/duration/taskType modified per chunk)
    ///   - config: Generation config
    ///   - plan: Pre-computed progressive plan
    ///   - generateChunk: Closure that runs one chunk generation
    ///   - progress: Progressive progress callback
    /// - Returns: Final stitched GenerationResult
    public static func generate(
        params: GenerationParams,
        config: GenerationConfig,
        plan: ProgressiveGenerationPlan,
        generateChunk: ChunkGenerator,
        progress: ((ProgressiveProgressUpdate) -> Void)?
    ) async throws -> GenerationResult {
        let totalChunks = plan.chunks.count

        // Single chunk: pass through without orchestration
        guard plan.isProgressive else {
            let result = try await generateChunk(params, config, 0) { frac, msg in
                progress?(ProgressiveProgressUpdate(
                    chunkIndex: 0, totalChunks: 1,
                    chunkProgress: frac, overallProgress: frac,
                    message: msg
                ))
            }
            return result
        }

        var chunkAudios: [[Float]] = []
        var chunkSampleRate: Int = AceStepConstants.defaultSampleRate
        var chunkChannels: Int = 2

        for chunk in plan.chunks {
            let chunkIdx = chunk.index

            progress?(ProgressiveProgressUpdate(
                chunkIndex: chunkIdx, totalChunks: totalChunks,
                chunkProgress: 0, overallProgress: Double(chunkIdx) / Double(totalChunks),
                message: "Starting chunk \(chunkIdx + 1)/\(totalChunks)"
            ))

            // Build per-chunk params.
            // ALL chunks use text2music with full inference steps.
            // Musical continuity between chunks is achieved through the
            // equal-power crossfade at overlap boundaries during stitching.
            var chunkParams = params
            chunkParams.lyrics = chunk.lyrics
            chunkParams.duration = chunk.durationSec
            chunkParams.taskType = "text2music"
            chunkParams.srcAudio = nil
            chunkParams.repaintingStart = 0
            chunkParams.repaintingEnd = -1
            chunkParams.coverNoiseStrength = 0
            chunkParams.audioCoverStrength = 1.0

            // Generate this chunk
            let chunkResult = try await generateChunk(chunkParams, config, chunkIdx) { frac, msg in
                let overallBase = Double(chunkIdx) / Double(totalChunks)
                let overallFrac = overallBase + frac / Double(totalChunks)
                progress?(ProgressiveProgressUpdate(
                    chunkIndex: chunkIdx, totalChunks: totalChunks,
                    chunkProgress: frac, overallProgress: overallFrac,
                    message: "Chunk \(chunkIdx + 1)/\(totalChunks): \(msg)"
                ))
            }

            guard chunkResult.success else {
                throw OrchestratorError.chunkGenerationFailed(
                    index: chunkIdx,
                    message: chunkResult.error ?? "Unknown error"
                )
            }

            guard let firstAudio = chunkResult.audios.first,
                  let tensor = firstAudio["tensor"] as? [Float],
                  let sr = firstAudio["sample_rate"] as? Int else {
                throw OrchestratorError.noAudioInChunkResult(index: chunkIdx)
            }
            let channels = firstAudio["channels"] as? Int ?? 1
            chunkSampleRate = sr
            chunkChannels = channels
            chunkAudios.append(tensor)
        }

        // Stitch all chunks together
        progress?(ProgressiveProgressUpdate(
            chunkIndex: totalChunks - 1, totalChunks: totalChunks,
            chunkProgress: 1.0, overallProgress: 0.98,
            message: "Stitching \(totalChunks) chunks"
        ))

        let chunkStarts = plan.chunks.map { $0.startSec }
        let stitched = AudioStitcher.stitch(
            chunks: chunkAudios,
            sampleRate: chunkSampleRate,
            channels: chunkChannels,
            chunkStartsSec: chunkStarts,
            overlapSec: plan.overlapSec,
            crossfadeSec: plan.crossfadeSec
        )

        progress?(ProgressiveProgressUpdate(
            chunkIndex: totalChunks - 1, totalChunks: totalChunks,
            chunkProgress: 1.0, overallProgress: 1.0,
            message: "Progressive generation complete"
        ))

        return GenerationResult(
            audios: [
                [
                    "tensor": stitched,
                    "sample_rate": chunkSampleRate,
                    "channels": chunkChannels
                ] as [String: Any]
            ],
            statusMessage: "Generated \(plan.totalDuration)s song in \(totalChunks) chunks",
            extraOutputs: ["progressive_chunks": totalChunks],
            success: true,
            error: nil
        )
    }
}
