/**
 Progressive generation plan: splits a long song into overlapping chunks
 that can be generated sequentially within the device's memory budget.
 All chunks use text2music with full inference steps. Musical coherence
 is maintained through shared conditioning and equal-power crossfade
 at overlap boundaries during stitching.
 */

import Foundation

/// A single chunk in a progressive generation plan.
public struct ProgressiveChunk {
    /// Zero-based chunk index.
    public let index: Int
    /// Absolute start time in the final song (seconds).
    public let startSec: Double
    /// Absolute end time in the final song (seconds).
    public let endSec: Double
    /// Duration of overlap with the previous chunk (seconds). Zero for first chunk.
    public let overlapWithPreviousSec: Double
    /// Whether this is the first chunk (uses text2music) or subsequent (uses repaint).
    public let isFirstChunk: Bool
    /// Lyrics assigned to this chunk by `LyricSegmenter`.
    public var lyrics: String

    /// Duration of this chunk in seconds.
    public var durationSec: Double { endSec - startSec }

    public init(
        index: Int,
        startSec: Double,
        endSec: Double,
        overlapWithPreviousSec: Double,
        isFirstChunk: Bool,
        lyrics: String
    ) {
        self.index = index
        self.startSec = startSec
        self.endSec = endSec
        self.overlapWithPreviousSec = overlapWithPreviousSec
        self.isFirstChunk = isFirstChunk
        self.lyrics = lyrics
    }
}

/// Plan describing how to split a long song into progressive chunks.
public struct ProgressiveGenerationPlan {
    /// Total requested duration (seconds).
    public let totalDuration: Double
    /// Ordered list of chunks.
    public let chunks: [ProgressiveChunk]
    /// Overlap duration between consecutive chunks (seconds).
    public let overlapSec: Double
    /// Crossfade duration at splice points (seconds, must be ≤ overlapSec).
    public let crossfadeSec: Double

    /// Whether this plan requires progressive generation (more than 1 chunk).
    public var isProgressive: Bool { chunks.count > 1 }

    /// Estimates how many chunks a given duration would produce without computing a full plan.
    /// Uses the same logic as `plan()` including the small-remainder merge rule.
    public static func estimateChunkCount(
        totalDuration: Double,
        maxChunkDuration: Double,
        overlapSec: Double
    ) -> Int {
        guard totalDuration > maxChunkDuration else { return 1 }
        let advancePerChunk = maxChunkDuration - overlapSec
        guard advancePerChunk > 0 else { return 1 }

        let minUsefulContentSec = 5.0  // must match plan()
        var count = 1
        var currentStart = advancePerChunk  // first chunk ends at maxChunkDuration
        while currentStart < totalDuration {
            let chunkEnd = min(currentStart + maxChunkDuration, totalDuration)
            let remaining = totalDuration - chunkEnd
            count += 1
            if remaining > 0 && remaining < overlapSec + minUsefulContentSec {
                break  // this chunk extends to cover the remaining
            }
            currentStart += advancePerChunk
        }
        return count
    }

    /// Create a progressive generation plan.
    ///
    /// - Parameters:
    ///   - totalDuration: Total requested song duration in seconds.
    ///   - maxChunkDuration: Maximum duration the device can generate in one pass.
    ///   - overlapSec: Overlap between consecutive chunks for continuity.
    ///   - crossfadeSec: Crossfade width at splice points (≤ overlapSec).
    ///   - lyrics: Full lyrics text.
    ///   - caption: Generation caption/prompt (not used for chunking, passed through).
    /// - Returns: A plan with computed chunks and segmented lyrics.
    public static func plan(
        totalDuration: Double,
        maxChunkDuration: Double,
        overlapSec: Double,
        crossfadeSec: Double,
        lyrics: String,
        caption: String
    ) -> ProgressiveGenerationPlan {
        let effectiveCrossfade = min(crossfadeSec, overlapSec)

        // If the total fits in a single chunk, no progressive needed
        guard totalDuration > maxChunkDuration else {
            let chunk = ProgressiveChunk(
                index: 0,
                startSec: 0,
                endSec: totalDuration,
                overlapWithPreviousSec: 0,
                isFirstChunk: true,
                lyrics: lyrics
            )
            return ProgressiveGenerationPlan(
                totalDuration: totalDuration,
                chunks: [chunk],
                overlapSec: overlapSec,
                crossfadeSec: effectiveCrossfade
            )
        }

        // Compute chunks:
        // Chunk 0: [0, maxChunkDuration]
        // Chunk N: [previousEnd - overlapSec, previousEnd - overlapSec + maxChunkDuration]
        // The effective new content per chunk (after the first) is maxChunkDuration - overlapSec.
        let advancePerChunk = maxChunkDuration - overlapSec
        guard advancePerChunk > 0 else {
            // overlapSec >= maxChunkDuration is invalid; fall back to single chunk
            let chunk = ProgressiveChunk(
                index: 0,
                startSec: 0,
                endSec: totalDuration,
                overlapWithPreviousSec: 0,
                isFirstChunk: true,
                lyrics: lyrics
            )
            return ProgressiveGenerationPlan(
                totalDuration: totalDuration,
                chunks: [chunk],
                overlapSec: overlapSec,
                crossfadeSec: effectiveCrossfade
            )
        }

        var chunks: [ProgressiveChunk] = []
        var currentStart: Double = 0

        while currentStart < totalDuration {
            let index = chunks.count
            let isFirst = (index == 0)
            let chunkEnd = min(currentStart + maxChunkDuration, totalDuration)
            let overlap = isFirst ? 0.0 : overlapSec

            chunks.append(ProgressiveChunk(
                index: index,
                startSec: currentStart,
                endSec: chunkEnd,
                overlapWithPreviousSec: overlap,
                isFirstChunk: isFirst,
                lyrics: "" // Will be filled by LyricSegmenter
            ))

            // If the remaining duration after this chunk is less than a minimum useful
            // chunk size (overlapSec + some content), extend this chunk to the end instead
            let nextStart = currentStart + advancePerChunk
            let remaining = totalDuration - chunkEnd
            let minUsefulContentSec = 5.0  // minimum new content to justify a separate chunk
            if remaining > 0 && remaining < overlapSec + minUsefulContentSec {
                // Extend this last chunk to cover the entire duration
                chunks[chunks.count - 1] = ProgressiveChunk(
                    index: index,
                    startSec: currentStart,
                    endSec: totalDuration,
                    overlapWithPreviousSec: overlap,
                    isFirstChunk: isFirst,
                    lyrics: ""
                )
                break
            }

            currentStart = nextStart
        }

        // Segment lyrics across chunks
        let segmented = LyricSegmenter.segment(
            lyrics: lyrics,
            chunks: chunks,
            totalDuration: totalDuration
        )
        for i in chunks.indices {
            chunks[i].lyrics = segmented[i]
        }

        return ProgressiveGenerationPlan(
            totalDuration: totalDuration,
            chunks: chunks,
            overlapSec: overlapSec,
            crossfadeSec: effectiveCrossfade
        )
    }
}
