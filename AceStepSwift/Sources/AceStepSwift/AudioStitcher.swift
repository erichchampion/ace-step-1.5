/**
 Assembles progressive generation chunks into a seamless audio output
 using crossfade at overlap boundaries.
 */

import Foundation

public enum AudioStitcher {

    /// Assembles chunks with crossfade at overlap boundaries.
    ///
    /// Each chunk's audio starts at its `startSec` in the final timeline.
    /// Consecutive chunks overlap by `overlapSec`; within that overlap,
    /// a crossfade of width `crossfadeSec` blends the outgoing and incoming
    /// audio using an equal-power (cosine) curve.
    ///
    /// - Parameters:
    ///   - chunks: Per-chunk audio samples (interleaved for stereo: L0 R0 L1 R1 …).
    ///   - sampleRate: Audio sample rate (e.g. 48000).
    ///   - channels: Number of audio channels (1=mono, 2=stereo interleaved).
    ///   - chunkStartsSec: Absolute start time of each chunk in the final timeline.
    ///   - overlapSec: Overlap duration between consecutive chunks.
    ///   - crossfadeSec: Crossfade width at splice points (≤ overlapSec).
    /// - Returns: Stitched audio samples for the complete timeline.
    public static func stitch(
        chunks: [[Float]],
        sampleRate: Int,
        channels: Int,
        chunkStartsSec: [Double],
        overlapSec: Double,
        crossfadeSec: Double
    ) -> [Float] {
        guard !chunks.isEmpty else { return [] }
        guard chunks.count > 1 else { return chunks[0] }

        let effectiveCrossfade = min(crossfadeSec, overlapSec)

        // Determine total output length from last chunk's end
        let lastChunkIdx = chunks.count - 1
        let lastChunkSamples = chunks[lastChunkIdx].count / channels
        let lastChunkEndSample = secondsToFrames(chunkStartsSec[lastChunkIdx], sampleRate: sampleRate) + lastChunkSamples
        let totalFrames = lastChunkEndSample
        var output = [Float](repeating: 0, count: totalFrames * channels)

        for (chunkIdx, chunkAudio) in chunks.enumerated() {
            let chunkStartFrame = secondsToFrames(chunkStartsSec[chunkIdx], sampleRate: sampleRate)
            let chunkFrames = chunkAudio.count / channels

            if chunkIdx == 0 {
                // First chunk: copy directly
                let copyCount = min(chunkAudio.count, output.count - chunkStartFrame * channels)
                for i in 0..<copyCount {
                    output[chunkStartFrame * channels + i] = chunkAudio[i]
                }
            } else {
                let crossfadeFrames = secondsToFrames(effectiveCrossfade, sampleRate: sampleRate)
                let overlapFrames = secondsToFrames(overlapSec, sampleRate: sampleRate)
                let actualFrames = min(chunkFrames, totalFrames - chunkStartFrame)
                
                // 1. Crossfade Region
                let crossfadeLimit = min(actualFrames, crossfadeFrames)
                for frame in 0..<crossfadeLimit {
                    let globalFrame = chunkStartFrame + frame
                    let t = Float(frame) / Float(max(1, crossfadeFrames))
                    // Raised-cosine (equal-gain) crossfade: preserves amplitude for correlated signals
                    // and maintains smooth envelope shape.
                    let alphaSin = sinf(t * .pi / 2.0)
                    let fadeOut = 1.0 - (alphaSin * alphaSin) // Equivalent to cos^2
                    let fadeIn = alphaSin * alphaSin

                    for ch in 0..<channels {
                        let outputIdx = globalFrame * channels + ch
                        let inputIdx = frame * channels + ch
                        if outputIdx < output.count && inputIdx < chunkAudio.count {
                            output[outputIdx] = output[outputIdx] * fadeOut + chunkAudio[inputIdx] * fadeIn
                        }
                    }
                }
                
                // 2. Overlap Region (Past crossfade, fully faded in)
                let overlapLimit = min(actualFrames, overlapFrames)
                if overlapLimit > crossfadeLimit {
                    for frame in crossfadeLimit..<overlapLimit {
                        let globalFrame = chunkStartFrame + frame
                        for ch in 0..<channels {
                            let outputIdx = globalFrame * channels + ch
                            let inputIdx = frame * channels + ch
                            if outputIdx < output.count && inputIdx < chunkAudio.count {
                                output[outputIdx] = chunkAudio[inputIdx]
                            }
                        }
                    }
                }
                
                // 3. Exclusive Region (Past overlap, pure new chunk)
                if actualFrames > overlapLimit {
                    for frame in overlapLimit..<actualFrames {
                        let globalFrame = chunkStartFrame + frame
                        for ch in 0..<channels {
                            let outputIdx = globalFrame * channels + ch
                            let inputIdx = frame * channels + ch
                            if outputIdx < output.count && inputIdx < chunkAudio.count {
                                output[outputIdx] = chunkAudio[inputIdx]
                            }
                        }
                    }
                }
            }
        }

        return output
    }

    /// Converts seconds to sample frames.
    private static func secondsToFrames(_ sec: Double, sampleRate: Int) -> Int {
        Int((sec * Double(sampleRate)).rounded())
    }
}
