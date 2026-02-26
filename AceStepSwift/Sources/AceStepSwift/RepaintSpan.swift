/**
 Repaint span math: convert repainting start/end (seconds) to latent indices.
 Mirrors acestep/core/generation/handler/conditioning_masks.py (sample_rate // 1920).
 */

import Foundation

/// Samples per latent step for repaint/cover span calculation. Matches Python (1920).
public let repaintSamplesPerLatentStep = 1920

/// Repaint region in latent indices. Clamped to [0, maxLatentLength).
public struct RepaintSpan {
    public let startLatent: Int
    public let endLatent: Int
    public let isValid: Bool

    /// - Parameters:
    ///   - repaintingStartSec: Start time in seconds.
    ///   - repaintingEndSec: End time in seconds; use -1 or nil for "until end".
    ///   - sampleRate: Audio sample rate (e.g. 48000).
    ///   - maxLatentLength: Maximum latent length (cap end index).
    public init(
        repaintingStartSec: Double,
        repaintingEndSec: Double?,
        sampleRate: Int,
        maxLatentLength: Int
    ) {
        let startSec = max(0, repaintingStartSec)
        let endSec: Double
        if let e = repaintingEndSec, e >= 0 {
            endSec = e
        } else {
            endSec = Double(maxLatentLength * repaintSamplesPerLatentStep) / Double(sampleRate)
        }
        let leftPaddingSec = max(0, -repaintingStartSec)
        let adjustedStartSec = startSec + leftPaddingSec
        let adjustedEndSec = endSec + leftPaddingSec
        var startLatent = Int(adjustedStartSec * Double(sampleRate) / Double(repaintSamplesPerLatentStep))
        var endLatent = Int(adjustedEndSec * Double(sampleRate) / Double(repaintSamplesPerLatentStep))
        startLatent = max(0, min(startLatent, maxLatentLength - 1))
        endLatent = max(startLatent + 1, min(endLatent, maxLatentLength))
        self.startLatent = startLatent
        self.endLatent = endLatent
        self.isValid = endSec > startSec && endLatent > startLatent
    }
}
