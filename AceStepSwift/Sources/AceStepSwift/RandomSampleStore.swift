/**
 Bundled example presets for create_random_sample. No file I/O; data is embedded.
 */

import Foundation

enum RandomSampleStore {
    static let simpleModeSamples: [RandomSampleResult] = [
        RandomSampleResult(
            caption: "aggressive thrash metal with shredding guitar solos and double bass drums",
            lyrics: "[Instrumental]",
            instrumental: false,
            vocalLanguage: "en",
            success: true
        ),
        RandomSampleResult(
            caption: "a soft Bengali love song for a quiet evening",
            lyrics: "[Instrumental]",
            instrumental: false,
            vocalLanguage: "bn",
            success: true
        ),
        RandomSampleResult(
            caption: "upbeat pop with catchy hooks and clean production",
            lyrics: "[Instrumental]",
            instrumental: true,
            vocalLanguage: "en",
            success: true
        ),
    ]

    static let customModeSamples: [RandomSampleResult] = [
        RandomSampleResult(
            caption: "A classic, gritty hip-hop track built on a steady boom-bap drum machine groove and a deep, foundational sub-bass line.",
            lyrics: "[Intro]\nHey\n\n[Verse 1]\nPa' la güera en el hospital\nDonde el 11 de 2001 suena criminal\n\n[Outro]\n[Instrumental fades out]",
            bpm: 100,
            duration: 124,
            keyscale: "F# minor",
            language: "es",
            timesignature: "4",
            instrumental: false,
            vocalLanguage: "es",
            success: true
        ),
        RandomSampleResult(
            caption: "Ambient electronic with soft pads and subtle percussion.",
            lyrics: "[Instrumental]",
            bpm: 90,
            duration: 60,
            keyscale: "C major",
            language: "en",
            timesignature: "4",
            instrumental: true,
            vocalLanguage: "unknown",
            success: true
        ),
        RandomSampleResult(
            caption: "Acoustic folk ballad with fingerpicked guitar and warm vocals.",
            lyrics: "[Verse 1]\nWalking down the road alone\nWhere the river meets the stone\n\n[Chorus]\nAnd I will wait for you\nTill the morning dew",
            bpm: 72,
            duration: 180,
            keyscale: "G major",
            language: "en",
            timesignature: "4",
            instrumental: false,
            vocalLanguage: "en",
            success: true
        ),
    ]

    /// Picks a random preset from the given list. Returns failure if empty. Used by pickRandom(sampleType:) and by tests for zero-examples path.
    static func pickRandom(from samples: [RandomSampleResult], sampleTypeLabel: String = "unknown") -> RandomSampleResult {
        guard !samples.isEmpty else {
            return RandomSampleResult(
                success: false,
                error: "No example data available for sample type: \(sampleTypeLabel)"
            )
        }
        let index = Int.random(in: 0 ..< samples.count)
        return samples[index]
    }

    /// Picks a random preset for the given sample type. Returns failure if no examples exist.
    static func pickRandom(sampleType: String) -> RandomSampleResult {
        let samples: [RandomSampleResult]
        switch sampleType {
        case "custom_mode":
            samples = customModeSamples
        default:
            samples = simpleModeSamples
        }
        return pickRandom(from: samples, sampleTypeLabel: sampleType)
    }
}
