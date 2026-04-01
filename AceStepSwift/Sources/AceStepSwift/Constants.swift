/**
 ACE-Step constants mirrored from acestep/constants.py.
 Used for validation and LLM constrained decoding.
 */

public enum AceStepConstants {
    public static let bpmMin = 30
    public static let bpmMax = 300
    public static let durationMin = 10
    public static let durationMax = 600
    /// Default audio sample rate (Hz). Matches Python handler.
    public static let defaultSampleRate = 48000
    /// Fallback duration (seconds) when user selects "Auto" (duration ≤ 0).
    /// Python's LLM resolves auto-duration via CoT; without LLM, Python uses random(10…120).
    /// We use a deterministic 30s default that works on all device tiers.
    public static let autoDuration: Double = 30.0

    /// Latent channel dimension (DiT/VAE)
    public static let latentChannels = 64
    /// Minimum latent length (matches Python conditioning_target padding for short clips)
    public static let minLatentLength = 128

    public static let validTimeSignatures: [Int] = [2, 3, 4, 6]

    public static let validLanguages: [String] = [
        "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
        "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
        "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
        "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
        "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
        "unknown"
    ]

    public static let keyscaleNotes = ["A", "B", "C", "D", "E", "F", "G"]
    public static let keyscaleModes = ["major", "minor"]
}
