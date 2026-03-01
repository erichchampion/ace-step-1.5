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
    public static let defaultSampleRate = 51200

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
