/**
 Result of create_random_sample: random preset for form filling.
 Mirrors Python API: simple_mode (description, instrumental, vocal_language) and
 custom_mode (caption, lyrics, bpm, duration, keyscale, language, timesignature).
 */

public struct RandomSampleResult {
    public var caption: String
    public var lyrics: String
    public var bpm: Int?
    public var duration: Double?
    public var keyscale: String
    public var language: String
    public var timesignature: String
    public var instrumental: Bool
    public var vocalLanguage: String
    public var success: Bool
    public var error: String?

    public init(
        caption: String = "",
        lyrics: String = "",
        bpm: Int? = nil,
        duration: Double? = nil,
        keyscale: String = "",
        language: String = "",
        timesignature: String = "",
        instrumental: Bool = false,
        vocalLanguage: String = "unknown",
        success: Bool = true,
        error: String? = nil
    ) {
        self.caption = caption
        self.lyrics = lyrics
        self.bpm = bpm
        self.duration = duration
        self.keyscale = keyscale
        self.language = language
        self.timesignature = timesignature
        self.instrumental = instrumental
        self.vocalLanguage = vocalLanguage
        self.success = success
        self.error = error
    }
}
