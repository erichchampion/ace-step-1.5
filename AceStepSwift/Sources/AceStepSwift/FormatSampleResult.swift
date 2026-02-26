/**
 Result of format_sample (LLM caption/lyrics formatting). Mirrors acestep/inference.py FormatSampleResult.
 */

public struct FormatSampleResult {
    public var caption: String
    public var lyrics: String
    public var bpm: Int?
    public var duration: Double?
    public var keyscale: String
    public var language: String
    public var timesignature: String
    public var statusMessage: String
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
        statusMessage: String = "",
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
        self.statusMessage = statusMessage
        self.success = success
        self.error = error
    }
}
