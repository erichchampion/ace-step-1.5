/**
 Result of create_sample: LLM-generated caption, lyrics, and metadata from a natural language query.
 Mirrors acestep/inference.py CreateSampleResult.
 */

public struct CreateSampleResult {
    public var caption: String
    public var lyrics: String
    public var bpm: Int?
    public var duration: Double?
    public var keyscale: String
    public var language: String
    public var timesignature: String
    public var instrumental: Bool
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
        instrumental: Bool = false,
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
        self.instrumental = instrumental
        self.statusMessage = statusMessage
        self.success = success
        self.error = error
    }
}
