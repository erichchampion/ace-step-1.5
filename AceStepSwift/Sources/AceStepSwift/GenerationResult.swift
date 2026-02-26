/**
 Result of music generation. Mirrors acestep/inference.py GenerationResult.
 */

import Foundation

public struct GenerationResult {
    public var audios: [[String: Any]]
    public var statusMessage: String
    public var extraOutputs: [String: Any]
    public var success: Bool
    public var error: String?

    public init(
        audios: [[String: Any]] = [],
        statusMessage: String = "",
        extraOutputs: [String: Any] = [:],
        success: Bool = true,
        error: String? = nil
    ) {
        self.audios = audios
        self.statusMessage = statusMessage
        self.extraOutputs = extraOutputs
        self.success = success
        self.error = error
    }
}
