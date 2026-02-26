/**
 Protocol for LLM-backed format (caption/lyrics → metadata). Allows mock for tests and real MLX LLM later.
 */

import Foundation

public protocol LLMFormatProvider: AnyObject {
    var isInitialized: Bool { get }
    func generateFormat(caption: String, lyrics: String, userMetadata: [String: Any]?, temperature: Double) async throws -> String
}
