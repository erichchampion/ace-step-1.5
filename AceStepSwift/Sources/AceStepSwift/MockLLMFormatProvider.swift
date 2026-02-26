/**
 Mock LLM provider that returns a fixed string. Used for testing formatSample without loading a model.
 */

import Foundation

public final class MockLLMFormatProvider: LLMFormatProvider {
    public var isInitialized: Bool { true }
    private let responseText: String

    public init(responseText: String) {
        self.responseText = responseText
    }

    public func generateFormat(caption: String, lyrics: String, userMetadata: [String: Any]?, temperature: Double) async throws -> String {
        responseText
    }

    public func generateFromQuery(query: String, instrumental: Bool, vocalLanguage: String?, temperature: Double) async throws -> String {
        responseText
    }
}
