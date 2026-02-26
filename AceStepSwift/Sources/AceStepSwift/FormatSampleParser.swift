/**
 Parses LLM output (<think>...</think>) into metadata dict and lyrics. Mirrors parse_lm_output and _extract_lyrics_from_output.
 */

import Foundation

public enum FormatSampleParser {

    /// Parse LM output to extract metadata (from <think> block) and audio codes. Returns (metadata, lyrics).
    public static func parseLMOutput(_ outputText: String) -> ([String: Any], String) {
        var metadata: [String: Any] = [:]
        let lyrics = extractLyricsFromOutput(outputText)

        guard let reasoning = extractReasoningBlock(outputText) else {
            return (metadata, lyrics)
        }

        let lines = reasoning.split(separator: "\n", omittingEmptySubsequences: false)
        var currentKey: String?
        var currentValueLines: [String] = []

        func saveCurrentField() {
            guard let key = currentKey, !currentValueLines.isEmpty else { return }
            let value = currentValueLines.joined(separator: "\n")
            switch key {
            case "bpm":
                metadata["bpm"] = Int(value.trimmingCharacters(in: .whitespaces)) ?? value.trimmingCharacters(in: .whitespaces)
            case "caption":
                metadata["caption"] = value.trimmingCharacters(in: .whitespaces)
            case "duration":
                metadata["duration"] = Int(value.trimmingCharacters(in: .whitespaces)) ?? value.trimmingCharacters(in: .whitespaces)
            case "genres":
                metadata["genres"] = value.trimmingCharacters(in: .whitespaces)
            case "keyscale":
                metadata["keyscale"] = value.trimmingCharacters(in: .whitespaces)
            case "language":
                metadata["language"] = value.trimmingCharacters(in: .whitespaces)
            case "timesignature":
                metadata["timesignature"] = value.trimmingCharacters(in: .whitespaces)
            default:
                break
            }
            currentKey = nil
            currentValueLines = []
        }

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("<") { continue }
            if let colonIndex = line.firstIndex(of: ":"), !line.hasPrefix(" ") && !line.hasPrefix("\t") {
                saveCurrentField()
                let key = String(line[..<colonIndex]).trimmingCharacters(in: .whitespaces).lowercased()
                let rest = String(line[line.index(after: colonIndex)...])
                currentKey = key
                currentValueLines = rest.trimmingCharacters(in: .whitespaces).isEmpty ? [] : [rest]
            } else if line.hasPrefix(" ") || line.hasPrefix("\t"), currentKey != nil {
                currentValueLines.append(String(line))
            }
        }
        saveCurrentField()

        return (metadata, lyrics)
    }

    /// Extract lyrics section (everything after </think>), strip # Lyric header.
    public static func extractLyricsFromOutput(_ outputText: String) -> String {
        let tag = "</think>"
        guard let range = outputText.range(of: tag) else { return "" }
        var after = String(outputText[range.upperBound...]).trimmingCharacters(in: .whitespaces)
        if after.isEmpty { return "" }
        let dropHeader: (String) -> String = { s in
            guard let r = s.range(of: "\n") else { return s }
            return String(s[r.upperBound...]).trimmingCharacters(in: .whitespaces)
        }
        if after.lowercased().hasPrefix("# lyric") {
            after = dropHeader(after)
        } else if after.lowercased().hasPrefix("# lyrics") {
            after = dropHeader(after)
        }
        if let imEnd = after.range(of: "<|im_end|>") {
            after = String(after[..<imEnd.lowerBound]).trimmingCharacters(in: .whitespaces)
        }
        return after.trimmingCharacters(in: .whitespaces)
    }

    /// Build FormatSampleResult from raw output and fallback lyrics.
    public static func parseToFormatSampleResult(outputText: String, fallbackLyrics: String) -> FormatSampleResult {
        let (metadata, extractedLyrics) = parseLMOutput(outputText)
        let lyrics = extractedLyrics.isEmpty ? fallbackLyrics : extractedLyrics

        var bpm: Int?
        if let v = metadata["bpm"] as? Int { bpm = v }
        else if let v = metadata["bpm"] as? String, let i = Int(v) { bpm = i }

        var duration: Double?
        if let v = metadata["duration"] as? Int { duration = Double(v) }
        else if let v = metadata["duration"] as? Double { duration = v }
        else if let v = metadata["duration"] as? String, let d = Double(v) { duration = d }

        let caption = (metadata["caption"] as? String) ?? ""
        let keyscale = (metadata["keyscale"] as? String) ?? ""
        let language = (metadata["language"] as? String) ?? ""
        let timesignature = (metadata["timesignature"] as? String) ?? ""

        return FormatSampleResult(
            caption: caption,
            lyrics: lyrics,
            bpm: bpm,
            duration: duration,
            keyscale: keyscale,
            language: language,
            timesignature: timesignature,
            statusMessage: "OK",
            success: true,
            error: nil
        )
    }

    /// Parse create_sample LLM output into CreateSampleResult. Reuses <think> metadata + lyrics extraction.
    public static func parseToCreateSampleResult(outputText: String, instrumental: Bool) -> CreateSampleResult {
        let (metadata, extractedLyrics) = parseLMOutput(outputText)
        let lyrics = extractedLyrics.isEmpty ? (instrumental ? "[Instrumental]" : "") : extractedLyrics

        var bpm: Int?
        if let v = metadata["bpm"] as? Int { bpm = v }
        else if let v = metadata["bpm"] as? String, let i = Int(v) { bpm = i }

        var duration: Double?
        if let v = metadata["duration"] as? Int { duration = Double(v) }
        else if let v = metadata["duration"] as? Double { duration = v }
        else if let v = metadata["duration"] as? String, let d = Double(v) { duration = d }

        let caption = (metadata["caption"] as? String) ?? ""
        let keyscale = (metadata["keyscale"] as? String) ?? ""
        let language = (metadata["language"] as? String) ?? ""
        let timesignature = (metadata["timesignature"] as? String) ?? ""

        return CreateSampleResult(
            caption: caption,
            lyrics: lyrics,
            bpm: bpm,
            duration: duration,
            keyscale: keyscale,
            language: language,
            timesignature: timesignature,
            instrumental: instrumental,
            statusMessage: "OK",
            success: true,
            error: nil
        )
    }

    private static func extractReasoningBlock(_ outputText: String) -> String? {
        let startTags = ["<think>", "`think`", "<reasoning>"]
        let endTags = ["</think>", "`/think`", "</reasoning>"]
        for i in 0..<startTags.count {
            let startTag = startTags[i]
            let endTag = endTags[i]
            guard let startRange = outputText.range(of: startTag),
                  let endRange = outputText.range(of: endTag, range: startRange.upperBound..<outputText.endIndex) else { continue }
            let content = String(outputText[startRange.upperBound..<endRange.lowerBound])
            return content.trimmingCharacters(in: .whitespaces)
        }
        if outputText.contains("<|audio_code_") {
            let before = outputText.components(separatedBy: "<|audio_code_")[0]
            return before.trimmingCharacters(in: .whitespaces)
        }
        return outputText.trimmingCharacters(in: .whitespaces)
    }
}
