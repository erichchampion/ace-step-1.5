/**
 Segments lyrics across progressive generation chunks.
 
 Uses structural markers ([Verse 1], [Chorus], [Bridge], etc.) to assign
 lyric sections proportionally to chunks based on time coverage.
 Falls back to line-by-line proportional splitting for unstructured lyrics.
 */

import Foundation

/// Parsed section of structured lyrics.
public struct LyricSection {
    /// Section header, e.g. "[Verse 1]", "[Chorus]", or nil for unstructured lines.
    public let header: String?
    /// Lines of lyrics in this section (excluding the header).
    public let lines: [String]
}

public enum LyricSegmenter {

    // MARK: - Public API

    /// Segments lyrics for a progressive generation plan.
    ///
    /// - Parameters:
    ///   - lyrics: Full lyrics text.
    ///   - chunks: Ordered chunks from the plan (must have at least 1).
    ///   - totalDuration: Total song duration in seconds.
    /// - Returns: One lyric string per chunk, maintaining structural markers where possible.
    public static func segment(
        lyrics: String,
        chunks: [ProgressiveChunk],
        totalDuration: Double
    ) -> [String] {
        guard !chunks.isEmpty else { return [] }
        guard chunks.count > 1 else { return [lyrics] }

        let trimmed = lyrics.trimmingCharacters(in: .whitespacesAndNewlines)

        // Purely instrumental: every chunk gets [Instrumental]
        if trimmed.isEmpty || trimmed == "[Instrumental]" {
            return chunks.map { _ in "[Instrumental]" }
        }

        let sections = parseSections(from: trimmed)

        if sections.count > 1 || (sections.count == 1 && sections[0].header != nil) {
            return distributeStructured(sections: sections, chunks: chunks, totalDuration: totalDuration)
        } else {
            return distributeLinewise(lyrics: trimmed, chunks: chunks, totalDuration: totalDuration)
        }
    }

    // MARK: - Section Parsing

    /// Regex pattern for structural markers: [Verse 1], [Chorus], [Bridge], [Outro], [Intro], [Pre-Chorus], etc.
    private static let sectionHeaderPattern = try! NSRegularExpression(
        pattern: #"^\[(?:Verse|Chorus|Bridge|Outro|Intro|Pre[- ]?Chorus|Hook|Refrain|Interlude|Instrumental|Solo|Break|Coda|Tag|End|Drop|Build|Breakdown|Riff|Skit|Spoken)(?:\s*\d*)?\]"#,
        options: [.caseInsensitive]
    )

    /// Parse lyrics into structured sections based on header markers.
    static func parseSections(from lyrics: String) -> [LyricSection] {
        let lines = lyrics.components(separatedBy: .newlines)
        var sections: [LyricSection] = []
        var currentHeader: String? = nil
        var currentLines: [String] = []

        for line in lines {
            let trimmedLine = line.trimmingCharacters(in: .whitespaces)
            let range = NSRange(trimmedLine.startIndex..., in: trimmedLine)

            if sectionHeaderPattern.firstMatch(in: trimmedLine, range: range) != nil {
                // Flush previous section
                if currentHeader != nil || !currentLines.isEmpty {
                    sections.append(LyricSection(header: currentHeader, lines: currentLines))
                }
                currentHeader = trimmedLine
                currentLines = []
            } else {
                currentLines.append(line)
            }
        }

        // Flush last section
        if currentHeader != nil || !currentLines.isEmpty {
            sections.append(LyricSection(header: currentHeader, lines: currentLines))
        }

        return sections
    }

    // MARK: - Structured Distribution

    /// Distributes structured lyrics (with section headers) proportionally across chunks.
    /// Each chunk gets the sections whose midpoint falls within its time range,
    /// plus 1 context line from the adjacent chunk for continuity.
    private static func distributeStructured(
        sections: [LyricSection],
        chunks: [ProgressiveChunk],
        totalDuration: Double
    ) -> [String] {
        guard !sections.isEmpty else {
            return chunks.map { _ in "[Instrumental]" }
        }

        // Compute the total "weight" of each section (number of non-empty lines + 1 for header)
        let weights: [Double] = sections.map { s in
            let contentLines = s.lines.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            return Double(contentLines.count + (s.header != nil ? 1 : 0))
        }
        let totalWeight = weights.reduce(0, +)
        guard totalWeight > 0 else {
            return chunks.map { _ in "[Instrumental]" }
        }

        // Assign each section a midpoint in the timeline based on proportional weight
        var sectionMidpoints: [Double] = []
        var accumulatedWeight: Double = 0
        for (i, w) in weights.enumerated() {
            let mid = (accumulatedWeight + w / 2.0) / totalWeight * totalDuration
            sectionMidpoints.append(mid)
            accumulatedWeight += w
            _ = i // silence unused warning
        }

        // Assign sections to chunks based on which chunk's time range contains the midpoint
        var chunkSections: [[Int]] = Array(repeating: [], count: chunks.count)
        for (sIdx, midpoint) in sectionMidpoints.enumerated() {
            var bestChunk = 0
            for (cIdx, chunk) in chunks.enumerated() {
                if midpoint >= chunk.startSec && midpoint < chunk.endSec {
                    bestChunk = cIdx
                    break
                }
                // If midpoint is past all chunks, assign to last
                if cIdx == chunks.count - 1 {
                    bestChunk = cIdx
                }
            }
            chunkSections[bestChunk].append(sIdx)
        }

        // Ensure every chunk has at least something: if a chunk has no sections,
        // it shares the last section from the previous chunk (context)
        for cIdx in 1..<chunks.count {
            if chunkSections[cIdx].isEmpty, let lastPrev = chunkSections[cIdx - 1].last {
                chunkSections[cIdx].append(lastPrev)
            }
        }

        // Build lyric strings
        var result: [String] = []
        for (cIdx, sectionIndices) in chunkSections.enumerated() {
            var parts: [String] = []

            // Add context lines from previous chunk's last section
            if cIdx > 0, let prevLast = chunkSections[cIdx - 1].last {
                let prevSection = sections[prevLast]
                let nonEmptyLines = prevSection.lines.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
                let contextLines = Array(nonEmptyLines.suffix(2))
                parts.append(contentsOf: contextLines)
            }

            for sIdx in sectionIndices {
                let section = sections[sIdx]
                if let header = section.header {
                    parts.append(header)
                }
                parts.append(contentsOf: section.lines)
            }

            // Add context lines for next chunk's first section
            if cIdx < chunks.count - 1, let nextFirst = chunkSections[cIdx + 1].first {
                let nextSection = sections[nextFirst]
                let nonEmptyLines = nextSection.lines.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
                let contextLines = Array(nonEmptyLines.prefix(2))
                parts.append(contentsOf: contextLines)
            }

            result.append(parts.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines))
        }

        return result
    }

    // MARK: - Line-wise Distribution

    /// Distributes unstructured lyrics (plain text) proportionally across chunks by line count.
    private static func distributeLinewise(
        lyrics: String,
        chunks: [ProgressiveChunk],
        totalDuration: Double
    ) -> [String] {
        let lines = lyrics.components(separatedBy: .newlines)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        guard !lines.isEmpty else {
            return chunks.map { _ in "[Instrumental]" }
        }

        let totalLines = lines.count

        // Assign lines proportionally based on chunk time coverage
        var result: [String] = []
        for (cIdx, chunk) in chunks.enumerated() {
            // Proportion of the total timeline this chunk covers (using unique content, not overlap)
            let chunkMidStart: Double
            let chunkMidEnd: Double
            if cIdx == 0 {
                chunkMidStart = 0
                chunkMidEnd = chunk.endSec - chunk.overlapWithPreviousSec / 2.0
            } else if cIdx == chunks.count - 1 {
                chunkMidStart = chunk.startSec + chunk.overlapWithPreviousSec / 2.0
                chunkMidEnd = totalDuration
            } else {
                chunkMidStart = chunk.startSec + chunk.overlapWithPreviousSec / 2.0
                let nextOverlap = chunks[cIdx + 1].overlapWithPreviousSec
                chunkMidEnd = chunk.endSec - nextOverlap / 2.0
            }

            let startFrac = max(0, chunkMidStart / totalDuration)
            let endFrac = min(1, chunkMidEnd / totalDuration)
            let startLine = Int((startFrac * Double(totalLines)).rounded(.down))
            let endLine = min(totalLines, Int((endFrac * Double(totalLines)).rounded(.up)))

            let chunkLines = Array(lines[max(0, startLine)..<max(startLine, endLine)])
            var parts: [String] = []

            // Context: last lines from previous chunk
            if cIdx > 0 && startLine > 0 {
                let ctxStart = max(0, startLine - 2)
                parts.append(contentsOf: lines[ctxStart..<startLine])
            }

            parts.append(contentsOf: chunkLines)

            // Context: first lines of next chunk
            if cIdx < chunks.count - 1 && endLine < totalLines {
                let ctxEnd = min(totalLines, endLine + 2)
                parts.append(contentsOf: lines[endLine..<ctxEnd])
            }

            result.append(parts.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines))
        }

        return result
    }
}
