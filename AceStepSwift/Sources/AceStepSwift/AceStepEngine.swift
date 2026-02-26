/**
 Top-level API for format_sample and generate_music. Cadenza app calls these.
 */

import Foundation

public enum AceStepEngine {

    /// Generate music from params. Returns GenerationResult with audios or error.
    public static func generateMusic(
        params: GenerationParams,
        config: GenerationConfig,
        progress: ((Double, String) -> Void)?,
        pipeline: GenerationPipeline?
    ) -> GenerationResult {
        guard let pipe = pipeline, pipe.isInitialized else {
            return GenerationResult(
                audios: [],
                statusMessage: "Pipeline not initialized",
                extraOutputs: [:],
                success: false,
                error: "Pipeline not initialized"
            )
        }
        var result: GenerationResult?
        let group = DispatchGroup()
        group.enter()
        Task {
            defer { group.leave() }
            do {
                result = try await pipe.run(params: params, config: config, progress: progress)
            } catch {
                result = GenerationResult(
                    audios: [],
                    statusMessage: "\(error)",
                    extraOutputs: [:],
                    success: false,
                    error: "\(error)"
                )
            }
        }
        group.wait()
        return result ?? GenerationResult(
            audios: [],
            statusMessage: "Unknown error",
            extraOutputs: [:],
            success: false,
            error: "Unknown error"
        )
    }

    /// Format caption and lyrics via LLM into structured metadata. Returns FormatSampleResult.
    public static func formatSample(
        caption: String,
        lyrics: String,
        userMetadata: [String: Any]? = nil,
        temperature: Double = 0.85,
        llmProvider: LLMFormatProvider?
    ) -> FormatSampleResult {
        guard let provider = llmProvider, provider.isInitialized else {
            return FormatSampleResult(
                caption: "",
                lyrics: lyrics,
                bpm: nil,
                duration: nil,
                keyscale: "",
                language: "",
                timesignature: "",
                statusMessage: "LLM not initialized",
                success: false,
                error: "LLM not initialized"
            )
        }
        let cap = caption.trimmingCharacters(in: .whitespaces).isEmpty ? "NO USER INPUT" : caption
        let lyr = lyrics.trimmingCharacters(in: .whitespaces).isEmpty ? "[Instrumental]" : lyrics
        var outputText: String?
        let group = DispatchGroup()
        group.enter()
        Task {
            defer { group.leave() }
            do {
                outputText = try await provider.generateFormat(caption: cap, lyrics: lyr, userMetadata: userMetadata, temperature: temperature)
            } catch {
                outputText = nil
            }
        }
        group.wait()
        guard let raw = outputText, !raw.isEmpty else {
            return FormatSampleResult(
                caption: "",
                lyrics: lyr,
                bpm: nil,
                duration: nil,
                keyscale: "",
                language: "",
                timesignature: "",
                statusMessage: "Format generation failed",
                success: false,
                error: "Format generation failed"
            )
        }
        return FormatSampleParser.parseToFormatSampleResult(outputText: raw, fallbackLyrics: lyr)
    }
}
