/**
 Task instruction templates. Mirrors acestep/constants.py TASK_INSTRUCTIONS.
 */

public enum TaskInstructions {
    public static let text2music = "Fill the audio semantic mask based on the given conditions:"
    public static let repaint = "Repaint the mask area based on the given conditions:"
    public static let cover = "Generate audio semantic tokens based on the given conditions:"
    public static let extractDefault = "Extract the track from the audio:"
    public static let legoDefault = "Generate the track based on the audio context:"
    public static let completeDefault = "Complete the input track:"

    /// Instruction for extract: "Extract the {TRACK_NAME} track from the audio:"
    public static func extract(trackName: String) -> String {
        "Extract the \(trackName.uppercased()) track from the audio:"
    }

    /// Instruction for lego: "Generate the {TRACK_NAME} track based on the audio context:"
    public static func lego(trackName: String) -> String {
        "Generate the \(trackName.uppercased()) track based on the audio context:"
    }

    /// Instruction for complete: "Complete the input track with {TRACK_CLASSES}:"
    public static func complete(trackClasses: [String]) -> String {
        guard !trackClasses.isEmpty else { return completeDefault }
        let joined = trackClasses.map { $0.uppercased() }.joined(separator: " | ")
        return "Complete the input track with \(joined):"
    }

    /// Returns the instruction string for the given task type and optional track/class context.
    public static func instruction(
        for taskType: TaskType,
        trackName: String?,
        completeTrackClasses: [String]?
    ) -> String {
        switch taskType {
        case .text2music:
            return text2music
        case .repaint:
            return repaint
        case .cover:
            return cover
        case .extract:
            if let name = trackName, !name.isEmpty { return extract(trackName: name) }
            return extractDefault
        case .lego:
            if let name = trackName, !name.isEmpty { return lego(trackName: name) }
            return legoDefault
        case .complete:
            if let classes = completeTrackClasses, !classes.isEmpty { return complete(trackClasses: classes) }
            return completeDefault
        }
    }
}

// MARK: - GenerationParams helper

import Foundation

extension TaskInstructions {
    /// Returns the task instruction string for the given generation params (task type + legoTrack + completeTrackClasses).
    public static func instruction(for params: GenerationParams) -> String {
        let taskType = TaskType(string: params.taskType) ?? .text2music
        return instruction(for: taskType, trackName: params.legoTrack, completeTrackClasses: params.completeTrackClasses)
    }
}
