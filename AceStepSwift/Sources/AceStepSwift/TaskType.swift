/**
 Generation task type. Mirrors acestep/constants.py TASK_TYPES and handler task_type usage.
 */

public enum TaskType: String, CaseIterable {
    case text2music
    case repaint
    case cover
    case lego
    case extract
    case complete

    public init?(string: String) {
        let lower = string.lowercased()
        guard let match = TaskType.allCases.first(where: { $0.rawValue == lower }) else {
            return nil
        }
        self = match
    }

    /// Default instruction when no track/class context is provided.
    public var defaultInstruction: String {
        TaskInstructions.instruction(for: self, trackName: nil, completeTrackClasses: nil)
    }
}
