/**
 FSM state transitions for constrained decoding. Mirrors constrained_logits_processor._build_state_transitions.
 Canonical field order: bpm -> caption -> duration -> genres -> keyscale -> language -> timesignature.
 */

import Foundation

/// Options for which metadata fields to include in the FSM (matches Python skip_* flags).
public struct FSMTransitionOptions {
    public var skipCaption: Bool
    public var skipGenres: Bool
    public var skipLanguage: Bool

    public init(skipCaption: Bool = false, skipGenres: Bool = true, skipLanguage: Bool = false) {
        self.skipCaption = skipCaption
        self.skipGenres = skipGenres
        self.skipLanguage = skipLanguage
    }
}

/// Returns the next FSM state after completing a field value (e.g. after generating "120" for bpm).
/// Field order: bpm, caption, duration, genres, keyscale, language, timesignature.
public func nextState(after state: FSMState, options: FSMTransitionOptions = FSMTransitionOptions()) -> FSMState? {
    switch state {
    case .thinkTag: return .newlineAfterThink
    case .newlineAfterThink: return .bpmName
    case .bpmName: return .bpmValue
    case .bpmValue: return nextFieldAfter(field: "bpm", options: options)
    case .newlineAfterBpm: return .captionName
    case .captionName: return .captionValue
    case .captionValue: return nextFieldAfter(field: "caption", options: options)
    case .durationName: return .durationValue
    case .durationValue: return nextFieldAfter(field: "duration", options: options)
    case .newlineAfterDuration: return .genresName
    case .genresName: return .genresValue
    case .genresValue: return nextFieldAfter(field: "genres", options: options)
    case .newlineAfterGenres: return .keyscaleName
    case .keyscaleName: return .keyscaleValue
    case .keyscaleValue: return nextFieldAfter(field: "keyscale", options: options)
    case .newlineAfterKeyscale: return .languageName
    case .languageName: return .languageValue
    case .languageValue: return nextFieldAfter(field: "language", options: options)
    case .timesigName: return .timesigValue
    case .timesigValue: return .thinkEndTag
    case .newlineAfterTimesig: return .thinkEndTag
    case .thinkEndTag: return .codesGeneration
    case .codesGeneration: return .completed
    case .completed: return nil
    }
}

private let fieldOrder = ["bpm", "caption", "duration", "genres", "keyscale", "language", "timesignature"]

private func nextFieldAfter(field: String, options: FSMTransitionOptions) -> FSMState {
    guard let idx = fieldOrder.firstIndex(of: field) else { return .thinkEndTag }
    for i in (idx + 1)..<fieldOrder.count {
        let f = fieldOrder[i]
        if f == "genres" && options.skipGenres { continue }
        if f == "caption" && options.skipCaption { continue }
        if f == "language" && options.skipLanguage { continue }
        return nameState(for: f)
    }
    return .thinkEndTag
}

private func nameState(for field: String) -> FSMState {
    switch field {
    case "bpm": return .bpmName
    case "caption": return .captionName
    case "duration": return .durationName
    case "genres": return .genresName
    case "keyscale": return .keyscaleName
    case "language": return .languageName
    case "timesignature": return .timesigName
    default: return .thinkEndTag
    }
}
