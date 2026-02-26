/**
 FSM states for constrained decoding. Mirrors acestep/constrained_logits_processor.py FSMState.
 */

public enum FSMState {
    case thinkTag
    case newlineAfterThink
    case bpmName
    case bpmValue
    case newlineAfterBpm
    case captionName
    case captionValue
    case durationName
    case durationValue
    case newlineAfterDuration
    case genresName
    case genresValue
    case newlineAfterGenres
    case keyscaleName
    case keyscaleValue
    case newlineAfterKeyscale
    case languageName
    case languageValue
    case timesigName
    case timesigValue
    case newlineAfterTimesig
    case thinkEndTag
    case codesGeneration
    case completed
}
