/**
 Music generation parameters. Mirrors acestep/inference.py GenerationParams.
 */

public struct GenerationParams {
    public var taskType: String
    public var instruction: String
    public var referenceAudio: String?
    public var srcAudio: String?
    public var audioCodes: String
    public var caption: String
    public var lyrics: String
    public var instrumental: Bool
    public var vocalLanguage: String
    public var bpm: Int?
    public var keyscale: String
    public var timesignature: String
    public var duration: Double
    public var enableNormalization: Bool
    public var normalizationDb: Double
    public var latentShift: Double
    public var latentRescale: Double
    public var inferenceSteps: Int
    public var seed: Int
    public var guidanceScale: Double
    public var useAdg: Bool
    public var cfgIntervalStart: Double
    public var cfgIntervalEnd: Double
    public var shift: Double
    public var inferMethod: String
    public var timesteps: [Double]?
    public var repaintingStart: Double
    public var repaintingEnd: Double
    public var audioCoverStrength: Double
    public var coverNoiseStrength: Double
    /// For lego/extract tasks: track name (e.g. "vocals", "drums"). Used in task instruction.
    public var legoTrack: String?
    /// For complete task: track classes (e.g. ["vocals", "drums"]). Used in task instruction.
    public var completeTrackClasses: [String]?
    public var thinking: Bool
    public var lmTemperature: Double
    public var lmCfgScale: Double
    public var lmTopK: Int
    public var lmTopP: Double
    public var useCotMetas: Bool
    public var useCotCaption: Bool
    public var useCotLanguage: Bool
    public var useConstrainedDecoding: Bool

    public init(
        taskType: String = "text2music",
        instruction: String = "Fill the audio semantic mask based on the given conditions:",
        referenceAudio: String? = nil,
        srcAudio: String? = nil,
        audioCodes: String = "",
        caption: String = "",
        lyrics: String = "",
        instrumental: Bool = false,
        vocalLanguage: String = "unknown",
        bpm: Int? = nil,
        keyscale: String = "",
        timesignature: String = "",
        duration: Double = -1.0,
        enableNormalization: Bool = true,
        normalizationDb: Double = -1.0,
        latentShift: Double = 0.0,
        latentRescale: Double = 1.0,
        inferenceSteps: Int = 8,
        seed: Int = -1,
        guidanceScale: Double = 7.0,
        useAdg: Bool = false,
        cfgIntervalStart: Double = 0.0,
        cfgIntervalEnd: Double = 1.0,
        shift: Double = 1.0,
        inferMethod: String = "ode",
        timesteps: [Double]? = nil,
        repaintingStart: Double = 0.0,
        repaintingEnd: Double = -1,
        audioCoverStrength: Double = 1.0,
        coverNoiseStrength: Double = 0.0,
        legoTrack: String? = nil,
        completeTrackClasses: [String]? = nil,
        thinking: Bool = true,
        lmTemperature: Double = 0.85,
        lmCfgScale: Double = 2.0,
        lmTopK: Int = 0,
        lmTopP: Double = 0.9,
        useCotMetas: Bool = true,
        useCotCaption: Bool = true,
        useCotLanguage: Bool = true,
        useConstrainedDecoding: Bool = true
    ) {
        self.taskType = taskType
        self.instruction = instruction
        self.referenceAudio = referenceAudio
        self.srcAudio = srcAudio
        self.audioCodes = audioCodes
        self.caption = caption
        self.lyrics = lyrics
        self.instrumental = instrumental
        self.vocalLanguage = vocalLanguage
        self.bpm = bpm
        self.keyscale = keyscale
        self.timesignature = timesignature
        self.duration = duration
        self.enableNormalization = enableNormalization
        self.normalizationDb = normalizationDb
        self.latentShift = latentShift
        self.latentRescale = latentRescale
        self.inferenceSteps = inferenceSteps
        self.seed = seed
        self.guidanceScale = guidanceScale
        self.useAdg = useAdg
        self.cfgIntervalStart = cfgIntervalStart
        self.cfgIntervalEnd = cfgIntervalEnd
        self.shift = shift
        self.inferMethod = inferMethod
        self.timesteps = timesteps
        self.repaintingStart = repaintingStart
        self.repaintingEnd = repaintingEnd
        self.audioCoverStrength = audioCoverStrength
        self.coverNoiseStrength = coverNoiseStrength
        self.legoTrack = legoTrack
        self.completeTrackClasses = completeTrackClasses
        self.thinking = thinking
        self.lmTemperature = lmTemperature
        self.lmCfgScale = lmCfgScale
        self.lmTopK = lmTopK
        self.lmTopP = lmTopP
        self.useCotMetas = useCotMetas
        self.useCotCaption = useCotCaption
        self.useCotLanguage = useCotLanguage
        self.useConstrainedDecoding = useConstrainedDecoding
    }

    public static func isBPMValid(_ bpm: Int) -> Bool {
        bpm >= AceStepConstants.bpmMin && bpm <= AceStepConstants.bpmMax
    }

    public static func isDurationValid(_ duration: Double) -> Bool {
        duration >= Double(AceStepConstants.durationMin) && duration <= Double(AceStepConstants.durationMax)
    }
}
