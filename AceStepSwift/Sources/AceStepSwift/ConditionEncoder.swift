/**
 Swift condition encoder: text projector + lyric encoder + timbre encoder + pack_sequences.
 Load encoder.text_projector.*, encoder.lyric_encoder.*, encoder.timbre_encoder.* from checkpoint.
 */

import Foundation
import MLX
import MLXNN

/// Config for the condition encoder. Matches Python AceStepConfig encoder-related fields.
public struct ConditionEncoderConfig {
    public var textHiddenDim: Int
    public var hiddenSize: Int
    public var timbreHiddenDim: Int
    public var numLyricLayers: Int
    public var numTimbreLayers: Int

    public static let turbo = ConditionEncoderConfig(
        textHiddenDim: 1024,
        hiddenSize: 2048,
        timbreHiddenDim: 64,
        numLyricLayers: 8,
        numTimbreLayers: 4
    )

    public init(
        textHiddenDim: Int = 1024,
        hiddenSize: Int = 2048,
        timbreHiddenDim: Int = 64,
        numLyricLayers: Int = 8,
        numTimbreLayers: Int = 4
    ) {
        self.textHiddenDim = textHiddenDim
        self.hiddenSize = hiddenSize
        self.timbreHiddenDim = timbreHiddenDim
        self.numLyricLayers = numLyricLayers
        self.numTimbreLayers = numTimbreLayers
    }
}

/// Full condition encoder: text projector, lyric encoder, timbre encoder, pack_sequences(lyric+timbre, text).
public class ConditionEncoder: Module {
    public let textProjector: Linear
    public let lyricEncoder: LyricEncoder
    public let timbreEncoder: TimbreEncoder
    public let hiddenSize: Int

    public init(config: ConditionEncoderConfig = .turbo) {
        self.hiddenSize = config.hiddenSize
        self.textProjector = Linear(config.textHiddenDim, config.hiddenSize, bias: false)
        self.lyricEncoder = LyricEncoder(
            textHiddenDim: config.textHiddenDim,
            hiddenSize: config.hiddenSize,
            numLayers: config.numLyricLayers
        )
        self.timbreEncoder = TimbreEncoder(
            timbreHiddenDim: config.timbreHiddenDim,
            hiddenSize: config.hiddenSize,
            numLayers: config.numTimbreLayers
        )
        super.init()
    }

    /// Text-only path: project text -> encoder hidden states. Use when lyric/timbre are not available.
    public func call(textHiddenStates: MLXArray, textAttentionMask: MLXArray?) -> (MLXArray, MLXArray?) {
        let encoderHiddenStates = textProjector(textHiddenStates)
        return (encoderHiddenStates, textAttentionMask)
    }

    /// Full path: lyric + timbre + text -> pack_sequences(lyric, timbre) then pack_sequences(., text). Optional lyric/timbre; when both nil return text only (avoids zeros with 0 sequence dim which can be 1D in MLX and break concatenate).
    public func call(
        textHiddenStates: MLXArray,
        textAttentionMask: MLXArray?,
        lyricHiddenStates: MLXArray?,
        lyricAttentionMask: MLXArray?,
        referAudioPacked: MLXArray?,
        referAudioOrderMask: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let B = textHiddenStates.dim(0)
        let textProj = textProjector(textHiddenStates)
        let textMask = textAttentionMask ?? MLXArray.ones([B, textHiddenStates.dim(1)])

        let hasLyric = lyricHiddenStates != nil && lyricAttentionMask != nil
        let hasTimbre = referAudioPacked != nil && referAudioOrderMask != nil

        let (enc, encMask): (MLXArray, MLXArray) = {
            if !hasLyric && !hasTimbre {
                return (textProj, textMask)
            }
            if hasLyric && !hasTimbre {
                let out = lyricEncoder.call(inputsEmbeds: lyricHiddenStates!, attentionMask: lyricAttentionMask!)
                return (out, lyricAttentionMask!)
            }
            if !hasLyric && hasTimbre {
                return timbreEncoder.call(referAudioPacked: referAudioPacked!, referAudioOrderMask: referAudioOrderMask!)
            }
            let lyricOut = lyricEncoder.call(inputsEmbeds: lyricHiddenStates!, attentionMask: lyricAttentionMask!)
            let (timbreOut, timbreMask) = timbreEncoder.call(referAudioPacked: referAudioPacked!, referAudioOrderMask: referAudioOrderMask!)
            return packSequences(hidden1: lyricOut, hidden2: timbreOut, mask1: lyricAttentionMask!, mask2: timbreMask)
        }()

        if !hasLyric && !hasTimbre {
            return (enc, encMask)
        }
        return packSequences(hidden1: enc, hidden2: textProj, mask1: encMask, mask2: textMask)
    }
}
