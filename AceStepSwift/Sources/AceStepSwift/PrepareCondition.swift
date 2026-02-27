/**
 Swift port of Python prepare_condition (context path + API for precomputed encoder).
 Builds context_latents from src_latents and chunk_masks; accepts optional precomputed
 encoder_hidden_states and optional cover hints (precomputed LM hints).
 */

import Foundation
import MLX

/// Inputs for building DiT conditions, mirroring Python prepare_condition arguments.
/// Encoder hidden states can be supplied by the app (e.g. from Python) or produced in Swift via conditionEncoder + textHiddenStates.
public struct PrepareConditionInputs {
    /// Source latents [B, T, 64]. Used for context_latents; optionally replaced by LM hints where is_covers.
    public var srcLatents: MLXArray
    /// Chunk masks: [B, T] (bool or float) or [B, T, 64]. Expanded to [B, T, 64] if 2D. Matches Python unsqueeze(-1).repeat(1,1,64).
    public var chunkMasks: MLXArray
    /// Precomputed encoder hidden states [B, encL, 2048]. When nil and conditionEncoder+textHiddenStates are used, encoder runs in Swift.
    public var precomputedEncoderHiddenStates: MLXArray?
    /// Precomputed encoder attention mask [B, encL]. Optional; DiT decoder currently uses nil.
    public var precomputedEncoderAttentionMask: MLXArray?
    /// Text hidden states [B, L_text, text_hidden_dim] for Swift condition encoder. When set with conditionEncoder, encoder runs in Swift.
    public var textHiddenStates: MLXArray?
    /// Text attention mask [B, L_text]. Used as encoder_attention_mask when encoder runs from textHiddenStates.
    public var textAttentionMask: MLXArray?
    /// Lyric hidden states [B, L_lyric, text_hidden_dim] for full condition encoder (optional).
    public var lyricHiddenStates: MLXArray?
    /// Lyric attention mask [B, L_lyric] (1=valid). Required when lyricHiddenStates is set.
    public var lyricAttentionMask: MLXArray?
    /// Reference audio packed [N, T, timbre_hidden_dim] for full condition encoder (optional).
    public var referAudioPacked: MLXArray?
    /// Reference audio order mask [N] (values 0..<B-1). Required when referAudioPacked is set.
    public var referAudioOrderMask: MLXArray?
    /// Precomputed LM hints at 25 Hz [B, T, 64] for cover path. When non-nil and isCovers is used, replaces src_latents where is_covers.
    public var precomputedLmHints25Hz: MLXArray?
    /// [B] float/bool: 1 where item is cover, 0 otherwise. Used with precomputedLmHints25Hz to select LM hints vs src_latents.
    public var isCovers: MLXArray?
    /// Reserved for future tokenize path. Unused in Phase 1.
    public var silenceLatent: MLXArray?
    /// Optional null condition embedding [1, 1, D] for CFG. Passed through to DiTConditions.
    public var nullConditionEmbedding: MLXArray?
    /// Optional initial latents for cover/repaint (start from this instead of noise). Passed through to DiTConditions.
    public var initialLatents: MLXArray?

    public init(
        srcLatents: MLXArray,
        chunkMasks: MLXArray,
        precomputedEncoderHiddenStates: MLXArray? = nil,
        precomputedEncoderAttentionMask: MLXArray? = nil,
        textHiddenStates: MLXArray? = nil,
        textAttentionMask: MLXArray? = nil,
        lyricHiddenStates: MLXArray? = nil,
        lyricAttentionMask: MLXArray? = nil,
        referAudioPacked: MLXArray? = nil,
        referAudioOrderMask: MLXArray? = nil,
        precomputedLmHints25Hz: MLXArray? = nil,
        isCovers: MLXArray? = nil,
        silenceLatent: MLXArray? = nil,
        nullConditionEmbedding: MLXArray? = nil,
        initialLatents: MLXArray? = nil
    ) {
        self.srcLatents = srcLatents
        self.chunkMasks = chunkMasks
        self.precomputedEncoderHiddenStates = precomputedEncoderHiddenStates
        self.precomputedEncoderAttentionMask = precomputedEncoderAttentionMask
        self.textHiddenStates = textHiddenStates
        self.textAttentionMask = textAttentionMask
        self.lyricHiddenStates = lyricHiddenStates
        self.lyricAttentionMask = lyricAttentionMask
        self.referAudioPacked = referAudioPacked
        self.referAudioOrderMask = referAudioOrderMask
        self.precomputedLmHints25Hz = precomputedLmHints25Hz
        self.isCovers = isCovers
        self.silenceLatent = silenceLatent
        self.nullConditionEmbedding = nullConditionEmbedding
        self.initialLatents = initialLatents
    }
}

/// Context latent dimension (64) for expansion of chunk masks. Matches Python target_latents.shape[2].
private let contextLatentChannels = 64

/// Build context_latents [B, T, 128] = concat(src_latents [B, T, 64], chunk_masks [B, T, 64], axis: 2).
/// If chunkMasks has shape [B, T], expands to [B, T, 64] to match Python unsqueeze(-1).repeat(1, 1, 64).
public func buildContextLatents(srcLatents: MLXArray, chunkMasks: MLXArray) -> MLXArray {
    var masks = chunkMasks
    if masks.ndim == 2 {
        let (b, t) = (masks.dim(0), masks.dim(1))
        masks = masks.expandedDimensions(axis: 2)
        masks = MLX.broadcast(masks, to: [b, t, contextLatentChannels])
    }
    let dtype = srcLatents.dtype
    if masks.dtype != dtype {
        masks = masks.asType(dtype)
    }
    return concatenated([srcLatents, masks], axis: 2)
}

/// Where is_covers > 0 use lm_hints_25Hz, else src_latents. isCovers: [B], broadcast to [B, T, 64].
/// Returns effective src_latents for context building (cover path without tokenize/detokenize).
public func applyCoverHints(srcLatents: MLXArray, lmHints25Hz: MLXArray, isCovers: MLXArray) -> MLXArray {
    let b = srcLatents.dim(0)
    let t = srcLatents.dim(1)
    let c = srcLatents.dim(2)
    let coverExpanded = isCovers.expandedDimensions(axis: 1).expandedDimensions(axis: 2)
    let coverBroadcast = MLX.broadcast(coverExpanded, to: [b, t, c])
    return MLX.where(coverBroadcast .> 0, lmHints25Hz, srcLatents)
}

/// Build a full chunk mask [B, T] of ones (all positions valid). Use for text2music when no repaint/cover mask.
public func fullChunkMask(batchSize: Int, latentLength: Int) -> MLXArray {
    MLXArray.ones([batchSize, latentLength])
}

/// Prepare DiT conditions from inputs. Builds context_latents; uses precomputed encoder and optional cover hints when provided.
public func prepareCondition(inputs: PrepareConditionInputs) -> DiTConditions {
    prepareCondition(inputs: inputs, conditionEncoder: nil)
}

/// Prepare DiT conditions, optionally running the Swift condition encoder when text hidden states are provided.
/// When conditionEncoder is non-nil and inputs.textHiddenStates is non-nil, encoder hidden states are produced in Swift (text-only path); otherwise precomputed values are used.
public func prepareCondition(inputs: PrepareConditionInputs, conditionEncoder: ConditionEncoder?) -> DiTConditions {
    var srcEffective = inputs.srcLatents
    if let lmHints = inputs.precomputedLmHints25Hz, let isCovers = inputs.isCovers {
        let t = inputs.srcLatents.dim(1)
        let crop = lmHints[0..<lmHints.dim(0), 0..<t, 0..<lmHints.dim(2)]
        srcEffective = applyCoverHints(srcLatents: inputs.srcLatents, lmHints25Hz: crop, isCovers: isCovers)
    }
    let contextLatents = buildContextLatents(srcLatents: srcEffective, chunkMasks: inputs.chunkMasks)

    let encoderHiddenStates: MLXArray? = {
        if let encoder = conditionEncoder, let textHidden = inputs.textHiddenStates {
            let (enc, _) = encoder.call(
                textHiddenStates: textHidden,
                textAttentionMask: inputs.textAttentionMask,
                lyricHiddenStates: inputs.lyricHiddenStates,
                lyricAttentionMask: inputs.lyricAttentionMask,
                referAudioPacked: inputs.referAudioPacked,
                referAudioOrderMask: inputs.referAudioOrderMask
            )
            return enc
        }
        return inputs.precomputedEncoderHiddenStates
    }()

    return DiTConditions(
        encoderHiddenStates: encoderHiddenStates,
        contextLatents: contextLatents,
        nullConditionEmbedding: inputs.nullConditionEmbedding,
        initialLatents: inputs.initialLatents
    )
}
