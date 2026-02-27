/**
 Tests for prepare_condition Swift port: buildContextLatents, fullChunkMask, prepareCondition, applyCoverHints.
 Validates shapes and behavior against Python contract (context_latents [B, T, 128], encoder [B, L, 2048]).
 */

import XCTest
import MLX
@testable import AceStepSwift

final class PrepareConditionTests: XCTestCase {

    // MARK: - buildContextLatents

    func testBuildContextLatentsWith2DChunkMasksOutputShape() {
        let b = 2
        let t = 8
        let srcLatents = MLXArray.zeros([b, t, 64])
        let chunkMasks = MLXArray.ones([b, t])
        let out = buildContextLatents(srcLatents: srcLatents, chunkMasks: chunkMasks)
        XCTAssertEqual(out.dim(0), b)
        XCTAssertEqual(out.dim(1), t)
        XCTAssertEqual(out.dim(2), 128)
    }

    func testBuildContextLatentsWith3DChunkMasksOutputShape() {
        let b = 1
        let t = 4
        let srcLatents = MLXArray.zeros([b, t, 64])
        let chunkMasks = MLXArray.ones([b, t, 64])
        let out = buildContextLatents(srcLatents: srcLatents, chunkMasks: chunkMasks)
        XCTAssertEqual(out.dim(0), b)
        XCTAssertEqual(out.dim(1), t)
        XCTAssertEqual(out.dim(2), 128)
    }

    func testBuildContextLatentsChunkMaskDtypeCast() {
        let b = 1
        let t = 4
        let srcLatents = MLXArray.zeros([b, t, 64])
        let chunkMasks = MLXArray.ones([b, t]).asType(.int32)
        let out = buildContextLatents(srcLatents: srcLatents, chunkMasks: chunkMasks)
        XCTAssertEqual(out.dtype, srcLatents.dtype)
        XCTAssertEqual(out.shape, [b, t, 128])
    }

    // MARK: - fullChunkMask

    func testFullChunkMaskShapeAndOnes() {
        let batchSize = 3
        let latentLength = 10
        let mask = fullChunkMask(batchSize: batchSize, latentLength: latentLength)
        XCTAssertEqual(mask.dim(0), batchSize)
        XCTAssertEqual(mask.dim(1), latentLength)
        let first = mask[0..<1, 0..<1].item(Float.self)
        XCTAssertEqual(first, 1.0)
    }

    // MARK: - prepareCondition (inputs only)

    func testPrepareConditionWithOnlySrcAndChunkMasks() {
        let b = 2
        let t = 6
        let srcLatents = MLXArray.zeros([b, t, 64])
        let chunkMasks = fullChunkMask(batchSize: b, latentLength: t)
        let inputs = PrepareConditionInputs(srcLatents: srcLatents, chunkMasks: chunkMasks)
        let conditions = prepareCondition(inputs: inputs)
        XCTAssertNotNil(conditions.contextLatents)
        XCTAssertEqual(conditions.contextLatents!.dim(0), b)
        XCTAssertEqual(conditions.contextLatents!.dim(1), t)
        XCTAssertEqual(conditions.contextLatents!.dim(2), 128)
        XCTAssertNil(conditions.encoderHiddenStates)
    }

    func testPrepareConditionWithPrecomputedEncoder() {
        let b = 1
        let t = 4
        let encL = 5
        let srcLatents = MLXArray.zeros([b, t, 64])
        let chunkMasks = fullChunkMask(batchSize: b, latentLength: t)
        let enc = MLXArray.zeros([b, encL, 2048])
        let inputs = PrepareConditionInputs(
            srcLatents: srcLatents,
            chunkMasks: chunkMasks,
            precomputedEncoderHiddenStates: enc
        )
        let conditions = prepareCondition(inputs: inputs)
        XCTAssertNotNil(conditions.encoderHiddenStates)
        XCTAssertEqual(conditions.encoderHiddenStates!.dim(0), b)
        XCTAssertEqual(conditions.encoderHiddenStates!.dim(1), encL)
        XCTAssertEqual(conditions.encoderHiddenStates!.dim(2), 2048)
        XCTAssertEqual(conditions.contextLatents!.shape, [b, t, 128])
    }

    // MARK: - prepareCondition with conditionEncoder (text-only)

    func testPrepareConditionWithConditionEncoderTextOnly() {
        let b = 1
        let t = 4
        let L = 6
        let encoder = ConditionEncoder(config: .turbo)
        let srcLatents = MLXArray.zeros([b, t, 64])
        let chunkMasks = fullChunkMask(batchSize: b, latentLength: t)
        let textHidden = MLXArray.zeros([b, L, 1024])
        let inputs = PrepareConditionInputs(
            srcLatents: srcLatents,
            chunkMasks: chunkMasks,
            textHiddenStates: textHidden
        )
        let conditions = prepareCondition(inputs: inputs, conditionEncoder: encoder)
        XCTAssertNotNil(conditions.encoderHiddenStates)
        XCTAssertEqual(conditions.encoderHiddenStates!.dim(0), b)
        XCTAssertEqual(conditions.encoderHiddenStates!.dim(1), L)
        XCTAssertEqual(conditions.encoderHiddenStates!.dim(2), 2048)
        XCTAssertEqual(conditions.contextLatents!.shape, [b, t, 128])
    }

    // MARK: - applyCoverHints

    func testApplyCoverHintsShapeAndWhereBehavior() {
        let b = 2
        let t = 4
        let c = 64
        let srcLatents = MLXArray.zeros([b, t, c])
        let lmHints = MLXArray.ones([b, t, c])
        let isCovers = MLXArray([Float(1), 0])
        let out = applyCoverHints(srcLatents: srcLatents, lmHints25Hz: lmHints, isCovers: isCovers)
        XCTAssertEqual(out.shape, [b, t, c])
        let row0First = out[0..<1, 0..<1, 0..<1].item(Float.self)
        let row1First = out[1..<2, 0..<1, 0..<1].item(Float.self)
        XCTAssertEqual(row0First, 1.0)
        XCTAssertEqual(row1First, 0.0)
    }
}
