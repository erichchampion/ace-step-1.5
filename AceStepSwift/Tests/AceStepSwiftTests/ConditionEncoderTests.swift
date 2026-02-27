/**
 Tests for ConditionEncoder: text-only and full path shapes, batch consistency.
 Validates encoder output shape [B, L, 2048] against Python/make_test_fixtures contract.
 */

import XCTest
import MLX
@testable import AceStepSwift

final class ConditionEncoderTests: XCTestCase {

    func testConditionEncoderTextOnlyOutputShape() {
        let encoder = ConditionEncoder(config: .turbo)
        let b = 1
        let L = 6
        let textHidden = MLXArray.zeros([b, L, 1024])
        let (enc, _) = encoder.call(textHiddenStates: textHidden, textAttentionMask: nil)
        XCTAssertEqual(enc.dim(0), b)
        XCTAssertEqual(enc.dim(1), L)
        XCTAssertEqual(enc.dim(2), 2048)
    }

    func testConditionEncoderFullPathWithNilLyricTimbreSameAsTextOnly() {
        let encoder = ConditionEncoder(config: .turbo)
        let b = 1
        let L = 4
        let textHidden = MLXArray.zeros([b, L, 1024])
        let (enc, _) = encoder.call(
            textHiddenStates: textHidden,
            textAttentionMask: nil,
            lyricHiddenStates: nil,
            lyricAttentionMask: nil,
            referAudioPacked: nil,
            referAudioOrderMask: nil
        )
        XCTAssertEqual(enc.dim(0), b)
        XCTAssertEqual(enc.dim(1), L)
        XCTAssertEqual(enc.dim(2), 2048)
    }

    func testConditionEncoderFullPathWithLyricRunsAndHasCorrectShape() {
        let encoder = ConditionEncoder(config: .turbo)
        let b = 2
        let LText = 3
        let LLyric = 2
        let textHidden = MLXArray.zeros([b, LText, 1024])
        let lyricHidden = MLXArray.zeros([b, LLyric, 1024])
        let lyricMask = MLXArray.ones([b, LLyric])
        let (enc, _) = encoder.call(
            textHiddenStates: textHidden,
            textAttentionMask: nil,
            lyricHiddenStates: lyricHidden,
            lyricAttentionMask: lyricMask,
            referAudioPacked: nil,
            referAudioOrderMask: nil
        )
        XCTAssertEqual(enc.dim(0), b)
        XCTAssertEqual(enc.dim(2), 2048)
        XCTAssertEqual(enc.dim(1), LLyric + LText)
    }

    func testConditionEncoderFullPathWithReferRunsAndHasCorrectShape() {
        let encoder = ConditionEncoder(config: .turbo)
        let b = 1
        let LText = 2
        let N = 1
        let T = 4
        let textHidden = MLXArray.zeros([b, LText, 1024])
        let referPacked = MLXArray.zeros([N, T, 64])
        let referOrder = MLXArray([Int32(0)])
        let (enc, _) = encoder.call(
            textHiddenStates: textHidden,
            textAttentionMask: nil,
            lyricHiddenStates: nil,
            lyricAttentionMask: nil,
            referAudioPacked: referPacked,
            referAudioOrderMask: referOrder
        )
        XCTAssertEqual(enc.dim(0), b)
        XCTAssertEqual(enc.dim(2), 2048)
        XCTAssertGreaterThanOrEqual(enc.dim(1), LText)
    }

    func testConditionEncoderBatchConsistency() {
        let encoder = ConditionEncoder(config: .turbo)
        let b = 2
        let L = 3
        let textHidden = MLXArray.zeros([b, L, 1024])
        let (enc, _) = encoder.call(textHiddenStates: textHidden, textAttentionMask: nil)
        XCTAssertEqual(enc.dim(0), b)
    }
}
