/**
 Tests for FSM states and transition logic (mirrors Python constrained_logits_processor).
 */

import XCTest
@testable import AceStepSwift

final class FSMStateTests: XCTestCase {

    func testFSMStateHasThinkTagAndCompleted() {
        let start: FSMState = .thinkTag
        let end: FSMState = .completed
        XCTAssertNotEqual(start, end)
    }

    func testFSMStateCasesAreDistinct() {
        let cases: [FSMState] = [.thinkTag, .bpmName, .keyscaleValue, .thinkEndTag, .codesGeneration, .completed]
        let unique = Set(cases.map { "\($0)" })
        XCTAssertEqual(unique.count, cases.count)
    }

    // MARK: - Transition tests

    func testNextStateThinkTagToNewlineAfterThink() {
        let next = nextState(after: .thinkTag)
        XCTAssertEqual(next, .newlineAfterThink)
    }

    func testNextStateBpmValueToCaptionNameWhenCaptionNotSkipped() {
        let opts = FSMTransitionOptions(skipCaption: false, skipGenres: true, skipLanguage: false)
        let next = nextState(after: .bpmValue, options: opts)
        XCTAssertEqual(next, .captionName)
    }

    func testNextStateBpmValueToDurationNameWhenCaptionSkipped() {
        let opts = FSMTransitionOptions(skipCaption: true, skipGenres: true, skipLanguage: false)
        let next = nextState(after: .bpmValue, options: opts)
        XCTAssertEqual(next, .durationName)
    }

    func testNextStateDurationValueToKeyscaleNameWhenGenresSkipped() {
        let opts = FSMTransitionOptions(skipCaption: false, skipGenres: true, skipLanguage: false)
        let next = nextState(after: .durationValue, options: opts)
        XCTAssertEqual(next, .keyscaleName)
    }

    func testNextStateTimesigValueToThinkEndTag() {
        let next = nextState(after: .timesigValue)
        XCTAssertEqual(next, .thinkEndTag)
    }

    func testNextStateThinkEndTagToCodesGeneration() {
        let next = nextState(after: .thinkEndTag)
        XCTAssertEqual(next, .codesGeneration)
    }

    func testNextStateCompletedReturnsNil() {
        let next = nextState(after: .completed)
        XCTAssertNil(next)
    }

    func testChainFromThinkTagToCompletedWithDefaultOptions() {
        var state: FSMState? = .thinkTag
        let steps: [FSMState] = [.newlineAfterThink, .bpmName, .bpmValue, .captionName, .captionValue,
                                  .durationName, .durationValue, .keyscaleName, .keyscaleValue,
                                  .languageName, .languageValue, .timesigName, .timesigValue,
                                  .thinkEndTag, .codesGeneration, .completed]
        for expected in steps {
            state = nextState(after: state!)
            XCTAssertEqual(state, expected, "chain step")
        }
        state = nextState(after: state!)
        XCTAssertNil(state)
    }
}
