import XCTest
@testable import AceStepSwift

final class TaskTypeTests: XCTestCase {

    func testTaskTypeFromStringText2Music() {
        XCTAssertEqual(TaskType(string: "text2music"), .text2music)
        XCTAssertEqual(TaskType(string: "TEXT2MUSIC"), .text2music)
    }

    func testTaskTypeFromStringRepaintCoverLego() {
        XCTAssertEqual(TaskType(string: "repaint"), .repaint)
        XCTAssertEqual(TaskType(string: "cover"), .cover)
        XCTAssertEqual(TaskType(string: "lego"), .lego)
        XCTAssertEqual(TaskType(string: "extract"), .extract)
        XCTAssertEqual(TaskType(string: "complete"), .complete)
    }

    func testTaskTypeFromStringInvalidReturnsNil() {
        XCTAssertNil(TaskType(string: "invalid"))
        XCTAssertNil(TaskType(string: ""))
    }

    func testGenerateInstructionText2Music() {
        let s = TaskInstructions.instruction(for: .text2music, trackName: nil, completeTrackClasses: nil)
        XCTAssertEqual(s, "Fill the audio semantic mask based on the given conditions:")
    }

    func testGenerateInstructionRepaintCover() {
        XCTAssertEqual(
            TaskInstructions.instruction(for: .repaint, trackName: nil, completeTrackClasses: nil),
            "Repaint the mask area based on the given conditions:"
        )
        XCTAssertEqual(
            TaskInstructions.instruction(for: .cover, trackName: nil, completeTrackClasses: nil),
            "Generate audio semantic tokens based on the given conditions:"
        )
    }

    func testGenerateInstructionLegoWithTrackName() {
        XCTAssertEqual(
            TaskInstructions.instruction(for: .lego, trackName: "vocals", completeTrackClasses: nil),
            "Generate the VOCALS track based on the audio context:"
        )
        XCTAssertEqual(
            TaskInstructions.instruction(for: .lego, trackName: nil, completeTrackClasses: nil),
            "Generate the track based on the audio context:"
        )
    }

    func testGenerateInstructionExtractWithTrackName() {
        XCTAssertEqual(
            TaskInstructions.instruction(for: .extract, trackName: "drums", completeTrackClasses: nil),
            "Extract the DRUMS track from the audio:"
        )
    }

    func testGenerateInstructionCompleteWithTrackClasses() {
        XCTAssertEqual(
            TaskInstructions.instruction(for: .complete, trackName: nil, completeTrackClasses: ["vocals", "drums"]),
            "Complete the input track with VOCALS | DRUMS:"
        )
        XCTAssertEqual(
            TaskInstructions.instruction(for: .complete, trackName: nil, completeTrackClasses: nil),
            "Complete the input track:"
        )
    }

    func testInstructionForParamsUsesTaskTypeAndLegoTrack() {
        var params = GenerationParams()
        params.taskType = "lego"
        params.legoTrack = "bass"
        XCTAssertEqual(
            TaskInstructions.instruction(for: params),
            "Generate the BASS track based on the audio context:"
        )
        params.taskType = "text2music"
        params.legoTrack = nil
        XCTAssertEqual(
            TaskInstructions.instruction(for: params),
            "Fill the audio semantic mask based on the given conditions:"
        )
    }

    func testInstructionForParamsCompleteWithTrackClasses() {
        var params = GenerationParams()
        params.taskType = "complete"
        params.completeTrackClasses = ["vocals", "drums"]
        XCTAssertEqual(
            TaskInstructions.instruction(for: params),
            "Complete the input track with VOCALS | DRUMS:"
        )
        params.completeTrackClasses = nil
        XCTAssertEqual(TaskInstructions.instruction(for: params), "Complete the input track:")
    }

    func testInstructionForParamsExtractWithTrackName() {
        var params = GenerationParams()
        params.taskType = "extract"
        params.legoTrack = "bass"
        XCTAssertEqual(
            TaskInstructions.instruction(for: params),
            "Extract the BASS track from the audio:"
        )
    }
}
