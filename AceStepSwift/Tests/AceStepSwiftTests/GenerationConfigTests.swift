import XCTest
@testable import AceStepSwift

final class GenerationConfigTests: XCTestCase {

    func testDefaultBatchSizeIs2() {
        let config = GenerationConfig()
        XCTAssertEqual(config.batchSize, 2)
    }

    func testDefaultUseRandomSeedIsTrue() {
        let config = GenerationConfig()
        XCTAssertTrue(config.useRandomSeed)
    }

    func testDefaultSeedsIsNil() {
        let config = GenerationConfig()
        XCTAssertNil(config.seeds)
    }

    func testDefaultAudioFormatIsFlac() {
        let config = GenerationConfig()
        XCTAssertEqual(config.audioFormat, "flac")
    }

    func testConfigWithExplicitSeeds() {
        let config = GenerationConfig(seeds: [42, 43])
        XCTAssertEqual(config.seeds, [42, 43])
    }
}
