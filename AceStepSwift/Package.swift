// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AceStepSwift",
    platforms: [
        .iOS(.v17),   // Minimum iOS 17 (MLX requires 16+). App target must also set deployment target ≥ 17.
        .macOS(.v14),
    ],
    products: [
        .library(name: "AceStepSwift", targets: ["AceStepSwift"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "2.29.1")),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.1.6")),
    ],
    targets: [
        .target(
            name: "AceStepSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/AceStepSwift"
        ),
        .testTarget(
            name: "AceStepSwiftTests",
            dependencies: [
                "AceStepSwift",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tests/AceStepSwiftTests"
        ),
    ]
)
