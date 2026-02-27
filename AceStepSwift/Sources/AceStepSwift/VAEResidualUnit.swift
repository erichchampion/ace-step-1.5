/**
 Two weight-normalized Conv1d layers (k=7 dilated + k=1) with Snake1d and residual. Mirrors vae_model.MLXOobleckResidualUnit.
 */

import Foundation
import MLX
import MLXNN

public class VAEResidualUnit: Module, UnaryLayer {
    public let snake1: VAESnake1d
    public let conv1: Conv1d
    public let snake2: VAESnake1d
    public let conv2: Conv1d

    public init(dimension: Int, dilation: Int = 1) {
        let pad = ((7 - 1) * dilation) / 2
        self.snake1 = VAESnake1d(hiddenDim: dimension)
        self.conv1 = Conv1d(
            inputChannels: dimension,
            outputChannels: dimension,
            kernelSize: 7,
            padding: pad,
            dilation: dilation
        )
        self.snake2 = VAESnake1d(hiddenDim: dimension)
        self.conv2 = Conv1d(
            inputChannels: dimension,
            outputChannels: dimension,
            kernelSize: 1
        )
        super.init()
    }

    public func callAsFunction(_ hiddenState: MLXArray) -> MLXArray {
        var output = conv1(snake1(hiddenState))
        output = conv2(snake2(output))
        let pad = (hiddenState.dim(1) - output.dim(1)) / 2
        let L = hiddenState.dim(1)
        // Single .ellipsis only in MLX; use explicit ranges for [B, L, C] slice on axis 1.
        let trimmed = pad > 0 ? hiddenState[0..<hiddenState.dim(0), pad ..< (L - pad), 0..<hiddenState.dim(2)] : hiddenState
        return trimmed + output
    }
}
