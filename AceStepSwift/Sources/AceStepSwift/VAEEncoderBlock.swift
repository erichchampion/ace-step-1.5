/**
 Encoder block: 3 residual units (dilations 1, 3, 9) -> Snake -> strided Conv1d down. Mirrors vae_model.MLXOobleckEncoderBlock.
 */

import Foundation
import MLX
import MLXNN

public class VAEEncoderBlock: Module, UnaryLayer {
    public let resUnit1: VAEResidualUnit
    public let resUnit2: VAEResidualUnit
    public let resUnit3: VAEResidualUnit
    public let snake1: VAESnake1d
    public let conv1: Conv1d

    public init(inputDim: Int, outputDim: Int, stride: Int = 1) {
        let pad = (stride + 1) / 2
        self.resUnit1 = VAEResidualUnit(dimension: inputDim, dilation: 1)
        self.resUnit2 = VAEResidualUnit(dimension: inputDim, dilation: 3)
        self.resUnit3 = VAEResidualUnit(dimension: inputDim, dilation: 9)
        self.snake1 = VAESnake1d(hiddenDim: inputDim)
        self.conv1 = Conv1d(
            inputChannels: inputDim,
            outputChannels: outputDim,
            kernelSize: 2 * stride,
            stride: stride,
            padding: pad
        )
        super.init()
    }

    public func callAsFunction(_ hiddenState: MLXArray) -> MLXArray {
        var h = resUnit1(hiddenState)
        h = resUnit2(h)
        h = snake1(resUnit3(h))
        h = conv1(h)
        return h
    }
}
