/**
 Decoder block: Snake -> ConvTranspose1d up -> 3 residual units. Mirrors vae_model.MLXOobleckDecoderBlock.
 */

import Foundation
import MLX
import MLXNN

public class VAEDecoderBlock: Module, UnaryLayer {
    public let snake1: VAESnake1d
    public let conv_t1: ConvTranspose1d
    public let res_unit1: VAEResidualUnit
    public let res_unit2: VAEResidualUnit
    public let res_unit3: VAEResidualUnit

    public init(inputDim: Int, outputDim: Int, stride: Int = 1) {
        let pad = (stride + 1) / 2
        self.snake1 = VAESnake1d(hiddenDim: inputDim)
        self.conv_t1 = ConvTranspose1d(
            inputChannels: inputDim,
            outputChannels: outputDim,
            kernelSize: 2 * stride,
            stride: stride,
            padding: pad,
            outputPadding: 0
        )
        self.res_unit1 = VAEResidualUnit(dimension: outputDim, dilation: 1)
        self.res_unit2 = VAEResidualUnit(dimension: outputDim, dilation: 3)
        self.res_unit3 = VAEResidualUnit(dimension: outputDim, dilation: 9)
        super.init()
    }

    public func callAsFunction(_ hiddenState: MLXArray) -> MLXArray {
        var h = snake1(hiddenState)
        h = conv_t1(h)
        h = res_unit1(h)
        h = res_unit2(h)
        h = res_unit3(h)
        return h
    }
}
