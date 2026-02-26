/**
 ConvTranspose1d module for NLC input. MLXNN does not provide it; we wrap MLX.convTransposed1d.
 Weight shape: [outputChannels, kernelSize, inputChannels] (MLX convention).
 */

import Foundation
import MLX
import MLXNN

open class ConvTranspose1d: Module, UnaryLayer {
    public let weight: MLXArray
    public let bias: MLXArray?
    public let stride: Int
    public let padding: Int
    public let outputPadding: Int

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        outputPadding: Int = 0,
        bias: Bool = true
    ) {
        let scale = sqrt(1 / Float(inputChannels * kernelSize))
        self.weight = MLXRandom.uniform(
            low: -scale, high: scale,
            [outputChannels, kernelSize, inputChannels]
        )
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        self.stride = stride
        self.padding = padding
        self.outputPadding = outputPadding
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = convTransposed1d(x, weight, stride: stride, padding: padding, outputPadding: outputPadding)
        if let b = bias {
            y = y + b
        }
        return y
    }
}
