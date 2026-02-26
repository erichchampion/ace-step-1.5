/**
 Oobleck Encoder: Conv1d -> N encoder blocks -> Snake -> Conv1d. Mirrors vae_model.MLXOobleckEncoder.
 Parameter names match Python for load_weights (encoder.conv1, encoder.block.0, ...).
 */

import Foundation
import MLX
import MLXNN

public class VAEOobleckEncoder: Module, UnaryLayer {
    public let conv1: Conv1d
    public let block: [VAEEncoderBlock]
    public let snake1: VAESnake1d
    public let conv2: Conv1d

    public init(
        encoderHiddenSize: Int,
        audioChannels: Int,
        downsamplingRatios: [Int],
        channelMultiples: [Int]
    ) {
        let cm = [1] + channelMultiples
        let strides = downsamplingRatios
        self.conv1 = Conv1d(
            inputChannels: audioChannels,
            outputChannels: encoderHiddenSize,
            kernelSize: 7,
            padding: 3
        )
        var blocks: [VAEEncoderBlock] = []
        for i in 0..<strides.count {
            let inputDim = encoderHiddenSize * cm[i]
            let outputDim = encoderHiddenSize * cm[i + 1]
            blocks.append(VAEEncoderBlock(inputDim: inputDim, outputDim: outputDim, stride: strides[i]))
        }
        self.block = blocks
        let dModel = encoderHiddenSize * cm[cm.count - 1]
        self.snake1 = VAESnake1d(hiddenDim: dModel)
        self.conv2 = Conv1d(
            inputChannels: dModel,
            outputChannels: encoderHiddenSize,
            kernelSize: 3,
            padding: 1
        )
        super.init()
    }

    public func callAsFunction(_ hiddenState: MLXArray) -> MLXArray {
        var h = conv1(hiddenState)
        for layer in block {
            h = layer(h)
        }
        h = snake1(h)
        h = conv2(h)
        return h
    }
}
