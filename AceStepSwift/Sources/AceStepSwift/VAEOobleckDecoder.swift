/**
 Oobleck Decoder: Conv1d -> N decoder blocks -> Snake -> Conv1d. Mirrors vae_model.MLXOobleckDecoder.
 Parameter names match Python for load_weights (decoder.conv1, decoder.block.0, ...).
 */

import Foundation
import MLX
import MLXNN

public class VAEOobleckDecoder: Module, UnaryLayer {
    public let conv1: Conv1d
    public let block: [VAEDecoderBlock]
    public let snake1: VAESnake1d
    public let conv2: Conv1d

    public init(
        channels: Int,
        inputChannels: Int,
        audioChannels: Int,
        upsamplingRatios: [Int],
        channelMultiples: [Int]
    ) {
        let cm = [1] + channelMultiples
        let strides = upsamplingRatios
        self.conv1 = Conv1d(
            inputChannels: inputChannels,
            outputChannels: channels * cm[cm.count - 1],
            kernelSize: 7,
            padding: 3
        )
        var blocks: [VAEDecoderBlock] = []
        for i in 0..<strides.count {
            let inputDim = channels * cm[strides.count - i]
            let outputDim = channels * cm[strides.count - i - 1]
            blocks.append(VAEDecoderBlock(inputDim: inputDim, outputDim: outputDim, stride: strides[i]))
        }
        self.block = blocks
        self.snake1 = VAESnake1d(hiddenDim: channels)
        self.conv2 = Conv1d(
            inputChannels: channels,
            outputChannels: audioChannels,
            kernelSize: 7,
            padding: 3,
            bias: false
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
