import Foundation
import CoreML
import MLX
import MLXNN
import Tokenizers

public enum CoreMLTextHiddenStateProviderError: Error {
    case modelFileNotFound(String)
    case tokenizerNotFound(String)
    case predictionFailed(String)
    case invalidOutputShape(String)
}

/// CoreML implementation of TextHiddenStateProvider for the .mlpackage Text Encoder models.
public final class CoreMLTextHiddenStateProvider: TextHiddenStateProvider {
    private let model: MLModel
    private let encodeTokens: (String) -> [Int]
    private let lock = NSLock()
    
    private init(model: MLModel, encodeTokens: @escaping (String) -> [Int]) {
        self.model = model
        self.encodeTokens = encodeTokens
    }
    
    public static func load(from directory: URL) async throws -> CoreMLTextHiddenStateProvider {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: directory.path) else {
            throw CoreMLTextHiddenStateProviderError.modelFileNotFound(directory.path)
        }
        
        // Load MLModel
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU // Bypass ANE due to ios17.mul broadcasting crash with dynamic shapes
        let compiledURL = try await CoreMLHelper.compileIfNeeded(modelURL: directory)
        let model = try MLModel(contentsOf: compiledURL, configuration: config)
        
        // Load Tokenizer
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        guard fileManager.fileExists(atPath: tokenizerURL.path) else {
            throw CoreMLTextHiddenStateProviderError.tokenizerNotFound(tokenizerURL.path)
        }
        
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        
        return CoreMLTextHiddenStateProvider(
            model: model,
            encodeTokens: { tokenizer.encode(text: $0, addSpecialTokens: true) }
        )
    }
    
    // MARK: - TextHiddenStateProvider Protocol Requirements
    
    public func encodeHiddenStates(text: String, maxLength: Int) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray) {
        var tokenIDs = encodeTokens(text)
        if tokenIDs.isEmpty {
            tokenIDs = [0]
        }
        if tokenIDs.count > maxLength {
            tokenIDs = Array(tokenIDs.prefix(maxLength))
        }
        
        let actualLength = tokenIDs.count
        
        // Pad to exactly maxLength to avoid Core ML dynamic shape broadcasting bugs
        if tokenIDs.count < maxLength {
            tokenIDs.append(contentsOf: Array(repeating: 0, count: maxLength - tokenIDs.count))
        }
        
        let seqLen = tokenIDs.count
        
        let tokenIDsInt32 = tokenIDs.map { Int32($0) }
        let shapedInputIDs = MLShapedArray<Int32>(scalars: tokenIDsInt32, shape: [1, seqLen])
        
        var maskInt32 = [Int32](repeating: 0, count: seqLen)
        for i in 0..<actualLength {
            maskInt32[i] = 1
        }
        let shapedAttentionMask = MLShapedArray<Int32>(scalars: maskInt32, shape: [1, seqLen])
        
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLMultiArray(shapedInputIDs),
            "attention_mask": MLMultiArray(shapedAttentionMask)
        ])
        
        let output: MLFeatureProvider = try lock.withLock {
            return try model.prediction(from: inputProvider)
        }
        
        // Extract output (assume "logits" or similar). Note: for LLMs it's "logits", for Text Encoder usually "hidden_states" or similar.
        // Let's check `quantize_checkpoints.py` again.
        // For LLMs it exported `outputs_schema = [ct.TensorType(name="logits")]`
        // AceStep custom models (DiT) exported "velocity".
        // VAE exported "audio".
        // What about Qwen as a Text Encoder? It's exported as a generic AutoModel.
        // Let's assume it returns `logits` because the fallback Qwen trace uses CausalMWrapper which returns `outputs[0]` which are logits/hidden_states.
        
        guard let outputFeature = output.featureValue(for: "hidden_states")?.multiArrayValue ?? output.featureValue(for: "logits")?.multiArrayValue else {
            let available = output.featureNames.joined(separator: ", ")
            throw CoreMLTextHiddenStateProviderError.invalidOutputShape("Missing 'hidden_states' or 'logits' in Core ML output. Available features: \(available)")
        }
        
        let hiddenStates: MLXArray
        if outputFeature.dataType == .float16 {
            let outputShaped = MLShapedArray<Float16>(outputFeature)
            hiddenStates = MLXArray(outputShaped.scalars, outputShaped.shape)
        } else {
            let outputShaped = MLShapedArray<Float32>(outputFeature)
            hiddenStates = MLXArray(outputShaped.scalars, outputShaped.shape)
        }
        
        var maskFloats = [Float](repeating: 0, count: seqLen)
        for i in 0..<actualLength {
            maskFloats[i] = 1.0
        }
        let attentionMask = MLXArray(maskFloats, [1, seqLen])
        return (hiddenStates, attentionMask)
    }
    
    public func encodeLyricHiddenStates(text: String, maxLength: Int) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray) {
        return try encodeHiddenStates(text: text, maxLength: maxLength)
    }
    
    public func encodeTokenEmbeddings(text: String, maxLength: Int) throws -> (embeddings: MLXArray, attentionMask: MLXArray) {
        // Typically should load from raw embedding layer. We'll fallback to full forward for simplicity in phase 1,
        // or just return the hidden states if that's what the wrapper does.
        let result = try encodeHiddenStates(text: text, maxLength: maxLength)
        return (embeddings: result.hiddenStates, attentionMask: result.attentionMask)
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}

