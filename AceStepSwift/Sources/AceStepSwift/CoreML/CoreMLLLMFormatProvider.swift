import Foundation
import CoreML
import MLX
import MLXNN
import Tokenizers

public enum CoreMLLLMFormatProviderError: Error {
    case modelFileNotFound(String)
    case tokenizerNotFound(String)
    case predictionFailed(String)
    case invalidOutputShape(String)
    case notInitialized
}

/// CoreML implementation of LLMFormatProvider for the .mlpackage LLM models.
/// This uses a manual autoregressive loop with temperature sampling, sending the full sequence
/// each step since KV cache is disabled in the CoreML export.
public final class CoreMLLLMFormatProvider: LLMFormatProvider {
    private var model: MLModel?
    private var tokenizer: Tokenizer?
    private let lock = NSLock()
    
    public var isInitialized: Bool { model != nil && tokenizer != nil }
    
    public init() {}
    
    public func load(from directory: URL) async throws {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: directory.path) else {
            throw CoreMLLLMFormatProviderError.modelFileNotFound(directory.path)
        }
        
        // Load MLModel — bypass ANE to avoid MLIR pass manager crashes and ios17.mul broadcasting bugs with dynamic shapes
        let config = CoreMLConfigFactory.makeConfig(computeUnits: .cpuOnly)
        
        let compiledURL = try await CoreMLHelper.compileIfNeeded(modelURL: directory)
        let loadedModel = try MLModel(contentsOf: compiledURL, configuration: config)
        
        // Load Tokenizer
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        guard fileManager.fileExists(atPath: tokenizerURL.path) else {
            throw CoreMLLLMFormatProviderError.tokenizerNotFound(tokenizerURL.path)
        }
        
        let loadedTokenizer = try await AutoTokenizer.from(modelFolder: directory)
        
        lock.withLock {
            self.model = loadedModel
            self.tokenizer = loadedTokenizer
        }
        
        print("[CoreMLLLMFormatProvider] Successfully loaded Core ML LM from \(directory.path)")
    }
    
    public func load(id: String) async throws {
        // Fallback for Hugging Face downloads if implemented, but we typically use `load(from:)` with a local path
        fatalError("load(id:) not implemented for CoreMLLLMFormatProvider. Use load(from:) with the .mlpackage URL.")
    }
    
    public func unload() {
        lock.withLock {
            self.model = nil
            self.tokenizer = nil
        }
    }
    
    // MARK: - Prompt Building
    
    private func buildInspirationPrompt(query: String, instrumental: Bool) -> String {
        let q = query.trimmingCharacters(in: .whitespaces).isEmpty ? "NO USER INPUT" : query
        let inst = instrumental ? "true" : "false"
        
        // Qwen chat format
        return """
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        # Instruction
        Expand the user's input into a more detailed and specific musical description.
        Output exactly in this format inside a <think> block:
        <think>
        bpm: [30-300]
        caption: [one sentence description ending with a period]
        duration: [10-600]
        keyscale: [e.g. C major or Am]
        language: [e.g. en, zh, ja]
        timesignature: [2, 3, 4, or 6]
        </think>

        \(q)

        instrumental: \(inst)<|im_end|>
        <|im_start|>assistant
        """
    }

    private func buildFormatPrompt(caption: String, lyrics: String) -> String {
        return """
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Format the following music caption and lyrics into structured metadata.
        Output exactly in this format inside a <think> block:
        <think>
        bpm: [30-300]
        caption: [one sentence description ending with a period]
        duration: [10-600]
        keyscale: [e.g. C major or Am]
        language: [e.g. en, zh, ja]
        timesignature: [2, 3, 4, or 6]
        </think>

        Caption: \(caption)
        Lyrics:
        \(lyrics)<|im_end|>
        <|im_start|>assistant
        """
    }
    
    // MARK: - Generation
    
    public func generateFormat(caption: String, lyrics: String, userMetadata: [String: Any]?, temperature: Double) async throws -> String {
        guard isInitialized else { throw CoreMLLLMFormatProviderError.notInitialized }
        let prompt = buildFormatPrompt(caption: caption, lyrics: lyrics)
        return try await generateAutoregressively(prompt: prompt, maxTokens: 512, temperature: Float(temperature))
    }
    
    public func generateFromQuery(query: String, instrumental: Bool, vocalLanguage: String?, temperature: Double) async throws -> String {
        guard isInitialized else { throw CoreMLLLMFormatProviderError.notInitialized }
        let prompt = buildInspirationPrompt(query: query, instrumental: instrumental)
        return try await generateAutoregressively(prompt: prompt, maxTokens: 512, temperature: Float(temperature))
    }
    
    private func generateAutoregressively(prompt: String, maxTokens: Int, temperature: Float) async throws -> String {
        guard let model = model, let tokenizer = tokenizer else { throw CoreMLLLMFormatProviderError.notInitialized }
        
        var tokenIDs = tokenizer.encode(text: prompt, addSpecialTokens: false)
        
        // Qwen special tokens
        let eosTokenID: Int? = 151645 // <|im_end|>
        _ = 151936 // Qwen2.5/3 vocab
        
        for _ in 0..<maxTokens {
            let seqLen = tokenIDs.count
            let tokenIDsInt32 = tokenIDs.map { Int32($0) }
            
            let shapedInputIDs = MLShapedArray<Int32>(scalars: tokenIDsInt32, shape: [1, seqLen])
            let shapedAttentionMask = MLShapedArray<Int32>(repeating: 1, shape: [1, seqLen])
            
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLMultiArray(shapedInputIDs),
                "attention_mask": MLMultiArray(shapedAttentionMask)
            ])
            
            // Forward pass
            let output = try lock.withLock {
                try model.prediction(from: inputProvider)
            }
            
            guard let logitsFeature = output.featureValue(for: "logits")?.multiArrayValue else {
                throw CoreMLLLMFormatProviderError.invalidOutputShape("Missing 'logits'")
            }
            
            let logitsShaped = MLShapedArray<Float32>(logitsFeature)
            // logits should be shape [1, seqLen, vocabSize]
            let shape = logitsShaped.shape
            guard shape.count == 3, shape[0] == 1, shape[1] == seqLen else {
                throw CoreMLLLMFormatProviderError.invalidOutputShape("Logits shape mismatch: \(shape)")
            }
            
            let actualVocabSize = shape[2]
            
            // Extract the last token's logits
            let scalars = logitsShaped.scalars
            let lastTokenLogitsStart = (seqLen - 1) * actualVocabSize
            let lastTokenLogits = Array(scalars[lastTokenLogitsStart..<lastTokenLogitsStart + actualVocabSize])
            
            // Sample
            let nextTokenID = sample(logits: lastTokenLogits, temperature: temperature)
            tokenIDs.append(nextTokenID)
            
            if nextTokenID == eosTokenID {
                break
            }
        }
        
        // Decode
        return tokenizer.decode(tokens: tokenIDs)
    }
    
    private func sample(logits: [Float32], temperature: Float) -> Int {
        if temperature <= 0.0 {
            // Argmax
            var maxVal = Float32.leastNormalMagnitude
            var maxIdx = 0
            for (i, val) in logits.enumerated() {
                if val > maxVal {
                    maxVal = val
                    maxIdx = i
                }
            }
            return maxIdx
        }
        
        // Gumbel-max trick or naive softmax
        // Compute softmax with temperature
        var maxLogit = Float32.leastNormalMagnitude
        for val in logits {
            if val > maxLogit { maxLogit = val }
        }
        
        var sum: Float32 = 0
        var expLogits = [Float32](repeating: 0, count: logits.count)
        for (i, val) in logits.enumerated() {
            let e = exp((val - maxLogit) / temperature)
            expLogits[i] = e
            sum += e
        }
        
        // Random sample
        let randomVal = Float32.random(in: 0..<sum)
        var cumulative: Float32 = 0
        for (i, e) in expLogits.enumerated() {
            cumulative += e
            if randomVal <= cumulative {
                return i
            }
        }
        return logits.count - 1
    }
}

private extension NSLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
