import Foundation
import CoreML

let path = "/Users/erich/Library/Containers/com.cadenza-audio/Data/Library/Application Support/CadenzaAudio/Checkpoints/Qwen3-Embedding-0.6B-8bit.mlpackage"
let url = URL(fileURLWithPath: path)
do {
    let compiledUrl = try MLModel.compileModel(at: url)
    let model = try MLModel(contentsOf: compiledUrl)
    print("Model Inputs:")
    for (name, desc) in model.modelDescription.inputDescriptionsByName {
        print("- \(name): \(desc)")
    }
    print("Model Outputs:")
    for (name, desc) in model.modelDescription.outputDescriptionsByName {
        print("- \(name): \(desc)")
    }
} catch {
    print("Error: \(error)")
}
