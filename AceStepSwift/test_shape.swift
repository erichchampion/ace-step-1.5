import Foundation
import CoreML

let path = "/Users/erich/Library/Containers/com.cadenza-audio/Data/Library/Application Support/CadenzaAudio/Checkpoints/Qwen3-Embedding-0.6B-8bit.mlpackage"
let url = URL(fileURLWithPath: path)
do {
    let compiledUrl = try MLModel.compileModel(at: url)
    let model = try MLModel(contentsOf: compiledUrl)
    print("Model Outputs:")
    for (name, desc) in model.modelDescription.outputDescriptionsByName {
        if let multiDesc = desc.multiArrayConstraint {
            print("- \(name): type=\(multiDesc.dataType), shape=\(multiDesc.shape)")
        } else {
            print("- \(name): not a multi-array constraint")
        }
    }
} catch {
    print("Error: \(error)")
}
