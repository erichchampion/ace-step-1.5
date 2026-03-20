import Foundation
import CoreML

let path = "/Users/erich/Library/Containers/com.cadenza-audio/Data/Library/Application Support/CadenzaAudio/Checkpoints/Qwen3-Embedding-0.6B-8bit.mlpackagec"
let url = URL(fileURLWithPath: path)
do {
    let model = try MLModel(contentsOf: url)
    print("Model Outputs:")
    for (name, desc) in model.modelDescription.outputDescriptionsByName {
        if let multiDesc = desc.multiArrayConstraint {
            print("- \(name): type=\(multiDesc.dataType), shape=\(multiDesc.shape)")
        } else {
            print("- \(name): \(desc)")
        }
    }
} catch {
    print("Error: \(error)")
}
