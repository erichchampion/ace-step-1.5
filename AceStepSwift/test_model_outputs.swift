import Foundation
import CoreML

let path = "/Users/erich/Library/Containers/com.cadenza-audio/Data/Library/Application Support/CadenzaAudio/Checkpoints/acestep-5Hz-lm-0.6B-8bit.mlpackage"
let url = URL(fileURLWithPath: path)
do {
    let model = try MLModel(contentsOf: try MLModel.compileModel(at: url))
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
