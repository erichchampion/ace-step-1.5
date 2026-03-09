import Foundation
import CoreML

/// Helper for compiling and bridging Core ML `.mlpackage` bundles.
public enum CoreMLHelper {
    public static func compileIfNeeded(modelURL: URL) async throws -> URL {
        let fileManager = FileManager.default
        let isPackage = modelURL.pathExtension == "mlpackage" || modelURL.pathExtension == "mlmodel"
        
        guard isPackage else {
            // Assume it's already an .mlmodelc or similar and try to load it directly
            return modelURL
        }
        
        let compiledName = modelURL.lastPathComponent + "c"
        let compiledURL = modelURL.deletingLastPathComponent().appendingPathComponent(compiledName)
        
        if fileManager.fileExists(atPath: compiledURL.path) {
            return compiledURL
        }
        
        // Compile to a temporary URL off the main actor
        print("[CoreMLHelper] Compiling model at \(modelURL.path)... This may take a moment.")
        let tempURL = try await Task.detached(priority: .userInitiated) {
            return try MLModel.compileModel(at: modelURL)
        }.value
        
        // Move to permanent cached location
        if !fileManager.fileExists(atPath: compiledURL.path) {
            do {
                try fileManager.moveItem(at: tempURL, to: compiledURL)
                print("[CoreMLHelper] Cached compiled model to \(compiledURL.path)")
            } catch {
                // If another thread already moved it, ignore the error
                if !fileManager.fileExists(atPath: compiledURL.path) {
                    throw error
                }
            }
        }
        
        return compiledURL
    }
}
