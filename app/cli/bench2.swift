import CoreML
import Foundation
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let shape: [NSNumber] = [1, 3, 360, 640]
func arr() -> MLMultiArray { let a = try! MLMultiArray(shape: shape, dataType: .float16); let p = a.dataPointer.bindMemory(to: Float16.self, capacity: a.count); for i in 0..<a.count { p[i] = Float16(Float(i % 255) / 255) }; return a }
let inp = try MLDictionaryFeatureProvider(dictionary: ["prev2": arr(), "prev1": arr(), "cur": arr()])
for (name, cu) in [("cpuOnly", MLComputeUnits.cpuOnly), ("cpuAndGPU", .cpuAndGPU), ("cpuAndNE", .cpuAndNeuralEngine), ("all", .all)] {
    let cfg = MLModelConfiguration(); cfg.computeUnits = cu
    let m = try MLModel(contentsOf: url, configuration: cfg)
    for _ in 0..<10 { _ = try m.prediction(from: inp) }
    let t = Date(); for _ in 0..<100 { _ = try m.prediction(from: inp) }
    print(name, String(format: "%.2f ms", Date().timeIntervalSince(t) / 100 * 1000))
}
