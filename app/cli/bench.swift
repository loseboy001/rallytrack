import CoreML
import Foundation
// 比较几种调用方式的单帧耗时
let dir = URL(fileURLWithPath: CommandLine.arguments[1])
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
let m = try MLModel(contentsOf: dir.appendingPathComponent("BallTracker.mlmodelc"), configuration: cfg)
let shape: [NSNumber] = [1, 3, 360, 640]
func arr() -> MLMultiArray { let a = try! MLMultiArray(shape: shape, dataType: .float16); let p = a.dataPointer.bindMemory(to: Float16.self, capacity: a.count); for i in 0..<a.count { p[i] = Float16(Float(i % 255) / 255) }; return a }
let a0 = arr(), a1 = arr(), a2 = arr()
let inp = try MLDictionaryFeatureProvider(dictionary: ["prev2": a0, "prev1": a1, "cur": a2])
for _ in 0..<10 { _ = try m.prediction(from: inp) }
var t = Date(); for _ in 0..<200 { _ = try m.prediction(from: inp) }
print(String(format: "同步 prediction: %.2f ms", Date().timeIntervalSince(t) / 200 * 1000))
let sem = DispatchSemaphore(value: 0)
Task {
    t = Date(); for _ in 0..<200 { _ = try await m.prediction(from: inp) }
    print(String(format: "异步 await prediction: %.2f ms", Date().timeIntervalSince(t) / 200 * 1000))
    // 每次新建输入数组（App 里是这样）
    t = Date(); for _ in 0..<200 { let p = try MLDictionaryFeatureProvider(dictionary: ["prev2": arr(), "prev1": a1, "cur": a2]); _ = try await m.prediction(from: p) }
    print(String(format: "每帧新建一个输入数组: %.2f ms", Date().timeIntervalSince(t) / 200 * 1000))
    // 批量
    let opts = MLPredictionOptions()
    let batch = MLArrayBatchProvider(array: Array(repeating: inp, count: 8))
    t = Date(); for _ in 0..<25 { _ = try m.predictions(from: batch, options: opts) }
    print(String(format: "批量 8 帧一起: %.2f ms/帧", Date().timeIntervalSince(t) / 200 * 1000))
    sem.signal()
}
sem.wait()
// IOSurface 后备的 fp16 数组：神经网络引擎可以直接读，不用拷贝
import CoreVideo
func surfaceArr() -> MLMultiArray {
    var pb: CVPixelBuffer?
    let attrs = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary
    CVPixelBufferCreate(nil, 640, 360 * 3, kCVPixelFormatType_OneComponent16Half, attrs, &pb)
    return MLMultiArray(pixelBuffer: pb!, shape: [1, 3, 360, 640])
}
let s0 = surfaceArr(), s1 = surfaceArr(), s2 = surfaceArr()
let sinp = try MLDictionaryFeatureProvider(dictionary: ["prev2": s0, "prev1": s1, "cur": s2])
for _ in 0..<10 { _ = try m.prediction(from: sinp) }
t = Date(); for _ in 0..<200 { _ = try m.prediction(from: sinp) }
print(String(format: "IOSurface 输入: %.2f ms", Date().timeIntervalSince(t) / 200 * 1000))
// 第二个参数：8 位模型的 .mlpackage
let mq = try MLModel(contentsOf: try MLModel.compileModel(at: URL(fileURLWithPath: CommandLine.arguments[2])), configuration: cfg)
for _ in 0..<10 { _ = try mq.prediction(from: sinp) }
t = Date(); for _ in 0..<200 { _ = try mq.prediction(from: sinp) }
print(String(format: "IOSurface 输入 + 8 位模型: %.2f ms", Date().timeIntervalSince(t) / 200 * 1000))
