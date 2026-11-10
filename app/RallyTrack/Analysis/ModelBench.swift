import CoreML
import Foundation

#if DEBUG
/// 真机测速：fp16 和 8 位两个球模型，各跑 200 次，写进 Documents/analysis_log.txt
enum ModelBench {
    static func run() {
        let shape: [NSNumber] = [1, 3, 360, 640]
        func arr() -> MLMultiArray {
            let a = try! MLMultiArray(shape: shape, dataType: .float16)
            let p = a.dataPointer.bindMemory(to: Float16.self, capacity: a.count)
            for i in 0..<a.count { p[i] = Float16(Float(i % 255) / 255) }
            return a
        }
        guard let inp = try? MLDictionaryFeatureProvider(dictionary: ["prev2": arr(), "prev1": arr(), "cur": arr()]) else { return }
        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
        let models: [(String, () throws -> MLModel)] = [("fp16", { try BallTracker(configuration: cfg).model })]   // 比较别的版本时临时加进 ML/ 再测
        for (name, make) in models {
            guard let m = try? make() else { AnalysisQueue.log("bench \(name) load failed"); continue }
            for _ in 0..<20 { _ = try? m.prediction(from: inp) }
            let t = Date()
            for _ in 0..<200 { _ = try? m.prediction(from: inp) }
            AnalysisQueue.log(String(format: "bench %@ %.2f ms/frame", name, Date().timeIntervalSince(t) / 200 * 1000))
        }
    }
}
#endif
