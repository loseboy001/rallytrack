import CoreML
import Foundation

// 命令行验证：rtcli <models_dir> <video> <out.json>
// 用与 App 相同的分析代码和 CoreML 模型（M1 原生，可用神经网络引擎），输出结果、逐帧轨迹和落台概率
let args = CommandLine.arguments
let dir = URL(fileURLWithPath: args[1]), video = URL(fileURLWithPath: args[2]), out = URL(fileURLWithPath: args[3])
let cfg = MLModelConfiguration(); cfg.computeUnits = ProcessInfo.processInfo.environment["ANE"] != nil ? .cpuAndNeuralEngine : .all
func load(_ n: String) throws -> MLModel { try MLModel(contentsOf: dir.appendingPathComponent(n + ".mlmodelc"), configuration: cfg) }

let sem = DispatchSemaphore(value: 0)
Task {
    do {
        let a = VideoAnalyzer(ball: try load("BallTracker"), seg: try load("TableSeg"), event: try load("EventNet"))
        a.skipStride = Int(ProcessInfo.processInfo.environment["STRIDE"] ?? "4") ?? 4
        if let t = ProcessInfo.processInfo.environment["MASKT"], let v = Float(t) { a.maskThreshold = v }
        let t0 = Date()
        // QUAD="x1,y1,x2,y2,x3,y3,x4,y4" 手动指定球台四角（远左、远右、近右、近左），模拟 App 里手动拖角
        var tf: TableFit
        if let q = ProcessInfo.processInfo.environment["QUAD"]?.split(separator: ",").compactMap({ Double($0) }), q.count == 8 {
            tf = TableFit(quad: stride(from: 0, to: 8, by: 2).map { CGPoint(x: q[$0], y: q[$0 + 1]) }, k1: 0)
        } else {
            tf = try await a.detectTable(url: video)
        }
        if let k = ProcessInfo.processInfo.environment["K1"], let v = Double(k) { tf.k1 = v }
        let lens = tf.lens(try await VideoAnalyzer.info(of: video).size)
        let quad = tf.quad, qImg = quad.map(lens.distort)
        if ProcessInfo.processInfo.environment["TABLE_ONLY"] != nil {  // 只测球台识别
            print("quad", quad.map { "\(Int($0.x)),\(Int($0.y))" }.joined(separator: " "), "k1", tf.k1); exit(0)
        }
        let t1 = Date()
        let (track, info) = try await a.track(url: video, quad: ProcessInfo.processInfo.environment["NOROI"] == nil ? qImg : nil) { _ in }
        let t2 = Date()
        let prob = try a.bounceProbability(track, quad: qImg)
        let r = a.summarize(raw: track, prob: prob, quad: quad, lens: lens, info: info)
        let t3 = Date()
        print(String(format: "table %.1fs | track %d frames in %.1fs = %.0f fps | events %.2fs", t1.timeIntervalSince(t0), track.count,
                     t2.timeIntervalSince(t1), Double(track.count) / t2.timeIntervalSince(t1), t3.timeIntervalSince(t2)))
        print(String(format: "stride %d, inferred %.1f%% of frames", a.skipStride, a.lastInferenceRatio * 100))
        print("quad", quad.map { "(\(Int($0.x)),\(Int($0.y)))" }.joined(separator: " "), "k1", tf.k1)
        print("rallies", r.rallies.count, "shots", r.rallies.map(\.shots), "total", r.totalShots, "longest", r.longestRally,
              "detected", String(format: "%.3f", r.detectedRatio))
        let dump: [String: Any] = ["quad": quad.map { [$0.x, $0.y] }, "k1": tf.k1, "x": track.x.map { $0.isNaN ? -1 : $0 }, "y": track.y.map { $0.isNaN ? -1 : $0 },
                                   "conf": track.conf, "prob": prob.map(Double.init),
                                   "rallies": r.rallies.map { ["start": $0.startFrame, "end": $0.endFrame, "speeds": $0.bounces.map { $0.speedKmh ?? -1 }, "bf": $0.bounces.map(\.frame), "tx": $0.bounces.map(\.tableX), "ty": $0.bounces.map(\.tableY), "out": $0.bounces.map { $0.out == true }, "crossings": $0.crossings ?? [], "bounces": $0.bounces.map(\.frame)] }]
        try JSONSerialization.data(withJSONObject: dump).write(to: out)
    } catch { print("ERROR", error) }
    sem.signal()
}
sem.wait()
