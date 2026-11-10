import CoreML

extension VideoAnalyzer {
    /// App 内使用：加载打包进 App 的三个模型（Xcode 自动生成的模型类）
    convenience init() throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine  // 后台不能用 GPU，统一走神经网络引擎
        self.init(ball: try BallTracker(configuration: cfg).model,
                  seg: try TableSeg(configuration: cfg).model,
                  event: try EventNet(configuration: cfg).model)
    }
}
