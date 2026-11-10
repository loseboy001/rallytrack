import Accelerate
import AVFoundation
import CoreGraphics
import CoreML
import Foundation

// MARK: - 结果

struct BounceInfo: Codable, Hashable {
    var frame: Int
    var tableX: Double      // 沿球台长度方向，米（0…2.74）
    var tableY: Double      // 沿球台宽度方向，米（0…1.525）
    var speedKmh: Double?   // 落台前的估算过台速度
    var out: Bool?          // true = 这一板没落到台上（出界/下网），位置是轨迹最低点
}

struct RallyInfo: Codable, Identifiable, Hashable {
    var id: Int
    var startFrame: Int
    var endFrame: Int
    var bounces: [BounceInfo]
    var crossings: [Int]?   // 过网帧，每次过网 = 一板（旧记录没有这个字段）
    var path: [[Double]?]?  // 回合内每帧球的位置（原视频像素），nil = 看不到球；旧记录没有
    var shots: Int { crossings?.count ?? max(0, bounces.count - 1) }
}

/// 球台：四角（拉直后的视频像素坐标：远左、远右、近右、近左）+ 镜头畸变系数
struct TableFit: Codable, Hashable {
    var quad: [CGPoint]
    var k1: Double = 0
    func lens(_ size: CGSize) -> Lens { Lens(k1: k1, w: Double(size.width), h: Double(size.height)) }
}

struct SessionResult: Codable {
    var fps: Double
    var duration: Double
    var videoWidth: Double
    var videoHeight: Double
    var quad: [CGPoint]          // 拉直后的坐标
    var k1: Double?              // 镜头畸变；旧记录没有 = 0
    var rallies: [RallyInfo]
    var detectedRatio: Double

    var totalShots: Int { rallies.reduce(0) { $0 + $1.shots } }
    var longestRally: Int { rallies.map(\.shots).max() ?? 0 }
    var landings: [BounceInfo] { rallies.flatMap(\.bounces).filter { $0.out != true } }   // 落在台上的
    var speeds: [Double] { landings.compactMap(\.speedKmh) }
    var table: TableFit { TableFit(quad: quad, k1: k1 ?? 0) }
    var lens: Lens { table.lens(CGSize(width: videoWidth, height: videoHeight)) }
    func time(of frame: Int) -> Double { Double(frame) / fps }
}

enum AnalysisError: LocalizedError {
    case noVideoTrack, tableNotFound, readerFailed(String)
    var errorDescription: String? {
        switch self {
        case .noVideoTrack: return String(localized: "The file has no video track.")
        case .tableNotFound: return String(localized: "Couldn't find the table. Please mark its four corners.")
        case .readerFailed(let s): return s
        }
    }
}

// MARK: - 分析器

/// 本地离线分析：球追踪 → 球台识别 → 落台检测 → 回合统计。全部在设备上运行。
final class VideoAnalyzer {
    private let ball: MLModel
    private let seg: MLModel
    private let event: MLModel
    let inW: Int, inH: Int
    private let segW: Int, segH: Int   // 球台分割模型的输入尺寸，从模型读
    private let eventLen = 512
    /// 长时间没球时每隔几帧推理一次（1 = 不跳帧）
    var skipStride = 4
    /// 球台掩码投票阈值：一个像素在多少比例的抽样帧里被判成球台才算（球员遮挡时低一点能把被挡的那截补回来）
    var maskThreshold: Float = 0.5
    /// 上一次 track() 实际推理的帧占比
    private(set) var lastInferenceRatio = 1.0
    /// 调试：设置后 analyze() 会把中间结果写到这个文件
    var debugDump: URL?
    private var debugInfo: [String: Any] = [:]

    init(ball: MLModel, seg: MLModel, event: MLModel) {
        self.ball = ball; self.seg = seg; self.event = event
        let shape = ball.modelDescription.inputDescriptionsByName["cur"]!.multiArrayConstraint!.shape.map(\.intValue)
        inH = shape[2]; inW = shape[3]
        let ss = seg.modelDescription.inputDescriptionsByName["image"]!.multiArrayConstraint!.shape.map(\.intValue)
        segH = ss[2]; segW = ss[3]
    }

    struct VideoInfo { var size: CGSize; var fps: Double; var duration: Double; var frameCount: Int }

    static func info(of url: URL) async throws -> VideoInfo {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw AnalysisError.noVideoTrack }
        let (natural, transform, fps) = try await track.load(.naturalSize, .preferredTransform, .nominalFrameRate)
        let size = natural.applying(transform)
        let dur = try await asset.load(.duration).seconds
        return VideoInfo(size: CGSize(width: abs(size.width), height: abs(size.height)), fps: Double(fps), duration: dur, frameCount: Int((dur * Double(fps)).rounded()))
    }

    // MARK: 球台识别

    /// 在视频中均匀取若干帧做分割，平均掩码后拟合四角（源视频像素坐标）
    func detectTable(url: URL, samples: Int = 12) async throws -> TableFit {
        let asset = AVURLAsset(url: url)
        let info = try await Self.info(of: url)
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: segW, height: segH)
        gen.requestedTimeToleranceBefore = .positiveInfinity; gen.requestedTimeToleranceAfter = .positiveInfinity
        var acc = [Float](repeating: 0, count: segW * segH)
        var used = 0
        for i in 0..<samples {
            let t = CMTime(seconds: info.duration * (Double(i) + 0.5) / Double(samples), preferredTimescale: 600)
            guard let img = try? await gen.image(at: t).image, let input = rgbArray(from: img, w: segW, h: segH) else { continue }
            let out = try await seg.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": input]))
            let m = out.featureValue(for: "mask")!.multiArrayValue!
            let p = m.dataPointer.bindMemory(to: Float.self, capacity: segW * segH)
            if m.dataType == .float32 { for j in 0..<(segW * segH) { acc[j] += p[j] } }
            else { let h = m.dataPointer.bindMemory(to: Float16.self, capacity: segW * segH); for j in 0..<(segW * segH) { acc[j] += Float(h[j]) } }
            used += 1
        }
        guard used > 0 else { throw AnalysisError.tableNotFound }
        let prob = acc.map { $0 / Float(used) }
        let vw = Double(info.size.width), vh = Double(info.size.height)
        let sx = vw / Double(segW), sy = vh / Double(segH)

        // 按畸变系数 k1 把台面掩码「拉直」：拉直后每个像素去原画面里取（正向畸变是闭式的，不用迭代）
        func straightened(_ k1: Double) -> [Bool] {
            if abs(k1) < 1e-6 { return prob.map { $0 > maskThreshold } }
            let lens = Lens(k1: k1, w: vw, h: vh)
            var out = [Bool](repeating: false, count: segW * segH)
            for v in 0..<segH {
                for u in 0..<segW {
                    let d = lens.distort(CGPoint(x: (Double(u) + 0.5) * sx, y: (Double(v) + 0.5) * sy))
                    let su = Int(Double(d.x) / sx), sv = Int(Double(d.y) / sy)
                    if su >= 0, sv >= 0, su < segW, sv < segH, prob[sv * segW + su] > maskThreshold { out[v * segW + u] = true }
                }
            }
            return out
        }
        // 拉直后的台面有多像一个直边四边形：最大内接四边形和掩码的重合度
        func fit(_ on: [Bool]) -> (q: [CGPoint], score: Double, hull: [CGPoint], n: Int)? {
            let pts = Geometry.largestComponent(on, w: segW, h: segH)
            guard pts.count > 150 else { return nil }
            let hull = Geometry.convexHull(pts)
            guard let q = Geometry.maxAreaQuad(hull) else { return nil }
            // 得分 = 直不直（边缘到四边形的距离，越小越好，取负）
            return (q, -Geometry.edgeResidual(q, pts: pts, w: segW, h: segH), hull, pts.count)
        }
        // 先粗搜再细搜；普通镜头 k1≈0，只有明显更直（重合度高出 0.03 以上）才认为有畸变
        let base = fit(straightened(0))?.score ?? 0
        var bestK = 0.0, bestS = base
        // 自动估畸变：分割图分辨率不够时估不准（会把正常视频改坏），先关掉，只用 k1 = 0；以后改成在原分辨率上找台边直线再打开
        let estimateLens = ProcessInfo.processInfo.environment["EST_LENS"] != nil
        for k in stride(from: -0.5, through: 0.2, by: 0.05) where estimateLens && abs(k) > 1e-6 {
            if let f = fit(straightened(k)) {
                if ProcessInfo.processInfo.environment["DBG_LENS"] != nil { print(String(format: "  k1 %+.2f residual %.4f", k, -f.score)) }
                if f.score > bestS { bestS = f.score; bestK = k }
            }
        }
        if bestK != 0 {
            for k in stride(from: bestK - 0.04, through: bestK + 0.04, by: 0.01) {
                if let f = fit(straightened(k)), f.score > bestS { bestS = f.score; bestK = k }
            }
            if bestS / base > 0.8 { bestK = 0 }   // 边缘偏差至少减少 20% 才算有畸变（得分为负数）
        }
        let on = straightened(bestK)
        guard let f = fit(on) else { throw AnalysisError.tableNotFound }
        if debugDump != nil || ProcessInfo.processInfo.environment["DBG_LENS"] != nil { debugInfo["k1"] = bestK; print("lens k1", bestK, "score", bestS, "base", base) }
        // 按边长分出长边/短边，再微调四角（每个角保持身份）
        let q = Geometry.refineQuad(Geometry.orderQuad(f.q), mask: on, w: segW, h: segH)   // orderQuadByLength 在斜角机位上还没验证对，先用原来的
        // 形状检查：远边在上、近边在下，远边不比近边长太多，四边形要覆盖区域的大部分；不像球台就让用户手动标
        let area = Geometry.polygonArea(q), hullArea = Geometry.polygonArea(f.hull)
        let far = Geometry.dist(q[0], q[1]), near = Geometry.dist(q[3], q[2])
        guard q[0].y < q[3].y, q[1].y < q[2].y, far < near * 1.3, area > 0.6 * hullArea,
              area > 0.003 * Double(segW * segH), Double(f.n) > 0.45 * hullArea else { throw AnalysisError.tableNotFound }
        return TableFit(quad: q.map { CGPoint(x: (Double($0.x) + 0.5) * sx, y: (Double($0.y) + 0.5) * sy) }, k1: bestK)
    }


    // MARK: 球追踪

    static let decodeW = 1280, decodeH = 720
    private var scaled: [UInt8] = []   // vImage 缩放的中间缓冲，反复使用

    /// 球台周围的裁剪框：宽 = 2 倍台宽、和模型输入同比例，偏上一点（球在台面上方飞）。与训练（train_ball.py table_roi）一致。
    func tableROI(_ quad: [CGPoint], fw: Double, fh: Double) -> CGRect {
        let xs = quad.map { Double($0.x) }, ys = quad.map { Double($0.y) }
        let tw = xs.max()! - xs.min()!, cx = (xs.max()! + xs.min()!) / 2
        var rw = 2.0 * tw, rh = rw * Double(inH) / Double(inW)
        let cy = (ys.min()! + ys.max()!) / 2 - 0.2 * rh
        if rw > fw { rw = fw; rh = rw * Double(inH) / Double(inW) }
        if rh > fh { rh = fh; rw = rh * Double(inW) / Double(inH) }
        let x0 = min(max(0, cx - rw / 2), fw - rw), y0 = min(max(0, cy - rh / 2), fh - rh)
        return CGRect(x: x0, y: y0, width: rw, height: rh)
    }

    /// 逐帧追踪。第 i 帧的结果由第 i-2、i-1、i 帧推出（因果，与训练一致）。
    /// 先把整帧解码成 1280x720，再裁出球台周围一块缩放到模型输入大小：机位远、球台小时球也不会小到看不见。
    func track(url: URL, quad: [CGPoint]? = nil, progress: @escaping (Double) -> Void) async throws -> (Track, VideoInfo) {
        let info = try await Self.info(of: url)
        let asset = AVURLAsset(url: url)
        guard let vt = try await asset.loadTracks(withMediaType: .video).first else { throw AnalysisError.noVideoTrack }
        // 手机拍的视频常带旋转（竖拍、倒着横拍）：解码出来的帧是未旋转的原始方向，而球台四角和界面用的是转正后的坐标。
        // 下面所有坐标都按转正后的画面算，取像素时再按旋转关系去原始帧里取。
        let (natural, T) = try await vt.load(.naturalSize, .preferredTransform)
        let Tinv = T.inverted()
        let vw = Double(info.size.width), vh = Double(info.size.height)     // 转正后的尺寸
        // 转正画面里的裁剪框（球台周围），和训练一致
        let roiO = quad.map { q in tableROI(q, fw: vw, fh: vh) } ?? CGRect(x: 0, y: 0, width: vw, height: vh)
        // 解码尺寸按裁剪框定：裁剪框缩到模型输入时刚好不损失细节即可；球台占满画面时和以前一样小，远机位才解码得大
        let scale = min(Double(Self.decodeW) / Double(inW), max(1, vw / Double(roiO.width)))
        let s = Double(inW) * scale / vw                                    // 视频像素 -> 解码像素
        let dW = Int((Double(natural.width) * s / 2).rounded()) * 2, dH = Int((Double(natural.height) * s / 2).rounded()) * 2
        let roiNat = roiO.applying(Tinv).standardized                        // 原始方向下的裁剪框（视频像素）
        let crop = CGRect(x: roiNat.minX * s, y: roiNat.minY * s, width: roiNat.width * s, height: roiNat.height * s)
        let rotated = abs(T.b) > 0.5                                         // 90°/270°：宽高互换
        let sw = rotated ? inH : inW, sh = rotated ? inW : inH
        // 模型输入每个像素 (x,y) 对应缩放后原始方向图里的哪个像素：只算一次
        var gather = [Int32](repeating: 0, count: inW * inH)
        var identity = true
        for y in 0..<inH {
            for x in 0..<inW {
                let po = CGPoint(x: roiO.minX + (Double(x) + 0.5) * roiO.width / Double(inW), y: roiO.minY + (Double(y) + 0.5) * roiO.height / Double(inH))
                let pn = po.applying(Tinv)
                let gx = min(sw - 1, max(0, Int((pn.x - roiNat.minX) / roiNat.width * Double(sw))))
                let gy = min(sh - 1, max(0, Int((pn.y - roiNat.minY) / roiNat.height * Double(sh))))
                gather[y * inW + x] = Int32(gy * sw + gx)
                if gy * sw + gx != y * inW + x { identity = false }
            }
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: vt, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: dW, kCVPixelBufferHeightKey as String: dH,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else { throw AnalysisError.readerFailed(reader.error?.localizedDescription ?? "reader") }
        var xs: [Double] = [], ys: [Double] = [], cs: [Double] = []
        var buf: [MLMultiArray] = []          // 最近 bufLen 帧，buf.last 是当前帧
        // 高帧率视频隔帧推理：120fps 时每 2 帧算一次（模型训练时加过隔帧输入，60fps 和 120fps 结果一致），推理次数减半
        let step = Int(ProcessInfo.processInfo.environment["FSTEP"] ?? "") ?? (info.fps >= 100 ? 2 : 1)
        let bufLen = skipStride + 2, idleAfter = max(6, 24 / step)
        var frameIdx = 0
        var lastSeen = Int.min / 2
        // 模型输出（输入像素）-> 转正后的视频像素
        let sx = Double(roiO.width) / Double(inW), sy = Double(roiO.height) / Double(inH)
        let ox = Double(roiO.minX), oy = Double(roiO.minY)
        let total = max(1, info.frameCount)
        var inferred = 0

        // 用缓冲区里的第 j-2、j-1、j 帧推第 j 帧的球位置；返回是否有球
        func run(_ j: Int, current i: Int) async throws -> Bool {
            let k = buf.count - 1 - (i - j)
            let inp = try MLDictionaryFeatureProvider(dictionary: ["prev2": buf[k - 2], "prev1": buf[k - 1], "cur": buf[k]])
            let peak = try await ball.prediction(from: inp).featureValue(for: "peak")!.multiArrayValue!
            let c = peak[2].doubleValue
            cs[j] = c; inferred += 1
            guard c >= 0.5 else { return false }
            xs[j] = ox + peak[0].doubleValue * sx; ys[j] = oy + peak[1].doubleValue * sy
            return true
        }

        while let sb = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            defer { frameIdx += 1 }
            if frameIdx % step != 0 { continue }   // 隔帧：这一帧不用，连预处理都省掉
            buf.append(try planarFP16(pb, crop: crop, sw: sw, sh: sh, gather: identity ? nil : gather))
            if buf.count > bufLen { buf.removeFirst() }
            let i = xs.count
            xs.append(.nan); ys.append(.nan); cs.append(-1)   // cs = -1 表示这一帧没推理
            if i >= 2 {
                // 最近见过球就逐帧推理；长时间没球（捡球、准备发球）每 skipStride 帧查一次
                let dense = i - lastSeen <= idleAfter
                if dense || i % skipStride == 0 {
                    if try await run(i, current: i) {
                        if !dense {  // 刚发现球：把跳过的几帧补算回来，避免丢掉回合开头
                            for j in max(2, i - skipStride + 1)..<i where cs[j] < 0 {
                                if try await run(j, current: i) { lastSeen = max(lastSeen, j) }
                            }
                        }
                        lastSeen = i
                    }
                }
            }
            if xs.count % 60 == 0 { progress(min(1, Double(xs.count * step) / Double(total))) }
        }
        for j in cs.indices where cs[j] < 0 { cs[j] = 0 }
        lastInferenceRatio = Double(inferred) / Double(max(1, xs.count))
        if step > 1 {
            // 展开回原始帧号：算过的帧放回原位，中间跳过的帧前后都有球就线性插值
            let n = frameIdx
            var fx = [Double](repeating: .nan, count: n), fy = fx, fc = [Double](repeating: 0, count: n)
            for (p, _) in xs.enumerated() where p * step < n { fx[p * step] = xs[p]; fy[p * step] = ys[p]; fc[p * step] = cs[p] }
            for p in 0..<(xs.count - 1) {
                guard !xs[p].isNaN, !xs[p + 1].isNaN else { continue }
                for k in 1..<step where p * step + k < n {
                    let t = Double(k) / Double(step)
                    fx[p * step + k] = xs[p] * (1 - t) + xs[p + 1] * t; fy[p * step + k] = ys[p] * (1 - t) + ys[p + 1] * t
                    fc[p * step + k] = min(cs[p], cs[p + 1])
                }
            }
            xs = fx; ys = fy; cs = fc
        }
        if reader.status == .failed { throw AnalysisError.readerFailed(reader.error?.localizedDescription ?? "reader") }
        progress(1)
        return (Track(x: xs, y: ys, conf: cs), info)
    }

    // MARK: 落台概率

    /// EventNet 以 512 帧为一段、相邻段重叠 128 帧滑动，拼接时只取每段中间部分
    func bounceProbability(_ t: Track, quad: [CGPoint]) throws -> [Float] {
        let f = Events.features(t, quad: quad)
        let n = t.count, L = eventLen, margin = 64
        var prob = [Float](repeating: 0, count: n)
        var start = 0
        while start < n {
            let arr = try MLMultiArray(shape: [1, 4, NSNumber(value: L)], dataType: .float32)
            let p = arr.dataPointer.bindMemory(to: Float.self, capacity: 4 * L)
            for c in 0..<4 { for i in 0..<L { p[c * L + i] = start + i < n ? f[c][start + i] : 0 } }
            let out = try event.prediction(from: MLDictionaryFeatureProvider(dictionary: ["track": arr])).featureValue(for: "prob")!.multiArrayValue!
            if start == 0 && debugDump != nil { debugInfo["eventOut"] = "type \(out.dataType.rawValue) shape \(out.shape) strides \(out.strides) first \((0..<8).map { out[[0, 0, NSNumber(value: 60 + $0)]].floatValue })" }
            let lo = start == 0 ? 0 : margin, hi = start + L >= n ? L : L - margin
            for i in lo..<hi where start + i < n { prob[start + i] = out[[0, 0, NSNumber(value: i)]].floatValue }
            if start + L >= n { break }
            start += L - 2 * margin
        }
        return prob
    }

    // MARK: 汇总

    /// 每一板（每次过网）找它在对面台上的落点，只要有把握的：
    /// 1. 检测到的落台，且在台面上；或
    /// 2. 轨迹在台面上有一个「先往下、再往上」的拐点（落台反弹），拐点前后都看得到球。
    /// 都没有就不给落点（轨迹不清楚时宁可空着，也不乱标）
    /// 每一板过网后的落点。三种结果：落在台上（out=false）、确认没落在台上（out=true，出界）、看不清（不返回）
    static func landings(after crossings: [Int], bounces: [Int], track: Track, quad: [CGPoint], toTable: Homography, fps: Double) -> [(frame: Int, out: Bool, cross: Int)] {
        let maxLen = Int(1.5 * fps), k = max(1, Int((3 * fps / 120).rounded()))
        var out: [(Int, Bool, Int)] = []
        let tableW = Double(hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y))
        // 压线球：球心在画面里会落在四角连线外一点（四角本身也有几个像素误差），放宽到台长的 6% 再找一遍
        let loose = max(6, 0.06 * tableW)
        func dist(_ i: Int) -> Double? { track.point(i).map { Geometry.signedDistance($0, to: quad) } }
        // 球台长边方向（左端中点 -> 右端中点），用来看球沿台长往哪边飞
        let ux = (quad[1].x + quad[2].x - quad[0].x - quad[3].x) / 2, uy = (quad[1].y + quad[2].y - quad[0].y - quad[3].y) / 2
        let un = max(1e-6, hypot(ux, uy))
        func along(_ i: Int, _ j: Int) -> Double? {
            guard let a = track.point(i), let b = track.point(j) else { return nil }
            return Double(((b.x - a.x) * ux + (b.y - a.y) * uy) / un)
        }
        for (n, c) in crossings.enumerated() {
            let end = min(n + 1 < crossings.count ? crossings[n + 1] : c + maxLen, c + maxLen, track.count - 1)
            guard end > c + 2 * k else { continue }
            // 过网方向：+1 往右端飞，-1 往左端飞
            guard let v0 = along(max(0, c - k), min(track.count - 1, c + k)), abs(v0) > 0.5 else { continue }
            let dir: Double = v0 > 0 ? 1 : -1
            // 落台：落点在过网后那半张台上，且落台前后球都继续往同一方向飞；球拍击球后球会往回飞，不算
            func isLanding(_ i: Int, margin: Double) -> Bool {
                guard let d = dist(i), d > -margin, let p = track.point(i) else { return false }
                let tx = Double(toTable.apply(p).x)
                guard (tx - Table.length / 2) * dir > 0.05 else { return false }
                guard let vb = along(max(c, i - 2 * k), i), let va = along(i, min(end, i + 2 * k)) else { return false }
                return vb * dir > 0 && va * dir > 0
            }
            func find(margin: Double) -> Int? {
                if let b = bounces.first(where: { $0 > c && $0 < end && isLanding($0, margin: margin) }) { return b }
                var best: Int?, bestY = -Double.infinity
                for i in (c + k)..<(end - k + 1) where isLanding(i, margin: margin) {
                    guard let p = track.point(i), let a = track.point(i - k), let b = track.point(i + k) else { continue }
                    // 图像里 y 向下：落台前 y 在变大（往下掉），落台后 y 变小（弹起）；最低处可能连着两帧一样高（平顶），再往外看一步
                    guard p.y >= a.y, p.y >= b.y else { continue }
                    let a2 = track.point(max(0, i - 2 * k)) ?? a, b2 = track.point(min(track.count - 1, i + 2 * k)) ?? b
                    guard p.y > a2.y + 1, p.y > b2.y + 1 else { continue }
                    if Double(p.y) > bestY { bestY = Double(p.y); best = i }
                }
                return best
            }
            if let f = find(margin: 6) ?? find(margin: loose) { out.append((f, false, c)); continue }
            // 没落台：过网后球一路往下掉（中间没有弹起过），并且飞出了端线或近侧边线 -> 出界。
            // 中间弹起过、或者跟丢了的，说不清落在哪，不标
            var prevY = -Double.infinity, exit: Int?
            for i in (c + 1)...end {
                guard let p = track.point(i) else { continue }
                if Double(p.y) < prevY - 2 { break }   // 弹起来了
                prevY = max(prevY, Double(p.y))
                let t = toTable.apply(p)
                // 球在台面以上时，投影只会偏向远侧；投影都已经在近侧边线外，球一定在台外
                if (Double(t.x) - Table.length / 2) * dir > Table.length / 2 + 0.03 || Double(t.y) > Table.width + 0.03 { exit = i; break }
            }
            if let exit { out.append((exit, true, c)) }
        }
        return out
    }

    /// 过网到落台这一段的平均水平速度（km/h）。过网时球一定在球网正上方（x = 台长一半），落台点在台面上，
    /// 两个位置都不受球离台面高度影响，比用空中的球投影到台面准得多。离网太近或结果离谱的不给数
    static func netToLanding(cross: Int, landing: Int, tableX: Double, fps: Double) -> Double? {
        let dt = Double(landing - cross) / fps, dx = abs(tableX - Table.length / 2)
        guard dt > 0.03, dx > 0.2, dx < Table.length / 2 + 0.1 else { return nil }
        let v = dx / dt * 3.6
        return (8...160).contains(v) ? v : nil
    }

    func summarize(raw: Track, prob: [Float], quad: [CGPoint], lens: Lens? = nil, info: VideoInfo) -> SessionResult {
        let fpsScale = info.fps / 120
        let clean = Events.clean(raw)           // 画面坐标：回放、导出画轨迹用
        // 拉直后的轨迹：球台四角、球网、落点、速度都在这个坐标里算
        let lens = lens ?? Lens(k1: 0, w: Double(info.size.width), h: Double(info.size.height))
        var ux = clean.x, uy = clean.y
        if !lens.isIdentity {
            for i in ux.indices where !ux[i].isNaN { let u = lens.undistort(CGPoint(x: ux[i], y: uy[i])); ux[i] = Double(u.x); uy[i] = Double(u.y) }
        }
        let straight = Track(x: ux, y: uy, conf: clean.conf)
        let toTable = Homography.toTable(quad: quad)!
        let bounces = Events.finalBounces(raw: raw, prob: prob, quad: quad.map(lens.distort))
        let cross = Events.crossings(straight, quad: quad, fps: info.fps)
        var rallies: [RallyInfo] = []
        // 两段事件的剪辑范围有重叠（放高球、中间短暂看不到球）就是同一分球，合成一个回合
        var groups: [[Int]] = []
        for g in Events.rallies(bounces + cross, gap: Int(200 * fpsScale)) {
            // 只比较连续看得到球的那段（不含前后留的余量），避免把两分球连起来
            let margin = Int(2.2 * 120 * fpsScale)
            if let last = groups.last,
               rallySpan(first: g.min()!, last: g.max()!, raw: raw, fpsScale: fpsScale).0 + margin <= rallySpan(first: last.min()!, last: last.max()!, raw: raw, fpsScale: fpsScale).1 {
                groups[groups.count - 1] += g
            } else { groups.append(g) }
        }
        for group in groups {
            let inGroup = Set(group)
            let kept = Events.alternate(bounces.filter(inGroup.contains), track: straight, toTable: toTable)
            let crossed = cross.filter(inGroup.contains)
            // 至少一板：球过了网，并且落过台或者来回过（只飞过去一次多半是递球）
            guard crossed.count >= 2 || (crossed.count == 1 && !kept.isEmpty) else { continue }
            // 每一板过网后，在对面半台找它的落点：优先用检测到的落台；没检测到就取这段轨迹里台面上的最低点（落台反弹处）
            let landingFrames = Self.landings(after: crossed, bounces: kept, track: straight, quad: quad, toTable: toTable, fps: info.fps)
            let infos = landingFrames.compactMap { l -> BounceInfo? in
                guard let p = straight.point(l.frame) else { return nil }
                var t = toTable.apply(p)
                if l.out {
                    // 出界只标大概位置：飞出端线的画在端线外，从近侧飞出的画在边线外
                    let beyondEnd = t.x < 0 || t.x > Table.length
                    t.x = min(max(t.x, -0.12), Table.length + 0.12)
                    t.y = beyondEnd ? min(max(t.y, 0.1), Table.width - 0.1) : Table.width + 0.12
                } else {
                    t.x = min(max(t.x, 0), Table.length); t.y = min(max(t.y, 0), Table.width)
                }
                return BounceInfo(frame: l.frame, tableX: Double(t.x), tableY: Double(t.y),
                                  speedKmh: l.out ? nil : Self.netToLanding(cross: l.cross, landing: l.frame, tableX: Double(t.x), fps: info.fps), out: l.out ? true : nil)
            }
            let prevEnd = rallies.last?.endFrame ?? 0
            let (s, e) = rallySpan(first: group.min()!, last: group.max()!, raw: raw, fpsScale: fpsScale)
            let st = max(prevEnd, s)
            // 存下回合内的球轨迹（原视频像素，看不到球的帧为 nil），回放和导出时画轨迹特效
            let path: [[Double]?] = (st...e).map { i in clean.point(i).map { [Double($0.x), Double($0.y)] } }
            rallies.append(RallyInfo(id: rallies.count + 1, startFrame: st, endFrame: e, bounces: infos, crossings: crossed, path: path))
        }
        let detected = Double(raw.x.filter { !$0.isNaN }.count) / Double(max(1, raw.count))
        return SessionResult(fps: info.fps, duration: info.duration, videoWidth: info.size.width, videoHeight: info.size.height,
                             quad: quad, k1: lens.isIdentity ? nil : lens.k1, rallies: rallies, detectedRatio: detected)
    }

    /// 回合剪辑范围：从第一次落台往前、最后一次落台往后，沿着连续看得到球的帧延伸（抛球发球、最后一拍、球出界），
    /// 再前后各留一点余量，保证剪出来是完整的一分球
    func rallySpan(first: Int, last: Int, raw: Track, fpsScale: Double) -> (Int, Int) {
        let f = { (x: Double) in Int(x * 120 * fpsScale) }
        let maxGap = f(0.4), maxExtend = f(3)
        var s = first, miss = 0
        var i = first - 1
        while i >= 0 && first - i <= maxExtend {
            if raw.x[i].isNaN { miss += 1; if miss > maxGap { break } } else { miss = 0; s = i }
            i -= 1
        }
        var e = last; miss = 0; i = last + 1
        while i < raw.count && i - last <= maxExtend {
            if raw.x[i].isNaN { miss += 1; if miss > maxGap { break } } else { miss = 0; e = i }
            i += 1
        }
        return (max(0, s - f(1.0)), min(raw.count - 1, e + f(1.2)))
    }

    func analyze(url: URL, table: TableFit?, progress: @escaping (Double, String) -> Void) async throws -> SessionResult {
        progress(0, String(localized: "Finding the table…"))
        let tf: TableFit
        if let table { tf = table } else { tf = try await detectTable(url: url) }
        let info0 = try await Self.info(of: url)
        let lens = tf.lens(info0.size)
        let qImg = tf.quad.map(lens.distort)   // 画面里看到的四角：裁剪框、落台模型的输入都按画面坐标
        let q = tf.quad
        let (t, info) = try await track(url: url, quad: qImg) { progress(0.05 + 0.9 * $0, String(localized: "Tracking the ball…")) }
        progress(0.96, String(localized: "Counting shots…"))
        let p = try bounceProbability(t, quad: qImg)
        var r = summarize(raw: t, prob: p, quad: q, lens: lens, info: info)
        r.k1 = tf.k1
        if let debugDump {  // 调试：输出中间结果，便于和命令行版逐项对比
            let d: [String: Any] = ["quad": q.map { [$0.x, $0.y] }, "x": t.x.map { $0.isNaN ? -1 : $0 }, "y": t.y.map { $0.isNaN ? -1 : $0 },
                                    "conf": t.conf, "prob": p.map(Double.init), "rallies": r.rallies.map { $0.bounces.map(\.frame) }].merging(debugInfo) { a, _ in a }
            try? JSONSerialization.data(withJSONObject: d).write(to: debugDump)
        }
        progress(1, String(localized: "Done"))
        return r
    }

    // MARK: 像素转换

    /// BGRA 像素缓冲 -> [1,3,H,W] fp16，RGB，0…1
    /// BGRA 帧的 roi 区域缩放到 [1,3,inH,inW] fp16（RGB，0…1）。缩放用 vImage（硬件加速，带抗锯齿，接近训练时的缩放方式）
    /// 原始方向帧里的 crop 区域缩放到 sw×sh（vImage，带抗锯齿），再按 gather 表取像素转正成 [1,3,inH,inW] fp16（RGB，0…1）
    private func planarFP16(_ pb: CVPixelBuffer, crop: CGRect, sw: Int, sh: Int, gather: [Int32]?) throws -> MLMultiArray {
        CVPixelBufferLockBaseAddress(pb, .readOnly); defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), rb = CVPixelBufferGetBytesPerRow(pb)
        let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        let x0 = max(0, min(w - 2, Int(crop.minX.rounded()))), y0 = max(0, min(h - 2, Int(crop.minY.rounded())))
        let cw = max(2, min(w - x0, Int(crop.width.rounded()))), ch = max(2, min(h - y0, Int(crop.height.rounded())))
        let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: inH), NSNumber(value: inW)], dataType: .float16)
        let dst = arr.dataPointer.bindMemory(to: Float16.self, capacity: 3 * inW * inH)
        let plane = inW * inH
        let k: Float16 = 1.0 / 255
        if gather == nil && cw == inW && ch == inH {  // 不旋转也不缩放：直接拆通道（球台占满画面的常见情况）
            let p = base + y0 * rb + x0 * 4
            for yy in 0..<inH {
                let row = p + yy * rb
                for xx in 0..<inW {
                    let q = row + xx * 4, o = yy * inW + xx
                    dst[o] = Float16(q[2]) * k; dst[plane + o] = Float16(q[1]) * k; dst[2 * plane + o] = Float16(q[0]) * k
                }
            }
            return arr
        }
        var src = vImage_Buffer(data: base + y0 * rb + x0 * 4, height: vImagePixelCount(ch), width: vImagePixelCount(cw), rowBytes: rb)
        if scaled.count != sw * sh * 4 { scaled = [UInt8](repeating: 0, count: sw * sh * 4) }
        try scaled.withUnsafeMutableBytes { sp in
            var out = vImage_Buffer(data: sp.baseAddress, height: vImagePixelCount(sh), width: vImagePixelCount(sw), rowBytes: sw * 4)
            let err = vImageScale_ARGB8888(&src, &out, nil, vImage_Flags(kvImageNoFlags))
            guard err == kvImageNoError else { throw AnalysisError.readerFailed("scale \(err)") }
            let p = sp.bindMemory(to: UInt8.self)
            if let gather {
                for o in 0..<plane {
                    let g = Int(gather[o]) * 4
                    dst[o] = Float16(p[g + 2]) * k; dst[plane + o] = Float16(p[g + 1]) * k; dst[2 * plane + o] = Float16(p[g]) * k
                }
            } else {
                for o in 0..<plane {
                    dst[o] = Float16(p[o * 4 + 2]) * k; dst[plane + o] = Float16(p[o * 4 + 1]) * k; dst[2 * plane + o] = Float16(p[o * 4]) * k
                }
            }
        }
        return arr
    }

    /// CGImage -> [1,3,h,w] fp32，RGB，0…1
    private func rgbArray(from img: CGImage, w: Int, h: Int) -> MLMultiArray? {
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        ctx.interpolationQuality = .medium
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let arr = try? MLMultiArray(shape: [1, 3, NSNumber(value: h), NSNumber(value: w)], dataType: .float32) else { return nil }
        let p = arr.dataPointer.bindMemory(to: Float.self, capacity: 3 * w * h)
        let plane = w * h
        for i in 0..<plane {
            p[i] = Float(buf[i * 4]) / 255; p[plane + i] = Float(buf[i * 4 + 1]) / 255; p[2 * plane + i] = Float(buf[i * 4 + 2]) / 255
        }
        return arr
    }
}
