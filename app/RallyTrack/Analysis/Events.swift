import CoreGraphics
import Foundation

/// 球轨迹 -> 落台 / 击球 / 回合。与 Python 版 events.py / vis_events.py 逻辑一致（参数以 120fps 为准）。
struct Track {
    var x: [Double]
    var y: [Double]
    var conf: [Double]
    var count: Int { x.count }
    func point(_ i: Int) -> CGPoint? { x[i].isNaN ? nil : CGPoint(x: x[i], y: y[i]) }
}

enum Events {
    /// 剔除孤立跳点，再线性补齐短缺口（clean）
    static func clean(_ t: Track, maxGap: Int = 6, jump: Double = 120) -> Track {
        var x = t.x, y = t.y
        func d(_ a: Int, _ b: Int) -> Double { hypot(x[a] - x[b], y[a] - y[b]) }
        var idx = x.indices.filter { !x[$0].isNaN }
        if idx.count >= 3 {
            var drop: [Int] = []
            for k in 1..<(idx.count - 1) {
                let a = idx[k - 1], b = idx[k], c = idx[k + 1]
                if d(b, a) > jump * Double(b - a).squareRoot() && d(c, b) > jump * Double(c - b).squareRoot() { drop.append(b) }
            }
            for b in drop { x[b] = .nan; y[b] = .nan }
        }
        idx = x.indices.filter { !x[$0].isNaN }
        for (a, b) in zip(idx, idx.dropFirst()) where b - a > 1 && b - a <= maxGap && d(a, b) < jump * Double(b - a).squareRoot() {
            for i in (a + 1)..<b {
                let s = Double(i - a) / Double(b - a)
                x[i] = x[a] * (1 - s) + x[b] * s; y[i] = y[a] * (1 - s) + y[b] * s
            }
        }
        return Track(x: x, y: y, conf: t.conf)
    }

    /// 窗口均值平滑（smooth）
    static func smooth(_ v: [Double], k: Int = 3) -> [Double] {
        v.indices.map { i in
            if v[i].isNaN { return .nan }
            let w = v[max(0, i - k)...min(v.count - 1, i + k)].filter { !$0.isNaN }
            return w.count >= k ? w.reduce(0, +) / Double(w.count) : .nan
        }
    }

    struct RuleResult { var bounces: [Int]; var hits: [Int]; var smoothed: Track }

    /// 规则检测（detect）：落台 = 画面 y 局部最大且在台面内；击球 = 水平方向反转
    static func detect(_ raw: Track, quad: [CGPoint], win: Int = 4, minSep: Int = 10, tableMargin: Double = 12, minVx: Double = 15) -> RuleResult {
        let c = clean(raw)
        let x = smooth(c.x), y = smooth(c.y)
        var bounces: [Int] = [], hits: [Int] = []
        let n = x.count
        guard n > 2 * win else { return RuleResult(bounces: [], hits: [], smoothed: Track(x: x, y: y, conf: raw.conf)) }
        for i in win..<(n - win) {
            if y[i].isNaN { continue }
            let pre = y[(i - win)..<i], post = y[(i + 1)...(i + win)]
            if pre.contains(where: { $0.isNaN }) || post.contains(where: { $0.isNaN }) { continue }
            let vx0 = x[i] - x[i - win], vx1 = x[i + win] - x[i]
            if y[i] >= pre.max()! && y[i] >= post.max()! && y[i] - pre.first! > 3 && y[i] - post.last! > 3 {
                let inPlay = vx0 * vx1 > 0 && min(abs(vx0), abs(vx1)) >= minVx
                if inPlay && Geometry.signedDistance(CGPoint(x: x[i], y: y[i]), to: quad) > -tableMargin {
                    if bounces.last.map({ i - $0 >= minSep }) ?? true { bounces.append(i) }
                }
            }
            if !vx0.isNaN && !vx1.isNaN && vx0 * vx1 < 0 && abs(vx0) > 6 && abs(vx1) > 6 {
                if hits.last.map({ i - $0 >= minSep }) ?? true { hits.append(i) }
            }
        }
        return RuleResult(bounces: bounces, hits: hits, smoothed: Track(x: x, y: y, conf: raw.conf))
    }

    /// EventNet 输入特征（features）：按球台外接框归一化的坐标、是否有球、置信度，4×N
    static func features(_ raw: Track, quad: [CGPoint]) -> [[Float]] {
        let c = clean(raw)
        let x0 = quad.map(\.x).min()!, x1 = quad.map(\.x).max()!, y0 = quad.map(\.y).min()!, y1 = quad.map(\.y).max()!
        var f = [[Float]](repeating: [Float](repeating: 0, count: c.count), count: 4)
        for i in 0..<c.count {
            if !c.x[i].isNaN {
                f[0][i] = Float((c.x[i] - x0) / (x1 - x0)); f[1][i] = Float((c.y[i] - y0) / (y1 - y0)); f[2][i] = 1
            }
            f[3][i] = Float(c.conf[i])
        }
        return f
    }

    /// 非极大抑制取峰（peaks）
    static func peaks(_ p: [Float], threshold: Float, nms: Int = 10) -> [Int] {
        var out: [Int] = []
        for i in p.indices.sorted(by: { p[$0] > p[$1] }) {
            if p[i] < threshold { break }
            if out.allSatisfy({ abs(i - $0) >= nms }) { out.append(i) }
        }
        return out.sorted()
    }

    /// 合法回合里落台在球网两侧交替；同侧连续的只保留第一个（alternate）
    static func alternate(_ bounces: [Int], track: Track, toTable: Homography) -> [Int] {
        guard let first = bounces.first else { return [] }
        func side(_ f: Int) -> Bool { track.point(f).map { toTable.apply($0).x < Table.length / 2 } ?? false }
        var keep = [first], last = side(first)
        for f in bounces.dropFirst() {
            let s = side(f)
            if s != last { keep.append(f); last = s }
        }
        return keep
    }

    /// 球过网的帧（crossings）：球从球网一侧翻到另一侧算一次，也就是一板。
    /// 比「落台次数 - 1」可靠：最后一板打丢、擦边、落台没看清都不影响计数。
    static func crossings(_ track: Track, quad: [CGPoint], fps: Double) -> [Int] {
        guard let toImage = Homography(from: [CGPoint(x: 0, y: 0), CGPoint(x: Table.length, y: 0),
                                              CGPoint(x: Table.length, y: Table.width), CGPoint(x: 0, y: Table.width)], to: quad) else { return [] }
        // 真实球网位置按透视投影回画面，斜着拍时球网不在四角连线的中点
        let a = toImage.apply(CGPoint(x: Table.length / 2, y: 0)), b = toImage.apply(CGPoint(x: Table.length / 2, y: Table.width))
        let len = max(1, hypot(b.x - a.x, b.y - a.y))
        var nx = Double(b.y - a.y) / Double(len), ny = -Double(b.x - a.x) / Double(len)
        if (Double(quad[1].x - a.x) * nx + Double(quad[1].y - a.y) * ny) < 0 { nx = -nx; ny = -ny }  // 远右一侧为正
        let tableW = Double(hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y))
        let k = fps / 120, margin = 0.02 * tableW
        let confirm = max(2, Int((3 * k).rounded())), maxGap = max(8, Int(30 * k)), minSpeed = 3 / k
        var out: [Int] = []
        var cur = 0, lastI = 0, cand = 0, candN = 0, candFirst = 0
        var lastD = 0.0, candD = 0.0
        for i in 0..<track.count where !track.x[i].isNaN {
            let d = (track.x[i] - Double(a.x)) * nx + (track.y[i] - Double(a.y)) * ny
            guard abs(d) <= tableW, abs(d) > margin else { continue }
            let s = d > 0 ? 1 : -1
            if cur == 0 { cur = s; lastI = i; lastD = d; continue }
            if s == cur { lastI = i; lastD = d; candN = 0; continue }
            if candN == 0 || s != cand { cand = s; candN = 1; candFirst = i; candD = d } else { candN += 1 }
            if candN >= confirm {
                let gap = candFirst - lastI
                // 太慢的是捡球、递球过网，不算一板
                // 必须从球网上方过：过网点沿球网方向（远端 0 -> 近端 1）不能低于近端台边，地上滚过去、弹过去的球不算一板
                let mx = (track.x[lastI] + track.x[candFirst]) / 2, my = (track.y[lastI] + track.y[candFirst]) / 2
                let u = ((mx - Double(a.x)) * Double(b.x - a.x) + (my - Double(a.y)) * Double(b.y - a.y)) / Double(len * len)
                if gap <= maxGap, u <= 1.05, (abs(lastD) + abs(candD)) / Double(max(1, gap)) >= minSpeed { out.append((lastI + candFirst) / 2) }
                cur = s; lastI = i; lastD = d; candN = 0
            }
        }
        return out
    }

    /// 按间隔切分回合（rallies）
    static func rallies(_ events: [Int], gap: Int = 150) -> [[Int]] {
        var out: [[Int]] = [], cur: [Int] = []
        for f in events.sorted() {
            if let l = cur.last, f - l > gap { out.append(cur); cur = [] }
            cur.append(f)
        }
        if !cur.isEmpty { out.append(cur) }
        return out
    }

    /// 落台前 win 帧沿球台长度方向的平均速度（km/h），球贴近台面时透视变换基本准确
    static func bounceSpeed(_ f: Int, track: Track, toTable: Homography, fps: Double, win: Int = 4) -> Double? {
        guard f - win >= 0, let a = track.point(f - win), let b = track.point(f) else { return nil }
        for i in (f - win)...f where track.point(i) == nil { return nil }
        let dx = abs(toTable.apply(b).x - toTable.apply(a).x)
        return Double(dx) / (Double(win) / fps) * 3.6
    }

    /// 最终落台：候选 = 轨迹模型峰值(0.3) ∪ 规则落台 ∪ 击球点（相距 >4 帧去重），保留轨迹概率 ≥0.5 的
    static func finalBounces(raw: Track, prob: [Float], quad: [CGPoint]) -> [Int] {
        let rule = detect(raw, quad: quad, tableMargin: 40, minVx: 0)
        var cand: [Int] = []
        for k in Set(peaks(prob, threshold: 0.3)).union(rule.bounces).union(rule.hits).sorted() {
            if cand.last.map({ k - $0 > 4 }) ?? true { cand.append(k) }
        }
        return cand.filter { $0 < prob.count && prob[$0] >= 0.5 }
    }
}
