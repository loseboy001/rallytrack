import CoreGraphics
import Foundation

/// 标准球台尺寸（米）
enum Table {
    static let length = 2.74
    static let width = 1.525
}

/// 4 点透视变换（DLT）。src 为图像像素，dst 为目标坐标（如球台米制坐标）。
struct Homography {
    private let h: [Double]  // 3x3 行优先，h[8] = 1

    init?(from src: [CGPoint], to dst: [CGPoint]) {
        guard src.count == 4, dst.count == 4 else { return nil }
        var a = [[Double]](repeating: [Double](repeating: 0, count: 9), count: 8)
        for i in 0..<4 {
            let x = Double(src[i].x), y = Double(src[i].y), u = Double(dst[i].x), v = Double(dst[i].y)
            a[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y, u]
            a[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y, v]
        }
        guard let sol = Homography.solve(a) else { return nil }
        h = sol + [1]
    }

    /// 图像像素 -> 球台米制坐标的变换（四角顺序：远左、远右、近右、近左）
    static func toTable(quad: [CGPoint]) -> Homography? {
        Homography(from: quad, to: [CGPoint(x: 0, y: 0), CGPoint(x: Table.length, y: 0),
                                     CGPoint(x: Table.length, y: Table.width), CGPoint(x: 0, y: Table.width)])
    }

    func apply(_ p: CGPoint) -> CGPoint {
        let x = Double(p.x), y = Double(p.y)
        let w = h[6] * x + h[7] * y + h[8]
        return CGPoint(x: (h[0] * x + h[1] * y + h[2]) / w, y: (h[3] * x + h[4] * y + h[5]) / w)
    }

    /// 8x9 增广矩阵高斯消元
    private static func solve(_ m: [[Double]]) -> [Double]? {
        var a = m
        let n = 8
        for c in 0..<n {
            guard let p = (c..<n).max(by: { abs(a[$0][c]) < abs(a[$1][c]) }), abs(a[p][c]) > 1e-12 else { return nil }
            a.swapAt(c, p)
            for r in 0..<n where r != c {
                let f = a[r][c] / a[c][c]
                if f != 0 { for k in c...n { a[r][k] -= f * a[c][k] } }
            }
        }
        return (0..<n).map { a[$0][n] / a[$0][$0] }
    }
}

/// 镜头径向畸变（一个参数 k1，OpenCV 同款模型）：拉直坐标 = 理想针孔镜头下的位置；畸变坐标 = 视频里实际看到的位置。
/// 坐标按画面中心、半对角线归一化。k1 < 0 是广角常见的桶形畸变
struct Lens: Hashable {
    var k1: Double
    var w: Double, h: Double
    var f: Double { hypot(w, h) / 2 }
    var isIdentity: Bool { abs(k1) < 1e-6 }

    /// 拉直坐标 -> 画面里看到的位置
    func distort(_ p: CGPoint) -> CGPoint {
        guard !isIdentity else { return p }
        let x = (Double(p.x) - w / 2) / f, y = (Double(p.y) - h / 2) / f, s = 1 + k1 * (x * x + y * y)
        return CGPoint(x: w / 2 + x * s * f, y: h / 2 + y * s * f)
    }

    /// 画面里看到的位置 -> 拉直坐标（迭代求逆）
    func undistort(_ p: CGPoint) -> CGPoint {
        guard !isIdentity else { return p }
        let xd = (Double(p.x) - w / 2) / f, yd = (Double(p.y) - h / 2) / f
        var x = xd, y = yd
        for _ in 0..<12 { let s = 1 + k1 * (x * x + y * y); x = xd / s; y = yd / s }
        return CGPoint(x: w / 2 + x * f, y: h / 2 + y * f)
    }

    /// 拉直坐标里的一条直线段，在画面里是一条弧：取若干点画出来
    func curve(_ a: CGPoint, _ b: CGPoint, n: Int = 24) -> [CGPoint] {
        (0...n).map { i in
            let t = Double(i) / Double(n)
            return distort(CGPoint(x: Double(a.x) + (Double(b.x) - Double(a.x)) * t, y: Double(a.y) + (Double(b.y) - Double(a.y)) * t))
        }
    }
}

enum Geometry {
    /// Andrew 单调链凸包，逆时针（图像坐标系下为顺时针）
    static func convexHull(_ points: [CGPoint]) -> [CGPoint] {
        let p = points.sorted { $0.x == $1.x ? $0.y < $1.y : $0.x < $1.x }
        guard p.count > 2 else { return p }
        func cross(_ o: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat { (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x) }
        var lower: [CGPoint] = [], upper: [CGPoint] = []
        for q in p {
            while lower.count >= 2 && cross(lower[lower.count - 2], lower[lower.count - 1], q) <= 0 { lower.removeLast() }
            lower.append(q)
        }
        for q in p.reversed() {
            while upper.count >= 2 && cross(upper[upper.count - 2], upper[upper.count - 1], q) <= 0 { upper.removeLast() }
            upper.append(q)
        }
        return Array(lower.dropLast() + upper.dropLast())
    }

    /// 闭合多边形的 Douglas-Peucker 简化（对应 cv2.approxPolyDP(closed=True)）
    static func simplifyClosed(_ poly: [CGPoint], epsilon: Double) -> [CGPoint] {
        guard poly.count > 3 else { return poly }
        // 以离首点最远的点切成两段开放折线
        let far = poly.indices.max { dist(poly[0], poly[$0]) < dist(poly[0], poly[$1]) }!
        let a = Array(poly[0...far]), b = Array(poly[far...]) + [poly[0]]
        return Array(dp(a, epsilon).dropLast() + dp(b, epsilon).dropLast())
    }

    private static func dp(_ pts: [CGPoint], _ eps: Double) -> [CGPoint] {
        guard pts.count > 2 else { return pts }
        var maxD = 0.0, idx = 0
        for i in 1..<(pts.count - 1) {
            let d = segDist(pts[i], pts[0], pts[pts.count - 1])
            if d > maxD { maxD = d; idx = i }
        }
        if maxD <= eps { return [pts[0], pts[pts.count - 1]] }
        return Array(dp(Array(pts[0...idx]), eps).dropLast() + dp(Array(pts[idx...]), eps))
    }

    static func dist(_ a: CGPoint, _ b: CGPoint) -> Double { Double(hypot(a.x - b.x, a.y - b.y)) }

    /// 凸包上面积最大的内接四边形。比"逐步简化到 4 点"稳定：边缘的小起伏（球网立柱等）不会被选成角点。
    static func maxAreaQuad(_ hullIn: [CGPoint]) -> [CGPoint]? {
        var hull = hullIn
        if hull.count > 120 {  // 点太多时等间隔抽样，O(n^3) 仍很快
            let step = Double(hull.count) / 120
            hull = (0..<120).map { hull[Int(Double($0) * step)] }
        }
        let n = hull.count
        guard n >= 4 else { return nil }
        func tri(_ a: Int, _ b: Int, _ c: Int) -> Double {
            let p = hull[a], q = hull[b], r = hull[c]
            return abs(Double((q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x))) / 2
        }
        var best = -1.0, res = [0, 1, 2, 3]
        for i in 0..<(n - 2) {
            for j in (i + 2)..<n {
                guard j - i < n - 1 else { continue }
                var bk = -1, ak = -1.0, bl = -1, al = -1.0
                for k in (i + 1)..<j { let t = tri(i, k, j); if t > ak { ak = t; bk = k } }
                var l = (j + 1) % n
                while l != i { let t = tri(j, l, i); if t > al { al = t; bl = l }; l = (l + 1) % n }
                if bk >= 0, bl >= 0, ak + al > best { best = ak + al; res = [i, bk, j, bl] }
            }
        }
        return res.map { hull[$0] }
    }

    static func segDist(_ p: CGPoint, _ a: CGPoint, _ b: CGPoint) -> Double {
        let dx = b.x - a.x, dy = b.y - a.y
        let l2 = dx * dx + dy * dy
        if l2 == 0 { return dist(p, a) }
        let t = max(0, min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / l2))
        return dist(p, CGPoint(x: a.x + t * dx, y: a.y + t * dy))
    }

    /// 球台区域的像素坐标：取最大的连通块，再把被球网切开的另一半（同一水平带上、面积不太小的块）合进来；
    /// 离得远的块（旁边的球台、地上的反光）不要
    static func largestComponent(_ on: [Bool], w: Int, h: Int) -> [CGPoint] {
        var seen = [Bool](repeating: false, count: on.count), comps: [[Int]] = []
        for start in on.indices where on[start] && !seen[start] {
            var stack = [start], comp: [Int] = []; seen[start] = true
            while let i = stack.popLast() {
                comp.append(i)
                let x = i % w, y = i / w
                for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] where nx >= 0 && ny >= 0 && nx < w && ny < h {
                    let j = ny * w + nx
                    if on[j] && !seen[j] { seen[j] = true; stack.append(j) }
                }
            }
            comps.append(comp)
        }
        guard let main = comps.max(by: { $0.count < $1.count }) else { return [] }
        func box(_ c: [Int]) -> (Int, Int, Int, Int) {
            var x0 = w, x1 = 0, y0 = h, y1 = 0
            for i in c { let x = i % w, y = i / w; x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y) }
            return (x0, x1, y0, y1)
        }
        let mb = box(main), mw = mb.1 - mb.0 + 1, mh = mb.3 - mb.2 + 1
        var keep = main
        for c in comps where c != main && c.count * 8 >= main.count {
            let b = box(c)
            let gap = max(0, max(b.0 - mb.1, mb.0 - b.1))                      // 水平间隙
            let overlapY = min(b.3, mb.3) - max(b.2, mb.2)                      // 竖直方向重叠
            if gap <= mw / 4, overlapY > -mh / 2 { keep += c }
        }
        return keep.map { CGPoint(x: $0 % w, y: $0 / w) }
    }

    /// 台面边缘有多「直」：掩码边界像素到四边形四条边的平均距离（越小越直），按四边形尺寸归一化
    static func edgeResidual(_ q: [CGPoint], pts: [CGPoint], w: Int, h: Int) -> Double {
        var inside = Set<Int>(); for p in pts { inside.insert(Int(p.y) * w + Int(p.x)) }
        var sum = 0.0, n = 0
        for p in pts {
            let x = Int(p.x), y = Int(p.y)
            // 边界：四邻域里有不在区域内的点
            guard !(inside.contains(y * w + x - 1) && inside.contains(y * w + x + 1) && inside.contains((y - 1) * w + x) && inside.contains((y + 1) * w + x)) else { continue }
            var d = Double.infinity
            for k in 0..<4 { d = min(d, segDist(p, q[k], q[(k + 1) % 4])) }
            sum += d; n += 1
        }
        let scale = sqrt(max(1, polygonArea(q)))
        return n == 0 ? .infinity : sum / Double(n) / scale
    }

    /// 四边形和掩码的重合度（IoU），掩码坐标系
    static func quadIoU(_ q: [CGPoint], mask: [Bool], w: Int, h: Int) -> Double {
        var inter = 0, poly = 0, m = 0
        for i in mask.indices where mask[i] { m += 1 }
        let x0 = max(0, Int(q.map(\.x).min()!)), x1 = min(w - 1, Int(q.map(\.x).max()!) + 1)
        let y0 = max(0, Int(q.map(\.y).min()!)), y1 = min(h - 1, Int(q.map(\.y).max()!) + 1)
        if x0 <= x1, y0 <= y1 {
            for y in y0...y1 {
                for x in x0...x1 {
                    let px = Double(x) + 0.5, py = Double(y) + 0.5
                    var c = false
                    for k in 0..<4 {
                        let a = q[k], b = q[(k + 1) % 4]
                        if (Double(a.y) > py) != (Double(b.y) > py), px < Double(b.x - a.x) * (py - Double(a.y)) / Double(b.y - a.y) + Double(a.x) { c.toggle() }
                    }
                    if c { poly += 1; if mask[y * w + x] { inter += 1 } }
                }
            }
        }
        return Double(inter) / Double(max(1, m + poly - inter))
    }

    /// 微调四角，让四边形和球台掩码的重合度（IoU）最高：每个角在 8 个方向上试着挪一步，变好就留下，步长逐渐减半。
    /// 用来修正「最大内接四边形」被球网、球员遮挡带偏的角
    static func refineQuad(_ q0: [CGPoint], mask: [Bool], w: Int, h: Int) -> [CGPoint] {
        var maskCount = 0, mx0 = w, mx1 = 0, my0 = h, my1 = 0
        for i in mask.indices where mask[i] { maskCount += 1; let x = i % w, y = i / w; mx0 = min(mx0, x); mx1 = max(mx1, x); my0 = min(my0, y); my1 = max(my1, y) }
        guard maskCount > 0 else { return q0 }
        func inside(_ px: Double, _ py: Double, _ q: [CGPoint]) -> Bool {
            var c = false
            for i in 0..<4 {
                let a = q[i], b = q[(i + 1) % 4]
                if (Double(a.y) > py) != (Double(b.y) > py), px < Double(b.x - a.x) * (py - Double(a.y)) / Double(b.y - a.y) + Double(a.x) { c.toggle() }
            }
            return c
        }
        func iou(_ q: [CGPoint]) -> Double {
            let x0 = max(0, min(mx0, Int(q.map(\.x).min()!) - 1)), x1 = min(w - 1, max(mx1, Int(q.map(\.x).max()!) + 1))
            let y0 = max(0, min(my0, Int(q.map(\.y).min()!) - 1)), y1 = min(h - 1, max(my1, Int(q.map(\.y).max()!) + 1))
            var inter = 0, poly = 0
            for y in y0...y1 {
                for x in x0...x1 where inside(Double(x) + 0.5, Double(y) + 0.5, q) {
                    poly += 1; if mask[y * w + x] { inter += 1 }
                }
            }
            return Double(inter) / Double(max(1, maskCount + poly - inter))
        }
        var q = q0, best = iou(q), step = 8.0
        let dirs: [(Double, Double)] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        while step >= 0.5 {
            var improved = true, rounds = 0
            while improved && rounds < 30 {
                improved = false; rounds += 1
                for c in 0..<4 {
                    for (dx, dy) in dirs {
                        var t = q; t[c] = CGPoint(x: Double(t[c].x) + dx * step, y: Double(t[c].y) + dy * step)
                        let v = iou(t)
                        if v > best + 1e-6 { q = t; best = v; improved = true }
                    }
                }
            }
            step /= 2
        }
        return q
    }

    /// 多边形面积（鞋带公式）
    static func polygonArea(_ p: [CGPoint]) -> Double {
        guard p.count >= 3 else { return 0 }
        var s = 0.0
        for i in p.indices {
            let a = p[i], b = p[(i + 1) % p.count]
            s += Double(a.x) * Double(b.y) - Double(b.x) * Double(a.y)
        }
        return abs(s) / 2
    }

    static func perimeter(_ poly: [CGPoint]) -> Double {
        zip(poly, poly.dropFirst() + [poly[0]]).reduce(0) { $0 + dist($1.0, $1.1) }
    }

    /// 四点排序为：远左、远右、近右、近左。按边长判断哪两条边是球台长边（2.74 米），
    /// 长边里平均 y 小的是远边。斜角机位（台面拍成菱形）时也不会把长宽弄反
    static func orderQuadByLength(_ q0: [CGPoint]) -> [CGPoint] {
        // 先按绕中心的角度排成一圈
        let cx = q0.map(\.x).reduce(0, +) / 4, cy = q0.map(\.y).reduce(0, +) / 4
        let q = q0.sorted { atan2(Double($0.y - cy), Double($0.x - cx)) < atan2(Double($1.y - cy), Double($1.x - cx)) }
        let e = (0..<4).map { dist(q[$0], q[($0 + 1) % 4]) }
        // 边 i 连 q[i] 和 q[i+1]；长边对是 (0,2) 还是 (1,3)
        let longA = e[0] + e[2] >= e[1] + e[3] ? 0 : 1
        let l1 = (q[longA], q[(longA + 1) % 4]), l2 = (q[(longA + 2) % 4], q[(longA + 3) % 4])
        let far = (l1.0.y + l1.1.y) < (l2.0.y + l2.1.y) ? l1 : l2
        let near = far == l1 ? l2 : l1
        // 远边两端里 x 小的是远左；近左是和远左同在一条短边上的那个点（短边连 far.0-near.1 和 far.1-near.0）
        let (fl, fr) = far.0.x <= far.1.x ? (far.0, far.1) : (far.1, far.0)
        let nl = fl == far.0 ? near.1 : near.0, nr = fl == far.0 ? near.0 : near.1
        return [fl, fr, nr, nl]
    }

    /// 四点排序为：远左、远右、近右、近左（图像 y 向下，远端 y 小）
    static func orderQuad(_ q: [CGPoint]) -> [CGPoint] {
        let s = q.sorted { $0.y < $1.y }
        let far = s.prefix(2).sorted { $0.x < $1.x }, near = s.suffix(2).sorted { $0.x < $1.x }
        return [far[0], far[1], near[1], near[0]]
    }

    /// 点到多边形边界的有符号距离：内部为正（对应 cv2.pointPolygonTest(measureDist=True)）
    static func signedDistance(_ p: CGPoint, to poly: [CGPoint]) -> Double {
        var inside = false
        var minD = Double.greatestFiniteMagnitude
        for i in poly.indices {
            let a = poly[i], b = poly[(i + 1) % poly.count]
            if (a.y > p.y) != (b.y > p.y), p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x { inside.toggle() }
            minD = min(minD, segDist(p, a, b))
        }
        return inside ? minD : -minD
    }
}
