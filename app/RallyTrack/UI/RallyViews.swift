import AVFoundation
import AVKit
import SwiftData
import SwiftUI

/// 播放一个回合（循环），可导出为单独视频分享
struct RallyPlayerView: View {
    let session: SessionRecord
    let rally: RallyInfo
    @State private var player: AVPlayer?
    @State private var exported: URL?
    @State private var exporting = false
    @State private var observer: Any?
    @State private var showPaywall = false
    @EnvironmentObject private var store: Store

    private var range: CMTimeRange {
        let fps = session.result?.fps ?? 120
        let s = CMTime(seconds: Double(rally.startFrame) / fps, preferredTimescale: 600)
        let e = CMTime(seconds: Double(rally.endFrame) / fps, preferredTimescale: 600)
        return CMTimeRange(start: s, end: e)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                VideoPlayer(player: player)
                    .overlay {
                        if let player, let r = session.result { TrailLive(rally: rally, player: player, fps: r.fps, video: CGSize(width: r.videoWidth, height: r.videoHeight)) }
                    }
                    .overlay(alignment: .bottomTrailing) { BrandBadge().padding(8).allowsHitTesting(false) }
                    .aspectRatio(16 / 9, contentMode: .fit)
                if rally.path == nil {
                    Label("Re-analyze this session to see the ball trail", systemImage: "info.circle")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                HStack {
                    StatTile(value: "\(rally.shots)", label: "Shots", highlight: true)
                    StatTile(value: rally.bounces.compactMap(\.speedKmh).max().map { "\(Int($0))" } ?? "–", label: "Top net speed (km/h)")
                }
                Group {
                    if let player { LiveLandingMap(rally: rally, player: player, fps: session.result?.fps ?? 120) }
                    else { TableHeatmap(points: rally.bounces.map { CGPoint(x: $0.tableX, y: $0.tableY) }, out: rally.bounces.map { $0.out == true }) }
                }
                .frame(height: 160)
                // 每一板要么落在台上（圆点），要么出界（红叉）；两样都确认不了的才算没看清
                let outCount = rally.bounces.filter { $0.out == true }.count
                if outCount > 0 {
                    Text("\(rally.shots) shots: \(rally.bounces.count - outCount) landed on the table, \(outCount) went out (red ✕)")
                        .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
                }
                if rally.bounces.count < rally.shots {
                    Text("\(rally.shots - rally.bounces.count) shots weren't clear enough to place")
                        .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
                }
                if let exported {
                    ShareLink(item: exported) { Label("Share clip", systemImage: "square.and.arrow.up").frame(maxWidth: .infinity) }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                } else {
                    Button { if store.canExport { Task { await export() } } else { showPaywall = true } } label: {
                        Label(exporting ? "Exporting…" : "Export clip", systemImage: store.canExport ? "film" : "lock.fill").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent).controlSize(.large).disabled(exporting)
                    if !store.isPro {
                        Button { showPaywall = true } label: {
                            Text(store.freeExportsLeft > 0 ? "\(store.freeExportsLeft) free exports left · Go Pro for unlimited" : "Free exports used up · Go Pro to keep exporting")
                                .font(.footnote).foregroundStyle(.secondary).multilineTextAlignment(.center)
                        }
                    }
                }
                Spacer()
            }
            .padding()
            .navigationTitle("Rally #\(rally.id)")
            .navigationBarTitleDisplayMode(.inline)
        }
        .onAppear(perform: setup)
        #if DEBUG
        .task {  // 调试：-autoExport YES 打开回合后自动做一次带水印导出，结果拷到 Documents/wm.mov
            guard UserDefaults.standard.bool(forKey: "autoExport") else { return }
            await export()
            if let exported {
                let dst = URL.documentsDirectory.appendingPathComponent("wm.mov")
                try? FileManager.default.removeItem(at: dst); try? FileManager.default.copyItem(at: exported, to: dst)
            }
        }
        #endif
        .sheet(isPresented: $showPaywall) { PaywallView() }
        .onDisappear { player?.pause(); if let observer { player?.removeTimeObserver(observer) } }
    }

    private func setup() {
        let p = AVPlayer(url: session.videoURL)
        let r = range
        p.seek(to: r.start, toleranceBefore: .zero, toleranceAfter: .zero)
        observer = p.addBoundaryTimeObserver(forTimes: [NSValue(time: r.end)], queue: .main) {
            p.seek(to: r.start, toleranceBefore: .zero, toleranceAfter: .zero)
        }
        p.play()
        player = p
    }

    private func export() async {
        exporting = true; defer { exporting = false }
        exported = try? await ClipExporter.export(url: session.videoURL, range: range, name: "Rally-\(rally.id)-\(rally.shots)shots",
                                                  watermark: String(localized: "RallyTrack"),  // 中文环境显示「球迹」
                                                  overlay: TrailOverlay(rally: rally, fps: session.result?.fps ?? 120))
        if exported != nil { store.didExport() }
    }
}

enum ClipExporter {
    /// 截取原视频片段。所有导出都带品牌水印（重新编码）；watermark 为 nil 时直接拷贝码流。
    static func export(url: URL, range: CMTimeRange, name: String, watermark: String? = nil, overlay: TrailOverlay? = nil) async throws -> URL {
        let out = URL.temporaryDirectory.appendingPathComponent(name + ".mov")
        try? FileManager.default.removeItem(at: out)
        let asset = AVURLAsset(url: url)
        let preset = watermark == nil ? AVAssetExportPresetPassthrough : AVAssetExportPresetHighestQuality
        guard let s = AVAssetExportSession(asset: asset, presetName: preset) else {
            throw AnalysisError.readerFailed("export")
        }
        s.timeRange = range
        if let watermark { s.videoComposition = try await composition(asset: asset, text: watermark, overlay: overlay) }
        try await s.export(to: out, as: .mov)
        return out
    }

    /// 用 Core Image 叠加水印（Core Animation 方案在模拟器上会崩）
    private static func composition(asset: AVAsset, text: String, overlay: TrailOverlay?) async throws -> AVVideoComposition {
        var badge: CIImage?
        return try await AVMutableVideoComposition.videoComposition(with: asset) { request in
            var frame = request.sourceImage
            if let overlay, let fx = overlay.image(at: request.compositionTime.seconds, size: frame.extent.size) {
                frame = fx.composited(over: frame)
            }
            if badge == nil { badge = watermarkImage(text: text, videoHeight: frame.extent.height) }
            guard let badge else { return request.finish(with: frame, context: nil) }
            // Core Image 原点在左下角：放到右下角
            let m = frame.extent.height * 0.04
            let placed = badge.transformed(by: .init(translationX: frame.extent.maxX - badge.extent.width - m, y: frame.extent.minY + m))
            request.finish(with: placed.composited(over: frame), context: nil)
        }
    }

    /// 品牌角标：App 图标 + 名字，半透明深色底，放在画面右下角
    private static func watermarkImage(text: String, videoHeight: CGFloat) -> CIImage? {
        let h = max(48, videoHeight * 0.085)            // 角标高度约为画面高度的 8.5%
        let pad = h * 0.16, icon = h - 2 * pad
        let title = NSAttributedString(string: text, attributes: [
            .font: UIFont.systemFont(ofSize: h * 0.40, weight: .heavy), .foregroundColor: UIColor.white,
        ])
        let sub = NSAttributedString(string: "AI 乒乓分析", attributes: [
            .font: UIFont.systemFont(ofSize: h * 0.22, weight: .semibold), .foregroundColor: UIColor(Theme.lime),
        ])
        let textW = max(title.size().width, sub.size().width)
        let size = CGSize(width: ceil(pad + icon + pad * 1.2 + textW + pad * 1.6), height: ceil(h))
        let fmt = UIGraphicsImageRendererFormat(); fmt.scale = 1; fmt.opaque = false
        let img = UIGraphicsImageRenderer(size: size, format: fmt).image { ctx in
            UIColor.black.withAlphaComponent(0.55).setFill()
            UIBezierPath(roundedRect: CGRect(origin: .zero, size: size), cornerRadius: h * 0.28).fill()
            if let logo = UIImage(named: "Logo") {
                let r = CGRect(x: pad, y: pad, width: icon, height: icon)
                ctx.cgContext.saveGState()
                UIBezierPath(roundedRect: r, cornerRadius: icon * 0.22).addClip()
                logo.draw(in: r)
                ctx.cgContext.restoreGState()
            }
            let tx = pad + icon + pad * 1.2
            let th = title.size().height, sh = sub.size().height, gap = h * 0.02
            let y0 = (h - th - sh - gap) / 2
            title.draw(at: CGPoint(x: tx, y: y0))
            sub.draw(at: CGPoint(x: tx, y: y0 + th + gap))
        }
        return img.cgImage.map { CIImage(cgImage: $0) }
    }
}

/// 所有场次的回合，按板数排序
struct ClipsView: View {
    @Query(sort: \SessionRecord.date, order: .reverse) private var sessions: [SessionRecord]
    @State private var selected: (SessionRecord, RallyInfo)?
    @State private var minShots = 0

    private var all: [(SessionRecord, RallyInfo)] {
        sessions.flatMap { s in (s.result?.rallies ?? []).map { (s, $0) } }
            .filter { $0.1.shots >= minShots }
            .sorted { $0.1.shots > $1.1.shots }
    }

    var body: some View {
        NavigationStack {
            List {
                Picker("Filter", selection: $minShots) {
                    Text("All").tag(0); Text("5+ shots").tag(5); Text("10+ shots").tag(10)
                }
                .pickerStyle(.segmented)
                .listRowBackground(Color.clear)
                if all.isEmpty { Text("No rallies yet").foregroundStyle(.secondary) }
                // 回合编号在每场内从 1 开始，跨场会重复，key 要带上场次 ID
                ForEach(all.map { (key: "\($0.0.id)-\($0.1.id)", pair: $0) }, id: \.key) { row in
                    let pair = row.pair
                    Button { selected = pair } label: {
                        HStack {
                            Text("\(pair.1.shots)").font(.title2.bold().monospacedDigit()).foregroundStyle(Theme.lime).frame(width: 44)
                            VStack(alignment: .leading) {
                                Text("shots").font(.caption).foregroundStyle(.secondary)
                                Text(pair.0.date, format: .dateTime.month().day().hour().minute()).font(.subheadline)
                            }
                            Spacer()
                            Image(systemName: "play.circle.fill").font(.title2).foregroundStyle(Theme.lime)
                        }
                    }
                }
            }
            .navigationTitle("Rallies")
            .sheet(isPresented: Binding(get: { selected != nil }, set: { if !$0 { selected = nil } })) {
                if let (s, r) = selected { RallyPlayerView(session: s, rally: r) }
            }
        }
    }
}

/// 导出视频上的球轨迹特效：发光的彗星尾、落台光圈、左上角实时板数
struct TrailOverlay {
    let rally: RallyInfo
    let fps: Double

    /// 原视频时间（秒）这一帧的特效层；没有轨迹数据返回 nil
    func image(at t: Double, size: CGSize) -> CIImage? {
        guard let path = rally.path, !path.isEmpty else { return nil }
        let f = Int((t * fps).rounded())
        let i = f - rally.startFrame
        let tail = max(6, Int(0.3 * fps))              // 尾巴长 0.3 秒
        let lo = max(0, i - tail), hi = min(path.count - 1, i)
        let pts: [(Int, CGPoint)] = hi >= lo ? (lo...hi).compactMap { j in path[j].map { (j, CGPoint(x: $0[0], y: $0[1])) } } : []
        let flashes = rally.bounces.filter { f >= $0.frame && Double(f - $0.frame) < 0.45 * fps }
        let shotsSoFar = (rally.crossings ?? []).filter { $0 <= f }.count
        let fmt = UIGraphicsImageRendererFormat(); fmt.scale = 1; fmt.opaque = false
        let unit = size.height / 1080
        let img = UIGraphicsImageRenderer(size: size, format: fmt).image { ctx in
            let g = ctx.cgContext
            g.setLineCap(.round); g.setLineJoin(.round)
            // 尾巴：越靠近球越粗越亮，外圈发光
            for k in 1..<max(1, pts.count) {
                let (ja, a) = pts[k - 1], (jb, b) = pts[k]
                guard jb - ja <= 3 else { continue }          // 中间丢了几帧就断开，别连出假线
                let s = CGFloat(k) / CGFloat(pts.count)
                g.setShadow(offset: .zero, blur: 14 * unit, color: UIColor(Theme.lime).withAlphaComponent(0.9 * s).cgColor)
                g.setStrokeColor(UIColor(Theme.cyan).withAlphaComponent(0.25 + 0.6 * s).cgColor)
                g.setLineWidth((2 + 9 * s) * unit)
                g.move(to: a); g.addLine(to: b); g.strokePath()
            }
            // 球：白芯 + 绿色光晕
            if let (j, p) = pts.last, j == i {
                g.setShadow(offset: .zero, blur: 22 * unit, color: UIColor(Theme.lime).cgColor)
                g.setFillColor(UIColor.white.cgColor)
                let r = 8 * unit
                g.fillEllipse(in: CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r))
            }
            // 落台：向外扩散的扁圆光圈（台面透视下是椭圆）
            g.setShadow(offset: .zero, blur: 10 * unit, color: UIColor(Theme.lime).cgColor)
            for b in flashes {
                guard let pp = path[safe: b.frame - rally.startFrame] ?? nil else { continue }
                let age = Double(f - b.frame) / (0.45 * fps)
                let r = (14 + 46 * age) * unit
                g.setStrokeColor(UIColor(Theme.lime).withAlphaComponent(1 - age).cgColor)
                g.setLineWidth(4 * unit)
                g.strokeEllipse(in: CGRect(x: pp[0] - r, y: pp[1] - r * 0.35, width: 2 * r, height: 0.7 * r))
            }
            // 左上角板数
            g.setShadow(offset: .zero, blur: 0, color: nil)
            let label = NSAttributedString(string: String(localized: "Shot \(shotsSoFar)"), attributes: [
                .font: UIFont.monospacedDigitSystemFont(ofSize: 44 * unit, weight: .heavy), .foregroundColor: UIColor.white,
            ])
            let ls = label.size(), pad = 14 * unit
            let box = CGRect(x: 30 * unit, y: 30 * unit, width: ls.width + 2 * pad, height: ls.height + pad)
            UIColor.black.withAlphaComponent(0.5).setFill()
            UIBezierPath(roundedRect: box, cornerRadius: box.height / 2).fill()
            label.draw(at: CGPoint(x: box.minX + pad, y: box.minY + pad / 2))
        }
        // CIImage(cgImage:) 会保持图片正向（和角标一样），不需要再翻转
        return img.cgImage.map { CIImage(cgImage: $0) }
    }
}

extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}

/// 回放时实时画球轨迹（和导出视频同样的彗星尾和落台光圈）
struct TrailLive: View {
    let rally: RallyInfo
    let player: AVPlayer
    let fps: Double
    let video: CGSize

    var body: some View {
        TimelineView(.animation) { _ in
            let now = player.currentTime().seconds   // 每一帧读一次播放时间，Canvas 随之重画
            Canvas { ctx, size in
                guard let path = rally.path, video.width > 0 else { return }
                // VideoPlayer 按比例缩放居中：算出画面在视图里的位置
                let k = min(size.width / video.width, size.height / video.height)
                let ox = (size.width - video.width * k) / 2, oy = (size.height - video.height * k) / 2
                func map(_ p: [Double]) -> CGPoint { CGPoint(x: ox + p[0] * k, y: oy + p[1] * k) }
                let f = Int((now * fps).rounded()), i = f - rally.startFrame
                let tail = max(6, Int(0.3 * fps)), lo = max(0, i - tail), hi = min(path.count - 1, i)
                guard hi >= lo else { return }
                var glow = ctx; glow.addFilter(.shadow(color: Theme.lime.opacity(0.9), radius: 5))
                var prev: (Int, CGPoint)?
                for j in lo...hi {
                    guard let p = path[j] else { continue }
                    let q = map(p)
                    if let (pj, pq) = prev, j - pj <= 3 {
                        let t = CGFloat(j - lo) / CGFloat(max(1, hi - lo))
                        var line = Path(); line.move(to: pq); line.addLine(to: q)
                        glow.stroke(line, with: .color(Theme.cyan.opacity(0.25 + 0.6 * t)), style: StrokeStyle(lineWidth: 1 + 3.5 * t, lineCap: .round))
                    }
                    prev = (j, q)
                }
                let shot = (rally.crossings ?? []).filter { $0 <= f }.count
                ctx.draw(Text(String(localized: "Shot \(shot)")).font(.system(size: 15, weight: .heavy).monospacedDigit()).foregroundColor(.white),
                         at: CGPoint(x: ox + 12, y: oy + 12), anchor: .topLeading)
                if let (j, q) = prev, j == i {
                    glow.fill(Path(ellipseIn: CGRect(x: q.x - 3.5, y: q.y - 3.5, width: 7, height: 7)), with: .color(.white))
                }
                for b in rally.bounces where f >= b.frame && Double(f - b.frame) < 0.45 * fps {
                    guard let raw = path[safe: b.frame - rally.startFrame] ?? nil else { continue }
                    let age = Double(f - b.frame) / (0.45 * fps), q = map(raw), r = 5 + 20 * age
                    glow.stroke(Path(ellipseIn: CGRect(x: q.x - r, y: q.y - r * 0.35, width: 2 * r, height: 0.7 * r)), with: .color(Theme.lime.opacity(1 - age)), lineWidth: 2)
                }
            }
        }
        .allowsHitTesting(false)
    }
}

/// App 图标 + 名字的小角标（回放画面右下角；导出视频里是同样的样式）
struct BrandBadge: View {
    var body: some View {
        HStack(spacing: 6) {
            Image("Logo").resizable().frame(width: 22, height: 22).clipShape(RoundedRectangle(cornerRadius: 5))
            VStack(alignment: .leading, spacing: 0) {
                Text("RallyTrack").font(.system(size: 12, weight: .heavy)).foregroundStyle(.white)
                Text("AI 乒乓分析").font(.system(size: 8, weight: .semibold)).foregroundStyle(Theme.lime)
            }
        }
        .padding(.horizontal, 7).padding(.vertical, 4)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
    }
}

/// 回放同步的落点图：球落一次台，小球台上就亮一个点（带编号和扩散光圈）；回合循环播放时重新开始
struct LiveLandingMap: View {
    let rally: RallyInfo
    let player: AVPlayer
    let fps: Double

    var body: some View {
        TimelineView(.animation) { _ in
            let now = player.currentTime().seconds
            Canvas { ctx, size in
                let k = min(size.width / Table.length, size.height / Table.width)
                let w = Table.length * k, h = Table.width * k
                let o = CGPoint(x: (size.width - w) / 2, y: (size.height - h) / 2)
                let rect = CGRect(origin: o, size: CGSize(width: w, height: h))
                ctx.fill(Path(roundedRect: rect, cornerRadius: 4), with: .color(Theme.tableBlue))
                ctx.stroke(Path(roundedRect: rect, cornerRadius: 4), with: .color(.white), lineWidth: 2)
                var mid = Path(); mid.move(to: CGPoint(x: o.x, y: o.y + h / 2)); mid.addLine(to: CGPoint(x: o.x + w, y: o.y + h / 2))
                ctx.stroke(mid, with: .color(.white.opacity(0.5)), lineWidth: 1)
                var net = Path(); net.move(to: CGPoint(x: o.x + w / 2, y: o.y - 6)); net.addLine(to: CGPoint(x: o.x + w / 2, y: o.y + h + 6))
                ctx.stroke(net, with: .color(.white), lineWidth: 3)

                let f = Int((now * fps).rounded())
                let shown = rally.bounces.enumerated().filter { $0.element.frame <= f }
                var prev: CGPoint?
                for (n, b) in shown {
                    let isOut = b.out == true
                    let px = min(max(b.tableX, isOut ? -0.12 : 0), Table.length + (isOut ? 0.12 : 0))
                    let py = min(max(b.tableY, isOut ? -0.12 : 0), Table.width + (isOut ? 0.12 : 0))
                    let c = CGPoint(x: o.x + px * k, y: o.y + py * k)
                    // 连线：上一板落点 -> 这一板落点，看出球的来回
                    if let p = prev {
                        var l = Path(); l.move(to: p); l.addLine(to: c)
                        ctx.stroke(l, with: .color(.white.opacity(0.25)), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
                    }
                    prev = c
                    let age = Double(f - b.frame) / fps
                    if age < 0.6 {  // 刚落下：扩散光圈
                        let r = 6 + 22 * age / 0.6
                        ctx.stroke(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)),
                                   with: .color((isOut ? Color.red : Theme.lime).opacity(1 - age / 0.6)), lineWidth: 2.5)
                    }
                    let latest = n == shown.last?.offset
                    if isOut {
                        let d: CGFloat = 5
                        var x = Path(); x.move(to: CGPoint(x: c.x - d, y: c.y - d)); x.addLine(to: CGPoint(x: c.x + d, y: c.y + d))
                        x.move(to: CGPoint(x: c.x + d, y: c.y - d)); x.addLine(to: CGPoint(x: c.x - d, y: c.y + d))
                        ctx.stroke(x, with: .color(.red), lineWidth: 2.5)
                    } else {
                        let r: CGFloat = latest ? 7 : 5
                        ctx.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)), with: .color(latest ? Theme.lime : .white))
                    }
                    ctx.draw(Text("\(n + 1)").font(.system(size: 10, weight: .bold)).foregroundColor(.white),
                             at: CGPoint(x: c.x, y: c.y - 12))
                }
            }
        }
    }
}
