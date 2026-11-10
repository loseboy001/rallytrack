import AVFoundation
import SwiftUI

/// 首页上的分析任务：缩略图 + 状态。找到球台后先让用户看一眼框得对不对，再开始分析
struct JobRow: View {
    let job: AnalysisQueue.Job
    @EnvironmentObject private var queue: AnalysisQueue
    @State private var thumb: UIImage?
    @State private var videoSize: CGSize = .zero
    @State private var fixing = false

    private var url: URL { SessionRecord.videosDir.appendingPathComponent(job.file) }

    var body: some View {
        Group {
            if job.state == .confirm { confirmCard } else { compactRow }
        }
        .task {
            let gen = AVAssetImageGenerator(asset: AVURLAsset(url: url))
            gen.appliesPreferredTrackTransform = true
            gen.maximumSize = CGSize(width: 640, height: 640)
            if let img = try? await gen.image(at: CMTime(seconds: 1, preferredTimescale: 600)).image { thumb = UIImage(cgImage: img) }
            if let info = try? await VideoAnalyzer.info(of: url) { videoSize = info.size }
        }
        .fullScreenCover(isPresented: $fixing) {
            TableFixView(videoURL: url, initial: job.quad, k1: job.k1 ?? 0) { queue.setTable(job, quad: $0) }
        }
    }

    private var thumbView: some View {
        Group {
            if let thumb { Image(uiImage: thumb).resizable().scaledToFill() } else { Color.white.opacity(0.08) }
        }
    }

    /// 确认球台：大图上画出自动找到的四角
    private var confirmCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Is the table outlined correctly?").font(.subheadline.weight(.semibold))
            Text("Green frame on the 4 table corners, pink line on the net").font(.caption).foregroundStyle(.secondary)
            thumbView
                .aspectRatio(videoSize.width > 0 ? videoSize.width / videoSize.height : 16 / 9, contentMode: .fit)
                .overlay {
                    GeometryReader { g in
                        if let q = job.quad, videoSize.width > 0 {
                            let sx = g.size.width / videoSize.width
                            TableGuides(quad: q, lens: Lens(k1: job.k1 ?? 0, w: Double(videoSize.width), h: Double(videoSize.height)), scale: sx, outline: true)
                        }
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .onTapGesture { fixing = true }
            HStack(spacing: 10) {
                Button { queue.confirm(job) } label: {
                    Label("Looks right, analyze", systemImage: "checkmark").lineLimit(1).minimumScaleFactor(0.8).frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                Button { fixing = true } label: {
                    Label("Adjust", systemImage: "hand.draw").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
            .controlSize(.regular)
        }
        .padding(.vertical, 4)
    }

    private var compactRow: some View {
        HStack(spacing: 12) {
            thumbView.frame(width: 72, height: 44).clipShape(RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 5) {
                switch job.state {
                case .running:
                    HStack {
                        Text(queue.stage.isEmpty ? String(localized: "Starting…") : queue.stage).font(.subheadline)
                        Spacer()
                        Text("\(Int(queue.progress * 100))%").font(.subheadline.monospacedDigit().bold()).foregroundStyle(Theme.lime)
                    }
                    ProgressView(value: queue.progress).tint(Theme.lime)
                    if let r = queue.remaining {
                        Text("About \(r / 60) min \(r % 60) s left").font(.caption2).foregroundStyle(.secondary)
                    }
                case .detect, .detecting:
                    Text("Finding the table…").font(.subheadline).foregroundStyle(.secondary)
                case .queued, .confirm:
                    Text("Waiting…").font(.subheadline).foregroundStyle(.secondary)
                case .needsTable:
                    Text("Couldn't find the table").font(.subheadline)
                    Button("Mark table corners") { fixing = true }.font(.footnote.weight(.semibold)).foregroundStyle(Theme.lime)
                case .failed:
                    Text("Analysis failed").font(.subheadline)
                    if let m = job.message { Text(m).font(.caption).foregroundStyle(.secondary).lineLimit(2) }
                    Button("Try Again") { queue.retry(job) }.font(.footnote.weight(.semibold)).foregroundStyle(Theme.lime)
                }
            }
            .buttonStyle(.plain)
        }
    }
}

/// 手动标球台四角：自动识别失败，或者用户想修正时用
struct TableFixView: View {
    let videoURL: URL
    var initial: [CGPoint]?
    var k1: Double = 0
    var onDone: ([CGPoint]) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var frame: UIImage?
    @State private var size: CGSize = .zero
    @State private var quad: [CGPoint] = []

    var body: some View {
        NavigationStack {
            Group {
                if quad.isEmpty { ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity) }
                else { CalibrationView(image: frame, videoSize: size, k1: k1, quad: $quad) { dismiss(); onDone(quad) } }
            }
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } } }
        }
        .task {
            guard let info = try? await VideoAnalyzer.info(of: videoURL) else { dismiss(); return }
            let gen = AVAssetImageGenerator(asset: AVURLAsset(url: videoURL))
            gen.appliesPreferredTrackTransform = true
            if let img = try? await gen.image(at: CMTime(seconds: min(1, info.duration / 2), preferredTimescale: 600)).image { frame = UIImage(cgImage: img) }
            size = info.size
            let w = info.size.width, h = info.size.height
            quad = initial ?? [CGPoint(x: w * 0.3, y: h * 0.45), CGPoint(x: w * 0.7, y: h * 0.45), CGPoint(x: w * 0.8, y: h * 0.62), CGPoint(x: w * 0.2, y: h * 0.62)]
        }
    }
}

/// 球台框（绿色，拉直坐标里的直线，按镜头畸变画回画面里会是弧线）、按透视算出的球网（粉色）和中线（虚线）：
/// 框准了，粉线应该正好压在真实球网上
struct TableGuides: View {
    let quad: [CGPoint]   // 拉直后的视频像素：远左、远右、近右、近左
    var lens: Lens
    var scale: CGFloat    // 视频像素 -> 视图点
    var outline = false

    var body: some View {
        Canvas { ctx, _ in
            guard quad.count == 4, let h = Homography(from: [CGPoint(x: 0, y: 0), CGPoint(x: Table.length, y: 0),
                                                            CGPoint(x: Table.length, y: Table.width), CGPoint(x: 0, y: Table.width)], to: quad) else { return }
            func path(_ a: CGPoint, _ b: CGPoint) -> Path {
                var p = Path(); p.addLines(lens.curve(a, b).map { CGPoint(x: $0.x * scale, y: $0.y * scale) }); return p
            }
            if outline {
                for k in 0..<4 {
                    ctx.stroke(path(quad[k], quad[(k + 1) % 4]), with: .color(Theme.lime), style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                }
            }
            ctx.stroke(path(h.apply(CGPoint(x: Table.length / 2, y: -0.1)), h.apply(CGPoint(x: Table.length / 2, y: Table.width + 0.1))), with: .color(.pink), lineWidth: 3)
            ctx.stroke(path(h.apply(CGPoint(x: 0, y: Table.width / 2)), h.apply(CGPoint(x: Table.length, y: Table.width / 2))),
                       with: .color(.white.opacity(0.7)), style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
        }
        .allowsHitTesting(false)
    }
}
