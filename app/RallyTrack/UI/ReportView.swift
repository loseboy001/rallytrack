import Charts
import SwiftUI

struct ReportView: View {
    let session: SessionRecord
    @State private var selected: RallyInfo?
    @State private var fixing = false
    @EnvironmentObject private var queue: AnalysisQueue
    @Environment(\.dismiss) private var dismiss

    /// 用新版算法（或手动修正的球台）重新分析，完成后原地更新这条记录
    private func reanalyze(_ quad: [CGPoint]?) {
        queue.add(file: session.videoFile, quad: quad ?? session.result?.quad, k1: session.result?.k1, replaces: session.id)
        dismiss()
    }

    var body: some View {
        ScrollView {
            if let r = session.result {
                VStack(alignment: .leading, spacing: 16) {
                    Text(session.date, format: .dateTime.year().month().day().hour().minute())
                        .font(.subheadline).foregroundStyle(.secondary)
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                        StatTile(value: "\(r.rallies.count)", label: "Rallies")
                        StatTile(value: "\(r.totalShots)", label: "Total shots")
                        StatTile(value: "\(r.longestRally)", label: "Longest rally", highlight: true)
                        StatTile(value: r.speeds.max().map { "\(Int($0))" } ?? "–", label: "Top net speed (km/h)")
                    }
                    Card(title: "Landing map") {
                        TableHeatmap(points: r.landings.map { CGPoint(x: $0.tableX, y: $0.tableY) })
                            .frame(height: 190)
                        LengthMix(landings: r.landings)
                    }
                    if !r.rallies.isEmpty {
                        Card(title: "Shots per rally") {
                            Chart(r.rallies) { rally in
                                BarMark(x: .value("Rally", rally.id), y: .value("Shots", rally.shots))
                                    .foregroundStyle(rally.shots == r.longestRally ? Theme.lime : Color.gray)
                            }
                            .frame(height: 140)
                        }
                        Card(title: "Rallies") {
                            ForEach(r.rallies.sorted { $0.shots > $1.shots }) { rally in
                                Button { selected = rally } label: {
                                    HStack {
                                        Text("#\(rally.id)").foregroundStyle(.secondary).monospacedDigit()
                                        Text("\(rally.shots) shots").bold()
                                        Spacer()
                                        Text(Duration.seconds(r.time(of: rally.startFrame)), format: .time(pattern: .minuteSecond))
                                            .foregroundStyle(.secondary).monospacedDigit()
                                        Image(systemName: "play.circle.fill").foregroundStyle(Theme.lime)
                                    }
                                }
                                .padding(.vertical, 4)
                            }
                        }
                    } else {
                        Text("No rallies found. Make sure the whole table is in view and the camera is still.")
                            .foregroundStyle(.secondary)
                    }
                    Text("Ball found in \(Int(r.detectedRatio * 100))% of frames · \(Int(r.fps)) fps")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding()
            }
        }
        .navigationTitle("Session report")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $selected) { rally in RallyPlayerView(session: session, rally: rally) }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button { reanalyze(nil) } label: { Label("Re-analyze", systemImage: "arrow.clockwise") }
                    Button { fixing = true } label: { Label("Adjust table", systemImage: "skew") }
                } label: { Image(systemName: "ellipsis.circle") }
            }
        }
        .fullScreenCover(isPresented: $fixing) {
            TableFixView(videoURL: session.videoURL, initial: session.result?.quad, k1: session.result?.k1 ?? 0) { reanalyze($0) }
        }
    }
}

struct StatTile: View {
    let value: String
    let label: LocalizedStringKey
    var highlight = false
    var body: some View {
        VStack(spacing: 4) {
            Text(value).font(.system(size: 34, weight: .heavy, design: .rounded)).monospacedDigit()
                .foregroundStyle(highlight ? Theme.lime : .primary)
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity).padding(.vertical, 14)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 16))
    }
}

struct Card<Content: View>: View {
    let title: LocalizedStringKey
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title).font(.headline)
            content
        }
        .padding().frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 16))
    }
}

/// 俯视球台 + 落点热区（坐标为米）
struct TableHeatmap: View {
    let points: [CGPoint]
    var out: [Bool] = []   // 和 points 一一对应；true 画成红叉（出界/下网）
    var body: some View {
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
            for (n, p) in points.enumerated() {
                if out.indices.contains(n), out[n] {
                    // 出界：夹到台边外一点，画红叉
                    let px = min(max(p.x, -0.12), Table.length + 0.12), py = min(max(p.y, -0.12), Table.width + 0.12)
                    let c = CGPoint(x: o.x + px * k, y: o.y + py * k), d: CGFloat = 5
                    var x = Path(); x.move(to: CGPoint(x: c.x - d, y: c.y - d)); x.addLine(to: CGPoint(x: c.x + d, y: c.y + d))
                    x.move(to: CGPoint(x: c.x + d, y: c.y - d)); x.addLine(to: CGPoint(x: c.x - d, y: c.y + d))
                    ctx.stroke(x, with: .color(.red), lineWidth: 2.5)
                    continue
                }
                let px = min(max(p.x, 0), Table.length), py = min(max(p.y, 0), Table.width)
                let c = CGPoint(x: o.x + px * k, y: o.y + py * k)
                let r = w * 0.07
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)),
                         with: .radialGradient(Gradient(colors: [Theme.lime.opacity(0.45), .clear]), center: c, startRadius: 0, endRadius: r))
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - 3, y: c.y - 3, width: 6, height: 6)), with: .color(.white))
            }
        }
    }
}

/// 短球 / 半出台 / 长球：落点离球网 <0.45m 为短球，>0.9m 为长球
struct LengthMix: View {
    let landings: [BounceInfo]
    var body: some View {
        let d = landings.map { abs($0.tableX - Table.length / 2) }
        let n = Double(max(1, d.count))
        let short = Double(d.filter { $0 < 0.45 }.count) / n, long = Double(d.filter { $0 > 0.9 }.count) / n
        let half = d.isEmpty ? 0 : max(0, 1 - short - long)  // 没有落点时三项都为 0
        HStack(spacing: 10) {
            mix("Short", short); mix("Half-long", half); mix("Long", long)
        }
    }

    private func mix(_ label: LocalizedStringKey, _ v: Double) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack { Text(label); Spacer(); Text("\(Int(v * 100))%").bold() }.font(.caption)
            ProgressView(value: v).tint(Theme.lime)
        }
    }
}
