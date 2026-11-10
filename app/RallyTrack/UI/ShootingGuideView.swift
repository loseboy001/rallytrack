import SwiftUI

/// 拍摄指引：录制 / 导入前给一张正确机位的参考图
struct ShootingGuideView: View {
    var actionTitle: LocalizedStringKey
    var onContinue: () -> Void
    @AppStorage("hideShootingGuide") private var hide = false
    @Environment(\.dismiss) private var dismiss

    // 参考图里球台四角（相对坐标：远左、远右、近右、近左）
    private let quad: [CGPoint] = [.init(x: 0.301, y: 0.572), .init(x: 0.719, y: 0.590), .init(x: 0.802, y: 0.685), .init(x: 0.238, y: 0.678)]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text("Reference shot").font(.headline)
                    Image("ShootingRef").resizable().aspectRatio(16 / 9, contentMode: .fit)
                        .overlay {
                            GeometryReader { g in
                                Path { p in
                                    p.addLines(quad.map { CGPoint(x: $0.x * g.size.width, y: $0.y * g.size.height) })
                                    p.closeSubpath()
                                }
                                .stroke(Theme.lime, style: StrokeStyle(lineWidth: 3, lineJoin: .round))
                                .shadow(color: Theme.lime.opacity(0.8), radius: 6)
                            }
                        }
                        .overlay(alignment: .topLeading) {
                            Label("Whole table in view", systemImage: "checkmark.circle.fill")
                                .font(.caption.weight(.bold)).foregroundStyle(.black)
                                .padding(.horizontal, 10).padding(.vertical, 5)
                                .background(Theme.lime, in: Capsule()).padding(10)
                        }
                        .clipShape(RoundedRectangle(cornerRadius: 14))

                    VStack(alignment: .leading, spacing: 14) {
                        tip("iphone.landscape", "Landscape, at the side of the table", "Level with the net, 2–4 m away. A tripod helps.")
                        tip("camera.metering.center.weighted", "Use the 1× lens, not 0.5×", "The ultra-wide lens bends the table edges, so landing spots and shot counts come out wrong.")
                        tip("rectangle.dashed", "All four table corners visible", "Keep both players' ends in frame. Nothing blocking the table.")
                        tip("speedometer", "Slow-mo 120 fps if you can", "In the Camera app choose Slo-Mo, or Settings › Camera › 120 fps.")
                        tip("lightbulb", "Bright, even light", "Avoid strong backlight from windows behind the table.")
                        tip("film", "Record full points", "Start before the serve and keep recording — rallies are cut automatically.")
                    }

                    Toggle("Don't show again", isOn: $hide).font(.subheadline).tint(Theme.lime)
                }
                .padding()
            }
            .safeAreaInset(edge: .bottom) {
                Button { dismiss(); onContinue() } label: {
                    Text(actionTitle).font(.headline).frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent).controlSize(.large).padding()
                .background(.bar)
            }
            .navigationTitle("How to film")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } } }
        }
    }

    private func tip(_ icon: String, _ title: LocalizedStringKey, _ detail: LocalizedStringKey) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon).font(.title3).foregroundStyle(Theme.lime).frame(width: 28)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.subheadline.weight(.semibold))
                Text(detail).font(.footnote).foregroundStyle(.secondary)
            }
        }
    }
}
