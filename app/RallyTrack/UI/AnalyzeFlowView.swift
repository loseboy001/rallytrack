import AVFoundation
import SwiftData
import SwiftUI

/// 在首帧上显示四个可拖动的角点（源视频像素坐标）。
/// 拖动是「相对移动」（手指不用压在角上，不会挡住），拖的时候画面角落有放大镜，带十字准星
struct CalibrationView: View {
    let image: UIImage?
    let videoSize: CGSize
    var k1: Double = 0
    @Binding var quad: [CGPoint]          // 拉直后的坐标；界面上显示的是畸变回画面后的位置
    let onConfirm: () -> Void
    @State private var active: Int?
    @State private var dragStart: CGPoint?

    var body: some View {
        VStack(spacing: 12) {
            Text("Drag the four corners onto the table top").font(.headline)
            Text("Pink line should sit on the net").font(.caption).foregroundStyle(.secondary)
            GeometryReader { geo in
                let scale = min(geo.size.width / max(1, videoSize.width), geo.size.height / max(1, videoSize.height))
                let size = CGSize(width: videoSize.width * scale, height: videoSize.height * scale)
                let lens = Lens(k1: k1, w: Double(videoSize.width), h: Double(videoSize.height))
                let shown = quad.map(lens.distort)
                ZStack(alignment: .topLeading) {
                    if let image { Image(uiImage: image).resizable().frame(width: size.width, height: size.height) }
                    else { Rectangle().fill(.gray.opacity(0.3)).frame(width: size.width, height: size.height) }
                    if quad.count == 4 {  // 拖角时球网位置实时跟着变，对准真实球网就说明框准了
                        TableGuides(quad: quad, lens: lens, scale: scale, outline: true).frame(width: size.width, height: size.height)
                    }
                    ForEach(quad.indices, id: \.self) { i in
                        ZStack {
                            Circle().fill(Theme.lime.opacity(active == i ? 0.15 : 0.3)).overlay(Circle().stroke(Theme.lime, lineWidth: 1.5))
                            Circle().fill(Theme.lime).frame(width: 4, height: 4)
                        }
                        .frame(width: 30, height: 30)
                        .contentShape(Circle().inset(by: -14))  // 触摸区域比看到的圈大一圈
                        .position(x: shown[i].x * scale, y: shown[i].y * scale)
                        .gesture(DragGesture(minimumDistance: 0, coordinateSpace: .named("calibration"))
                            .onChanged { v in
                                if active != i { active = i; dragStart = shown[i] }
                                guard let s0 = dragStart else { return }
                                // 相对移动，并放慢一半：手指移动 2 点，角移动 1 点，方便微调；拖的是画面里看到的位置，存的是拉直后的坐标
                                let p = CGPoint(x: min(max(0, s0.x + v.translation.width / scale * 0.5), videoSize.width),
                                                y: min(max(0, s0.y + v.translation.height / scale * 0.5), videoSize.height))
                                quad[i] = lens.undistort(p)
                            }
                            .onEnded { _ in active = nil; dragStart = nil })
                    }
                    if let a = active, let image, quad.indices.contains(a) {
                        Loupe(image: image, center: shown[a], videoSize: videoSize, k: scale * 4)
                            .position(x: shown[a].x * scale < size.width / 2 ? size.width - 70 : 70, y: 70)
                    }
                }
                .coordinateSpace(name: "calibration")
                .frame(width: size.width, height: size.height)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            Button(action: onConfirm) { Text("Analyze").frame(maxWidth: .infinity) }
                .buttonStyle(.borderedProminent).controlSize(.large).padding(.horizontal)
        }
        .padding(.vertical)
    }
}

/// 放大镜：以 center（视频像素）为中心放大显示，十字准星对准当前角
struct Loupe: View {
    let image: UIImage
    let center: CGPoint
    let videoSize: CGSize
    let k: CGFloat              // 放大后每个视频像素占多少点（画面缩放 × 放大倍数）
    private let d: CGFloat = 120

    var body: some View {
        let iw = videoSize.width * k, ih = videoSize.height * k
        Image(uiImage: image).resizable().frame(width: iw, height: ih)
            .offset(x: d / 2 - center.x * k, y: d / 2 - center.y * k)
            .frame(width: d, height: d, alignment: .topLeading)
            .clipShape(Circle())
            .overlay {
                ZStack {
                    Rectangle().fill(Theme.lime).frame(width: 1, height: d * 0.4)
                    Rectangle().fill(Theme.lime).frame(width: d * 0.4, height: 1)
                }
            }
            .overlay(Circle().stroke(.white, lineWidth: 2))
            .shadow(radius: 6)
            .allowsHitTesting(false)
    }
}
