import AVFoundation
import SwiftUI
import UIKit

/// 横屏录制：优先 1080p/120fps（模型按 120fps 训练），没有则取最高帧率
struct RecorderView: View {
    let onFinish: (URL?) -> Void
    @StateObject private var cam = Camera()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if cam.authorized {
                CameraPreview(session: cam.session).ignoresSafeArea()
                // 摆位参考：球台应落在这个框附近
                GeometryReader { g in
                    Path { p in
                        let w = g.size.width, h = g.size.height
                        p.addLines([CGPoint(x: w * 0.28, y: h * 0.45), CGPoint(x: w * 0.72, y: h * 0.45),
                                    CGPoint(x: w * 0.82, y: h * 0.63), CGPoint(x: w * 0.18, y: h * 0.63)])
                        p.closeSubpath()
                    }
                    .stroke(Theme.lime.opacity(cam.recording ? 0 : 0.8), style: StrokeStyle(lineWidth: 2, dash: [8, 6]))
                }
                .ignoresSafeArea()
            } else {
                VStack(spacing: 16) {
                    Image(systemName: "camera.fill").font(.largeTitle)
                    Text("Camera access is needed to record.")
                        .multilineTextAlignment(.center)
                    if cam.permissionDenied {
                        Button("Open Settings") {
                            if let settingsURL = URL(string: UIApplication.openSettingsURLString) {
                                UIApplication.shared.open(settingsURL)
                            }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }
                .foregroundStyle(.white)
                .padding()
            }
            VStack {
                HStack {
                    Button { cam.stop(); onFinish(nil) } label: { Image(systemName: "xmark").font(.title2).padding() }
                    Spacer()
                    Text(cam.formatLabel).font(.caption.monospaced()).padding(8).background(.black.opacity(0.5), in: Capsule())
                    if cam.recording {
                        Text(Duration.seconds(cam.elapsed), format: .time(pattern: .minuteSecond))
                            .font(.headline.monospacedDigit()).padding(8).background(.red, in: Capsule())
                    }
                }
                .foregroundStyle(.white)
                Spacer()
                if !cam.recording {
                    Text("Phone sideways · whole table in view · keep it still")
                        .font(.footnote).padding(8).background(.black.opacity(0.5), in: Capsule()).foregroundStyle(.white)
                }
                Button {
                    if cam.recording { cam.stopRecording { onFinish($0) } } else { cam.startRecording() }
                } label: {
                    ZStack {
                        Circle().stroke(.white, lineWidth: 4).frame(width: 72, height: 72)
                        RoundedRectangle(cornerRadius: cam.recording ? 6 : 30).fill(.red)
                            .frame(width: cam.recording ? 30 : 60, height: cam.recording ? 30 : 60)
                    }
                }
                .padding(.bottom, 24)
            }
        }
        .task { await cam.configure() }
        .onChange(of: scenePhase) { _, phase in
            guard phase == .active, !cam.authorized else { return }
            Task { await cam.configure() }
        }
    }
}

@MainActor
final class Camera: NSObject, ObservableObject, AVCaptureFileOutputRecordingDelegate {
    let session = AVCaptureSession()
    private let output = AVCaptureMovieFileOutput()
    @Published var authorized = false
    @Published var permissionDenied = false
    @Published var recording = false
    @Published var elapsed: Double = 0
    @Published var formatLabel = ""
    private var timer: Timer?
    private var done: ((URL?) -> Void)?
    private var configured = false

    func configure() async {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        if status == .authorized {
            authorized = true
        } else if status == .notDetermined {
            authorized = await AVCaptureDevice.requestAccess(for: .video)
        } else {
            authorized = false
        }
        let resolvedStatus = AVCaptureDevice.authorizationStatus(for: .video)
        permissionDenied = resolvedStatus == .denied || resolvedStatus == .restricted
        guard authorized, !configured,
              let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return }
        session.beginConfiguration()
        session.sessionPreset = .inputPriority
        if session.canAddInput(input) { session.addInput(input) }
        if session.canAddOutput(output) { session.addOutput(output) }
        // 选 1080p 下帧率最高的格式（最多 120）
        let best = device.formats.filter { CMVideoFormatDescriptionGetDimensions($0.formatDescription).height == 1080 }
            .max { a, b in (a.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0) < (b.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0) }
        if let f = best, (try? device.lockForConfiguration()) != nil {
            let fps = min(120, f.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 30)
            device.activeFormat = f
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))
            device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))
            device.unlockForConfiguration()
            formatLabel = "1080p · \(Int(fps)) fps"
        }
        if let c = output.connection(with: .video), c.isVideoRotationAngleSupported(0) { c.videoRotationAngle = 0 }  // 横屏
        session.commitConfiguration()
        configured = true
        let s = session
        Task.detached { s.startRunning() }
    }

    func startRecording() {
        let url = URL.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mov")
        output.startRecording(to: url, recordingDelegate: self)
        recording = true; elapsed = 0
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in Task { @MainActor in self?.elapsed += 1 } }
    }

    func stopRecording(_ completion: @escaping (URL?) -> Void) {
        done = completion
        output.stopRecording()
    }

    func stop() {
        if output.isRecording { output.stopRecording() }
        let s = session
        Task.detached { s.stopRunning() }
    }

    nonisolated func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo url: URL, from connections: [AVCaptureConnection], error: Error?) {
        Task { @MainActor in
            timer?.invalidate(); recording = false
            stop()
            done?(error == nil ? url : nil); done = nil
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.layer.session = session
        v.layer.videoGravity = .resizeAspect
        if let c = v.layer.connection, c.isVideoRotationAngleSupported(0) { c.videoRotationAngle = 0 }
        return v
    }
    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        override var layer: AVCaptureVideoPreviewLayer { super.layer as! AVCaptureVideoPreviewLayer }
    }
}
