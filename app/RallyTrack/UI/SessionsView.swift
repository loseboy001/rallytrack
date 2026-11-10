import PhotosUI
import SwiftData
import SwiftUI

/// 从相册导入的视频
struct Movie: Transferable {
    let url: URL
    /// 按原文件格式依次尝试：mp4 要 mp4、mov 要 mov。只声明 .movie 时，mp4 会被系统转码成 mov，帧率掉到 30
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .mpeg4Movie, importing: copy)
        FileRepresentation(importedContentType: .quickTimeMovie, importing: copy)
        FileRepresentation(importedContentType: .movie, importing: copy)
    }

    private static func copy(_ received: ReceivedTransferredFile) throws -> Movie {
        let dst = URL.temporaryDirectory.appendingPathComponent(UUID().uuidString + "." + received.file.pathExtension)
        try FileManager.default.copyItem(at: received.file, to: dst)
        return Movie(url: dst)
    }
}

struct SessionsView: View {
    @Environment(\.modelContext) private var ctx
    @Query(sort: \SessionRecord.date, order: .reverse) private var sessions: [SessionRecord]
    @State private var picks: [PhotosPickerItem] = []
    @EnvironmentObject private var queue: AnalysisQueue
    @State private var showRecorder = false
    @State private var importLeft = 0
    @State private var error: String?
    @State private var showPaywall = false
    @State private var path: [SessionRecord] = []
    @State private var debugRally: RallyInfo?
    @State private var guide: GuideTarget?
    @State private var showPicker = false
    @AppStorage("hideShootingGuide") private var hideGuide = false
    enum GuideTarget: Identifiable { case record, importVideo; var id: Self { self } }
    @EnvironmentObject private var store: Store

    var body: some View {
        NavigationStack(path: $path) {
            List {
                Section {
                    HStack(spacing: 12) {
                        Button {
                            if !store.canAnalyze { showPaywall = true } else if hideGuide { showRecorder = true } else { guide = .record }
                        } label: {
                            Label("Record", systemImage: "video.fill").frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        Button {
                            if !store.canAnalyze { showPaywall = true } else if hideGuide { showPicker = true } else { guide = .importVideo }
                        } label: {
                            Label(importLeft > 0 ? "Importing \(importLeft)…" : "Import", systemImage: "photo.on.rectangle").frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .disabled(importLeft > 0)
                    }
                    .controlSize(.large)
                    .listRowBackground(Color.clear)
                    Button { guide = .importVideo } label: {
                        Label("How to film · see reference shot", systemImage: "camera.viewfinder").font(.footnote)
                    }
                    .listRowBackground(Color.clear)
                    if !store.isPro {
                        Button { showPaywall = true } label: {
                            Text(store.freeAnalysesLeft > 0 ? "\(store.freeAnalysesLeft) free analyses left · Go Pro for unlimited" : "Free analyses used up · Go Pro to keep analyzing")
                                .font(.footnote.weight(.semibold)).foregroundStyle(Theme.lime)
                        }
                        .listRowBackground(Color.clear)
                    }
                }
                if !queue.jobs.isEmpty {
                    Section {
                        ForEach(queue.jobs) { JobRow(job: $0) }
                            .onDelete { idx in idx.map { queue.jobs[$0] }.forEach(queue.remove) }
                    } header: {
                        HStack {
                            Text("Analyzing")
                            Spacer()
                            if queue.confirmCount > 1 {
                                Button("Confirm all (\(queue.confirmCount))") { queue.confirmAll() }
                                    .font(.footnote.weight(.semibold)).textCase(nil)
                            }
                        }
                    }
                }
                Section("Sessions") {
                    if sessions.isEmpty && queue.jobs.isEmpty {
                        Text("No sessions yet").foregroundStyle(.secondary)
                    }
                    ForEach(sessions) { s in
                        NavigationLink(value: s) { SessionRow(session: s) }
                    }
                    .onDelete { idx in
                        for i in idx {
                            try? FileManager.default.removeItem(at: sessions[i].videoURL)
                            ctx.delete(sessions[i])
                        }
                    }
                }
            }
            .navigationTitle("RallyTrack")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    if store.isPro { Label("Pro", systemImage: "checkmark.seal.fill").foregroundStyle(Theme.lime) }
                    else { Button("Go Pro") { showPaywall = true } }
                }
            }
            .sheet(isPresented: $showPaywall) { PaywallView() }
            .sheet(item: $guide) { g in
                ShootingGuideView(actionTitle: g == .record ? "Start recording" : "Choose a video") {
                    Task { try? await Task.sleep(for: .milliseconds(400)); if g == .record { showRecorder = true } else { showPicker = true } }
                }
            }
            .photosPicker(isPresented: $showPicker, selection: $picks, maxSelectionCount: 20, selectionBehavior: .ordered, matching: .videos, preferredItemEncoding: .current)  // .current：不让系统转码，保留原始帧率
            .navigationDestination(for: SessionRecord.self) { ReportView(session: $0) }
            .onChange(of: picks) { _, items in
                guard !items.isEmpty else { return }
                picks = []
                // 没开会员时最多排到免费额度用完，多选的部分引导开通
                let allowed = store.isPro ? items.count : max(0, store.freeAnalysesLeft - queue.activeCount)
                if allowed < items.count { showPaywall = true }
                importLeft = min(allowed, items.count)
                Task {
                    for item in items.prefix(allowed) {
                        defer { importLeft -= 1 }
                        do {
                            guard let m = try await item.loadTransferable(type: Movie.self) else { continue }
                            let file = try SessionRecord.importVideo(from: m.url)
                            try? FileManager.default.removeItem(at: m.url)
                            queue.add(file: file)
                        } catch { self.error = error.localizedDescription }
                    }
                }
            }
            #if DEBUG
            .task {
                // 调试：启动参数 -demoFile <Videos 目录下的文件名> 直接分析，便于在模拟器里自动验证
                if let f = UserDefaults.standard.string(forKey: "demoFile"), sessions.isEmpty, queue.jobs.isEmpty {
                    queue.add(file: f)
                }
                // 调试：-analyzeFile <Videos 下的文件名> 远程触发一次分析（配合 -autoConfirm YES 跳过确认球台），用来在真机上核对结果
                if let f = CommandLine.arguments.firstIndex(of: "-analyzeFile").map({ CommandLine.arguments[$0 + 1] }),
                   !queue.jobs.contains(where: { $0.file == f }) {
                    queue.add(file: f)
                }
                // 调试：-debugScreen report|rally|paywall 直接打开对应页面截图
                let screen = UserDefaults.standard.string(forKey: "debugScreen")
                if screen == "paywall" { showPaywall = true }
                guard let latest = sessions.first else { return }
                switch screen {
                case "report": path = [latest]
                case "rally": debugRally = latest.result?.rallies.first
                default: break
                }
            }
            .sheet(item: $debugRally) { r in if let s = sessions.first { RallyPlayerView(session: s, rally: r) } }
            #endif
            .fullScreenCover(isPresented: $showRecorder) {
                RecorderView { url in
                    showRecorder = false
                    if let url, let file = try? SessionRecord.importVideo(from: url) {
                        try? FileManager.default.removeItem(at: url)
                        queue.add(file: file)
                    }
                }
            }
            .alert("Error", isPresented: .constant(error != nil)) { Button("OK") { error = nil } } message: { Text(error ?? "") }
        }
    }
}

struct SessionRow: View {
    let session: SessionRecord
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(session.date, format: .dateTime.month().day().hour().minute()).font(.headline)
                Text("\(session.rallyCount) rallies · \(Int(session.duration / 60)) min")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing) {
                Text("\(session.totalShots)").font(.title2.monospacedDigit().bold())
                Text("shots").font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}
