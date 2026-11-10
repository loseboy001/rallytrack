import SwiftData
import SwiftUI

@main
struct RallyTrackApp: App {
    @StateObject private var store = Store()
    @StateObject private var queue = AnalysisQueue()
    @Environment(\.scenePhase) private var phase

    var body: some Scene {
        WindowGroup {
            RootView()
                .preferredColorScheme(.dark)
                .tint(Theme.lime)
                .environmentObject(store)
                .environmentObject(queue)
        }
        .modelContainer(for: SessionRecord.self)
        .onChange(of: phase) { _, p in queue.scenePhase(p) }
    }

    init() {
        #if DEBUG
        // 调试：启动参数 -benchModels YES 在手机上测两个球模型的单帧耗时，结果写进分析日志
        if UserDefaults.standard.bool(forKey: "benchModels") { Task.detached { ModelBench.run() } }
        #endif
    }
}

enum Theme {
    static let lime = Color(red: 0.784, green: 0.941, blue: 0.290)
    static let cyan = Color(red: 0, green: 0.94, blue: 1)
    static let card = Color(white: 0.09)
    static let tableBlue = Color(red: 0.11, green: 0.31, blue: 0.85)
}

struct RootView: View {
    @Environment(\.modelContext) private var ctx
    @EnvironmentObject private var queue: AnalysisQueue
    @EnvironmentObject private var store: Store

    var body: some View {
        TabView {
            SessionsView()
                .tabItem { Label("Sessions", systemImage: "record.circle") }
            ClipsView()
                .tabItem { Label("Rallies", systemImage: "film.stack") }
            TrendsView()
                .tabItem { Label("Trends", systemImage: "chart.line.uptrend.xyaxis") }
        }
        .task { queue.attach(ctx: ctx, store: store) }
    }
}
