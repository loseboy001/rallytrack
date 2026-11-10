import BackgroundTasks
import SwiftData
import SwiftUI

/// 分析任务队列：导入后立刻回到首页，一个一个在后台跑，首页显示进度。
/// 队列存在本地，App 被杀或被系统打断后，下次打开接着跑。
@MainActor
final class AnalysisQueue: ObservableObject {
    struct Job: Identifiable, Codable, Equatable {
        /// detect 待找球台 → detecting 正在找 → confirm 等用户确认 → queued 待分析 → running 分析中
        enum State: String, Codable { case detect, detecting, confirm, queued, running, needsTable, failed }
        var id = UUID()
        var file: String
        var created = Date()
        var quad: [CGPoint]?     // 球台四角（拉直后的坐标；自动识别后由用户确认，或手动标）
        var k1: Double?          // 镜头畸变
        var replaces: UUID?      // 重新分析时要更新的记录
        var state = State.queued
        var message: String?
        var attempts = 0
    }

    @Published private(set) var jobs: [Job] = []
    @Published private(set) var progress = 0.0
    @Published private(set) var stage = ""
    @Published private(set) var startedAt = Date()
    /// 估计剩余秒数（进度过 5% 后才算，避免一开始乱跳）
    var remaining: Int? {
        guard progress > 0.05, progress < 1 else { return nil }
        let el = Date().timeIntervalSince(startedAt)
        return Int(el / progress * (1 - progress))
    }

    private var ctx: ModelContext?
    private var store: Store?
    private var analyzer: VideoAnalyzer?
    private var detector: VideoAnalyzer?   // 找球台专用，和分析用的分开，避免同时用一个对象
    private var worker: Task<SessionResult, Error>?
    private var running = false
    private var detecting = false   // 找球台单独一条线，不用等前面的视频分析完
    private var interrupted = false   // 分析途中进过后台：这种失败是系统打断的，回来自动重跑
    private var wentBackground = false  // 这次分析过程中 App 去过后台（后台时出的错一律当成被打断）
    private var active = true
    private var bgTask = UIBackgroundTaskIdentifier.invalid
    private var continued: AnyObject?      // iOS 26 的 BGContinuedProcessingTask：切到后台也能接着分析，系统显示进度
    private static let key = "analysisJobs"

    var activeCount: Int { jobs.filter { $0.state != .needsTable && $0.state != .failed }.count }
    var confirmCount: Int { jobs.filter { $0.state == .confirm }.count }

    func attach(ctx: ModelContext, store: Store) {
        guard self.ctx == nil else { return }
        self.ctx = ctx; self.store = store
        if let d = UserDefaults.standard.data(forKey: Self.key), let saved = try? JSONDecoder().decode([Job].self, from: d) {
            jobs = saved.map { j in
                var j = j
                if j.state == .running { j.state = .queued }
                if j.state == .detecting || (j.state == .queued && j.quad == nil) { j.state = .detect }
                return j
            }
        }
        pump()
    }

    func add(file: String, quad: [CGPoint]? = nil, k1: Double? = nil, replaces: UUID? = nil) {
        jobs.append(Job(file: file, quad: quad, k1: k1, replaces: replaces, state: quad == nil ? .detect : .queued))
        save(); pump()
    }

    func setTable(_ job: Job, quad: [CGPoint]) { update(job.id) { $0.quad = quad; $0.state = .queued; $0.message = nil }; pump() }
    func retry(_ job: Job) { update(job.id) { $0.state = $0.quad == nil ? .detect : .queued; $0.message = nil }; pump() }
    /// 用户确认自动找到的球台没问题
    func confirm(_ job: Job) { update(job.id) { $0.state = .queued }; pump() }
    func confirmAll() { for j in jobs where j.state == .confirm { update(j.id) { $0.state = .queued } }; pump() }

    /// 取消并删除任务；不是重新分析的话连视频一起删
    func remove(_ job: Job) {
        if job.state == .running { worker?.cancel() }
        jobs.removeAll { $0.id == job.id }
        if job.replaces == nil { try? FileManager.default.removeItem(at: SessionRecord.videosDir.appendingPathComponent(job.file)) }
        save()
    }

    func scenePhase(_ phase: ScenePhase) {
        active = phase != .background
        // 有系统后台任务撑着时，切到后台不算被打断；没有的话（iOS 26 以前）系统很快会暂停 App，回来要重跑
        if phase == .background, running, continued == nil { interrupted = true }
        if phase == .background, running { wentBackground = true }
        Self.log("scene \(phase) running=\(running) continued=\(continued != nil)")
        if phase == .active { pump() }
    }

    /// 分析日志：Documents/analysis_log.txt（排查用，最多留 200 KB）
    nonisolated static func log(_ msg: String) {
        let url = URL.documentsDirectory.appendingPathComponent("analysis_log.txt")
        let line = "\(ISO8601DateFormatter().string(from: Date())) \(msg)\n"
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); try? h.close()
            if let a = try? FileManager.default.attributesOfItem(atPath: url.path), (a[.size] as? Int ?? 0) > 200_000 { try? FileManager.default.removeItem(at: url) }
        } else {
            try? line.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    private func update(_ id: UUID, _ f: (inout Job) -> Void) {
        guard let i = jobs.firstIndex(where: { $0.id == id }) else { return }
        f(&jobs[i]); save()
    }

    private func save() { UserDefaults.standard.set(try? JSONEncoder().encode(jobs), forKey: Self.key) }

    private func pump() {
        guard active, ctx != nil else { return }
        // 找球台（每个约 1 秒）和分析视频并行：正在分析别的视频时，新导入的也能马上确认球台
        if !detecting, let job = jobs.first(where: { $0.state == .detect }) {
            detecting = true
            update(job.id) { $0.state = .detecting }
            Task { await detect(job); detecting = false; pump() }
        }
        guard !running else { return }
        guard let job = jobs.first(where: { $0.state == .queued }) else {
            if !running { setBusy(false) }
            return
        }
        running = true; interrupted = false; wentBackground = false; progress = 0; stage = ""; startedAt = Date()
        Self.log("start \(job.file) attempt \(job.attempts + 1)")
        update(job.id) { $0.state = .running }
        setBusy(true)
        Task { await run(job); running = false; pump() }
    }

    private func detect(_ job: Job) async {
        let url = SessionRecord.videosDir.appendingPathComponent(job.file)
        do {
            if detector == nil { detector = try VideoAnalyzer() }
            let a = detector!
            let q = try await Task.detached(priority: .userInitiated) { try await a.detectTable(url: url) }.value
            update(job.id) { $0.quad = q.quad; $0.k1 = q.k1; $0.state = .confirm }
            #if DEBUG
            if UserDefaults.standard.bool(forKey: "autoConfirm") { confirm(job) }  // 调试：-autoConfirm YES 跳过确认
            #endif
        } catch {
            update(job.id) { $0.state = .needsTable }
        }
    }

    private func run(_ job: Job) async {
        let url = SessionRecord.videosDir.appendingPathComponent(job.file)
        do {
            if analyzer == nil { analyzer = try VideoAnalyzer() }
            let a = analyzer!, table = job.quad.map { TableFit(quad: $0, k1: job.k1 ?? 0) }
            let t = Task.detached(priority: .userInitiated) {
                try await a.analyze(url: url, table: table) { p, s in
                    Task { @MainActor [weak self] in
                        self?.progress = p; self?.stage = s
                        if #available(iOS 26.0, *), let t = self?.continued as? BGContinuedProcessingTask {
                            t.progress.completedUnitCount = Int64(p * 100)
                        }
                    }
                }
            }
            worker = t
            if #available(iOS 26.0, *) { startContinued(for: job) }
            let result = try await t.value
            endContinued(success: true)
            Self.log("done \(job.file) rallies \(result.rallies.count) shots \(result.totalShots) in \(Int(Date().timeIntervalSince(startedAt)))s")
            finish(job, result)
        } catch is CancellationError {
            endContinued(success: false)
            if interrupted { update(job.id) { $0.state = .queued } }
            return
        } catch AnalysisError.tableNotFound {
            endContinued(success: false)
            update(job.id) { $0.state = .needsTable }
        } catch {
            endContinued(success: false)
            Self.log("error \(job.file): \(error) interrupted=\(interrupted) background=\(wentBackground)")
            if interrupted || wentBackground {
                // 后台时系统可能不让用神经网络引擎等：不算失败，回到前台重跑
                update(job.id) { $0.state = .queued }
            } else if job.attempts < 1 {
                update(job.id) { $0.state = .queued; $0.attempts += 1 }   // 自动再试一次
            } else {
                update(job.id) { $0.state = .failed; $0.message = error.localizedDescription }
            }
        }
    }

    private func finish(_ job: Job, _ result: SessionResult) {
        guard jobs.contains(where: { $0.id == job.id }), let ctx else { return }
        var rec: SessionRecord?
        if let id = job.replaces { rec = try? ctx.fetch(FetchDescriptor<SessionRecord>(predicate: #Predicate { $0.id == id })).first }
        if rec == nil {
            rec = SessionRecord(videoFile: job.file, date: job.created)
            ctx.insert(rec!)
            store?.didAnalyze()
        }
        rec?.result = result
        try? ctx.save()
        jobs.removeAll { $0.id == job.id }
        save()
    }

    /// iOS 26：申请「继续处理任务」，切到别的 App 或锁屏后分析继续跑，系统在灵动岛/锁屏显示进度，用户可以取消
    @available(iOS 26.0, *)
    private func startContinued(for job: Job) {
        let id = "com.rallytrack.app.analysis.\(job.id.uuidString.prefix(8))"
        let ok = BGTaskScheduler.shared.register(forTaskWithIdentifier: id, using: nil) { [weak self] task in
            guard let task = task as? BGContinuedProcessingTask else { task.setTaskCompleted(success: false); return }
            task.progress.totalUnitCount = 100
            task.expirationHandler = {
                // 系统要收回（或用户在系统界面点了取消）：停下，回到前台后重新排队
                Task { @MainActor in self?.interrupted = true; self?.worker?.cancel(); self?.continued = nil }
            }
            Task { @MainActor in self?.continued = task }
        }
        guard ok else { Self.log("continued task register failed"); return }
        let req = BGContinuedProcessingTaskRequest(identifier: id, title: String(localized: "Analyzing table tennis video"),
                                                   subtitle: String(localized: "RallyTrack"))
        req.strategy = .fail
        do { try BGTaskScheduler.shared.submit(req); Self.log("continued task submitted") }
        catch { Self.log("continued task submit failed: \(error)") }
    }

    private func endContinued(success: Bool) {
        if #available(iOS 26.0, *), let t = continued as? BGContinuedProcessingTask {
            t.progress.completedUnitCount = 100
            t.setTaskCompleted(success: success)
        }
        continued = nil
    }

    /// 分析时不锁屏；切到别的 App 时向系统多要一点时间
    private func setBusy(_ busy: Bool) {
        UIApplication.shared.isIdleTimerDisabled = busy
        if busy, bgTask == .invalid {
            bgTask = UIApplication.shared.beginBackgroundTask(withName: "analysis") { [weak self] in
                MainActor.assumeIsolated { self?.endBackground() }
            }
        } else if !busy { endBackground() }
    }

    private func endBackground() {
        if bgTask != .invalid { UIApplication.shared.endBackgroundTask(bgTask); bgTask = .invalid }
    }
}
