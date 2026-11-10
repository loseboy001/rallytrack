import Foundation
import SwiftData

/// 一场训练/比赛：视频存在 App 文档目录，分析结果以 JSON 保存
@Model
final class SessionRecord {
    var id: UUID
    var date: Date
    var videoFile: String
    var resultData: Data?
    var totalShots: Int
    var rallyCount: Int
    var longestRally: Int
    var duration: Double

    init(videoFile: String, date: Date = .now) {
        id = UUID()
        self.date = date
        self.videoFile = videoFile
        totalShots = 0; rallyCount = 0; longestRally = 0; duration = 0
    }

    var videoURL: URL { SessionRecord.videosDir.appendingPathComponent(videoFile) }

    var result: SessionResult? {
        get { resultData.flatMap { try? JSONDecoder().decode(SessionResult.self, from: $0) } }
        set {
            resultData = newValue.flatMap { try? JSONEncoder().encode($0) }
            totalShots = newValue?.totalShots ?? 0
            rallyCount = newValue?.rallies.count ?? 0
            longestRally = newValue?.longestRally ?? 0
            duration = newValue?.duration ?? 0
        }
    }

    var averageShots: Double { rallyCount == 0 ? 0 : Double(totalShots) / Double(rallyCount) }

    static var videosDir: URL {
        let d = URL.documentsDirectory.appendingPathComponent("Videos", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    /// 把外部视频复制进 App，返回文件名
    static func importVideo(from url: URL) throws -> String {
        let name = UUID().uuidString + "." + (url.pathExtension.isEmpty ? "mov" : url.pathExtension)
        try FileManager.default.copyItem(at: url, to: videosDir.appendingPathComponent(name))
        return name
    }
}
