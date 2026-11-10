import Charts
import SwiftData
import SwiftUI

/// 训练趋势：按场次看平均每回合板数、总板数和最长回合纪录
struct TrendsView: View {
    @Query(sort: \SessionRecord.date) private var sessions: [SessionRecord]

    private var done: [SessionRecord] { sessions.filter { $0.resultData != nil } }

    var body: some View {
        NavigationStack {
            ScrollView {
                if done.isEmpty {
                    ContentUnavailableView("No data yet", systemImage: "chart.line.uptrend.xyaxis",
                                           description: Text("Analyze a session to see your trends."))
                        .padding(.top, 80)
                } else {
                    VStack(spacing: 16) {
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                            StatTile(value: "\(done.count)", label: "Sessions")
                            StatTile(value: "\(done.reduce(0) { $0 + $1.totalShots })", label: "Total shots")
                            StatTile(value: "\(done.map(\.longestRally).max() ?? 0)", label: "Best rally", highlight: true)
                        }
                        Card(title: "Average shots per rally") {
                            Chart(done) { s in
                                LineMark(x: .value("Date", s.date), y: .value("Shots", s.averageShots))
                                    .foregroundStyle(Theme.lime).interpolationMethod(.monotone)
                                PointMark(x: .value("Date", s.date), y: .value("Shots", s.averageShots)).foregroundStyle(Theme.lime)
                            }
                            .frame(height: 180)
                        }
                        Card(title: "Shots per session") {
                            Chart(done) { s in
                                BarMark(x: .value("Date", s.date, unit: .day), y: .value("Shots", s.totalShots)).foregroundStyle(Theme.cyan)
                            }
                            .frame(height: 160)
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("Trends")
        }
    }
}
