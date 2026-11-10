import StoreKit
import SwiftUI

/// 会员状态：StoreKit 2。TestFlight 下走沙盒，不会真实扣费。
@MainActor
final class Store: ObservableObject {
    static let productIDs = ["com.rallytrack.pro.monthly", "com.rallytrack.pro.yearly"]

    @Published private(set) var isPro = false

    /// 新用户免费额度：分析 3 场、导出 3 次，用完需开通会员
    static let freeQuota = 3
    @Published private(set) var usedAnalyses = UserDefaults.standard.integer(forKey: "usedAnalyses")
    @Published private(set) var usedExports = UserDefaults.standard.integer(forKey: "usedExports")
    var freeAnalysesLeft: Int { max(0, Self.freeQuota - usedAnalyses) }
    var freeExportsLeft: Int { max(0, Self.freeQuota - usedExports) }
    var canAnalyze: Bool { isPro || freeAnalysesLeft > 0 }
    var canExport: Bool { isPro || freeExportsLeft > 0 }

    func didAnalyze() {
        guard !isPro else { return }
        usedAnalyses += 1; UserDefaults.standard.set(usedAnalyses, forKey: "usedAnalyses")
    }

    func didExport() {
        guard !isPro else { return }
        usedExports += 1; UserDefaults.standard.set(usedExports, forKey: "usedExports")
    }
    private var updates: Task<Void, Never>?

    init() {
        // 监听续订、退款、在其他设备上购买等变化
        updates = Task { [weak self] in
            for await result in Transaction.updates {
                if case .verified(let t) = result { await t.finish() }
                await self?.refresh()
            }
        }
        Task { await refresh() }
    }

    deinit { updates?.cancel() }

    /// 根据当前有效的订阅重新计算会员状态
    func refresh() async {
        var pro = false
        for await result in Transaction.currentEntitlements {
            if case .verified(let t) = result, Self.productIDs.contains(t.productID), t.revocationDate == nil {
                pro = true
            }
        }
        #if DEBUG
        // 调试版（Xcode 直接装到手机）默认不限次数、全部解锁；要看付费墙用启动参数 -forceFree YES
        if !UserDefaults.standard.bool(forKey: "forceFree") { pro = true }
        #endif
        isPro = pro
    }
}

/// 会员页：苹果自带的订阅商店界面（价格、试用、恢复购买、条款都由系统处理）
struct PaywallView: View {
    @EnvironmentObject private var store: Store
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        SubscriptionStoreView(productIDs: Store.productIDs) {
            VStack(spacing: 14) {
                Image(systemName: "sparkles").font(.system(size: 44)).foregroundStyle(Theme.lime)
                Text("RallyTrack Pro").font(.largeTitle.bold())
                VStack(alignment: .leading, spacing: 10) {
                    feature("waveform.path.ecg", "Unlimited session analysis")
                    feature("film", "Unlimited rally video exports")
                    feature("square.and.arrow.up", "Share highlights to social media")
                    feature("heart", "Support future features")
                }
                Text("Everything is analyzed on your phone.").font(.footnote).foregroundStyle(.secondary)
            }
            .padding()
        }
        .subscriptionStoreControlStyle(.prominentPicker)
        .subscriptionStoreButtonLabel(.multiline)
        .storeButton(.visible, for: .restorePurchases)
        .onInAppPurchaseCompletion { _, result in
            await store.refresh()
            if case .success(.success) = result { dismiss() }
        }
        .tint(Theme.lime)
    }

    private func feature(_ icon: String, _ text: LocalizedStringKey) -> some View {
        Label { Text(text) } icon: { Image(systemName: icon).foregroundStyle(Theme.lime) }
    }
}

/// 会员功能占位：非会员看到模糊内容 + 解锁按钮
struct ProGate<Content: View>: View {
    @EnvironmentObject private var store: Store
    @State private var showPaywall = false
    @ViewBuilder var content: Content

    var body: some View {
        if store.isPro {
            content
        } else {
            content
                .blur(radius: 8)
                .allowsHitTesting(false)
                .overlay {
                    Button { showPaywall = true } label: {
                        Label("Unlock with Pro", systemImage: "lock.fill").font(.headline)
                            .padding(.horizontal, 18).padding(.vertical, 10)
                            .background(Theme.lime, in: Capsule()).foregroundStyle(.black)
                    }
                }
                .sheet(isPresented: $showPaywall) { PaywallView() }
        }
    }
}
