import SwiftUI

// App-level appearance, chosen in the sidebar and applied at the root.
enum AppTheme: String, CaseIterable, Identifiable {
    case system, light, dark
    var id: String { rawValue }
    var scheme: ColorScheme? {
        switch self {
        case .system: return nil
        case .light: return .light
        case .dark: return .dark
        }
    }
    var icon: String {
        switch self {
        case .system: return "circle.lefthalf.filled"
        case .light: return "sun.max"
        case .dark: return "moon"
        }
    }
}

// The menu sidebar, shown as an overlay drawer sliding in from the leading
// edge. Search + read-only conversation history (swipe to delete) when there
// is any, a quiet hint when there is not, and Theme + Settings pinned at the
// bottom (the thumb zone). Selecting a conversation opens it read-only.

struct Sidebar: View {

    let model: ChatModel
    @Binding var theme: AppTheme
    let onClose: () -> Void
    let onOpen: (UUID) -> Void
    let onNewChat: () -> Void
    let onSearch: () -> Void
    let onSettings: () -> Void

    // The conversation whose trash was tapped (macOS): its row shows an inline
    // red Delete to confirm -- the same two-step feel as the iOS swipe -- and
    // auto-disarms after a beat. Only one row is armed at a time.
    @State private var armedDelete: UUID?
    @State private var disarmTask: Task<Void, Never>?

    private var hasHistory: Bool { !ConversationStore.shared.list.isEmpty }

    var body: some View {
        VStack(spacing: 0) {
            closeRow
            newChatRow
            if hasHistory {
                searchRow
                Divider()
                history
            } else {
                emptyState
            }
            Divider()
            footer
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.largeTitle)
                .foregroundStyle(.tertiary)
            Text("No conversations yet")
                .font(.callout)
                .foregroundStyle(.secondary)
            Text("Your chats will appear here.")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }

    private var closeRow: some View {
        HStack {
            Button(action: onClose) {
                Image(systemName: "xmark")
            }
            .buttonStyle(.plain)
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
    }

    // The sidebar's compose action: start a fresh conversation and close the
    // drawer (onNewChat) -- the same one-tap "pick a destination" as opening a
    // conversation row, so it is not a stray duplicate of the top-bar button
    // (which is disabled while the drawer is open). Accent-tinted to read as
    // the primary action; disabled mid-reply like the top-bar New Chat.
    private var newChatRow: some View {
        Button(action: onNewChat) {
            Label("New Chat", systemImage: "square.and.pencil")
                .foregroundStyle(.tint)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(model.busy)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var searchRow: some View {
        Button(action: onSearch) {
            Label("Search", systemImage: "magnifyingglass")
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    // Deleting the open chat starts a fresh one (deleteConversation).
    private var history: some View {
        List {
            ForEach(Sidebar.groups(ConversationStore.shared.list)) { group in
                Section(group.title) {
                    ForEach(group.items) { convo in
                        historyRow(convo)
                            .listRowBackground(Color.clear)
                    }
                    .onDelete { offsets in delete(offsets, in: group.items) }
                }
            }
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
    }

    // A tap opens the conversation; the trailing trash removes it. macOS has no
    // swipe-to-delete, so the trash IS the delete affordance there; iOS keeps
    // its swipe (onDelete) and needs no button. The trash routes through the
    // confirm alert unless the user turned confirmation off in Settings.
    private func historyRow(_ convo: ConversationStore.Convo) -> some View {
        HStack(spacing: 6) {
            Button { armedDelete = nil; onOpen(convo.id) } label: { row(convo) }
                .buttonStyle(.plain)
            if !isOS {
                if armedDelete == convo.id {
                    Button { confirmArmed(convo) } label: {
                        Text("Delete")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(Color.red, in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .help("Click to delete; this cannot be undone")
                } else {
                    Button { requestDelete(convo) } label: {
                        Image(systemName: "trash")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Delete conversation")
                }
            }
        }
        .animation(.easeInOut(duration: 0.15), value: armedDelete)
    }

    // macOS: the first trash click arms the row (inline red Delete); confirm
    // deletes. With confirmation off, the click deletes outright.
    private func requestDelete(_ convo: ConversationStore.Convo) {
        if model.confirmDeleteConversation {
            armedDelete = convo.id
            scheduleDisarm(convo.id)
        } else {
            model.deleteConversation(convo.id)
        }
    }

    private func confirmArmed(_ convo: ConversationStore.Convo) {
        disarmTask?.cancel()
        armedDelete = nil
        model.deleteConversation(convo.id)
    }

    // Drop the armed state after a few seconds so a stray red Delete does not
    // linger in the list.
    private func scheduleDisarm(_ id: UUID) {
        disarmTask?.cancel()
        disarmTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(4))
            if !Task.isCancelled, armedDelete == id { armedDelete = nil }
        }
    }

    struct Group: Identifiable {
        let id: String
        let items: [ConversationStore.Convo]
        var title: String { id }
    }

    // The list is already newest-first, so a stable bucketing keeps order.
    private static func groups(_ list: [ConversationStore.Convo]) -> [Group] {
        let cal = Calendar.current
        let now = Date()
        let titles = ["Today", "Yesterday", "Previous 7 Days", "Older"]
        var buckets: [[ConversationStore.Convo]] = [[], [], [], []]
        for convo in list {
            let idx: Int
            if cal.isDateInToday(convo.updated) {
                idx = 0
            } else if cal.isDateInYesterday(convo.updated) {
                idx = 1
            } else if let days = cal.dateComponents(
                [.day], from: convo.updated, to: now).day, days < 7 {
                idx = 2
            } else {
                idx = 3
            }
            buckets[idx].append(convo)
        }
        var result: [Group] = []
        for i in buckets.indices where !buckets[i].isEmpty {
            result.append(Group(id: titles[i], items: buckets[i]))
        }
        return result
    }

    private func row(_ convo: ConversationStore.Convo) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(convo.title).lineLimit(1)
            Text(Sidebar.when(convo.updated))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
    }

    // Snapshot the ids before deleting: deleting mutates the store's list, so
    // holding onto the offsets would index a shifting array.
    private func delete(_ offsets: IndexSet,
                        in items: [ConversationStore.Convo]) {
        let ids = offsets.map { i in items[i].id }
        for id in ids { model.deleteConversation(id) }
    }

    private var footer: some View {
        VStack(spacing: 10) {
            Picker("Theme", selection: $theme) {
                ForEach(AppTheme.allCases) { option in
                    Image(systemName: option.icon).tag(option)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            Button(action: onSettings) {
                Label("Settings", systemImage: "gearshape")
            }
            .buttonStyle(.plain)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
    }

    private static func when(_ date: Date) -> String {
        let f = RelativeDateTimeFormatter()
        return f.localizedString(for: date, relativeTo: Date())
    }
}
