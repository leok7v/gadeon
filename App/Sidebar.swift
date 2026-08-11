import SwiftUI

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

struct Sidebar: View {

    let model: ChatModel
    @Binding var theme: AppTheme
    let onClose: () -> Void
    let onOpen: (UUID) -> Void
    let onNewChat: () -> Void
    let onSettings: () -> Void
    let onRename: (ConversationStore.Convo) -> Void

    @State private var armedDelete: UUID?
    @State private var disarmTask: Task<Void, Never>?
    @State private var query = ""

    private var hasHistory: Bool { !ConversationStore.shared.list.isEmpty }

    var body: some View {
        let items = shown
        return VStack(spacing: 0) {
            closeRow
            newChatRow
            if hasHistory {
                searchField
                Divider()
                if items.isEmpty {
                    noMatches
                } else {
                    history(items)
                }
            } else {
                emptyState
            }
            Divider()
            footer
        }
    }

    private var shown: [ConversationStore.Convo] {
        let store = ConversationStore.shared
        return ConversationSearch.active(query)
            ? ConversationSearch.rank(store.list, store.words, query)
            : store.list
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "bubble.left.and.bubble.right")
                .appFont(.largeTitle)
                .foregroundStyle(.tertiary)
            Text("No conversations yet")
                .appFont(.callout)
                .foregroundStyle(.secondary)
            Text("Your chats will appear here.")
                .appFont(.caption)
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

    private var searchField: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Search", text: $query)
                .textFieldStyle(.plain)
            if !query.isEmpty {
                Button { query = "" } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var noMatches: some View {
        VStack(spacing: 6) {
            Text("No matches").foregroundStyle(.secondary)
            Text("for \u{201C}\(query)\u{201D}")
                .appFont(.caption)
                .foregroundStyle(.tertiary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }

    // A phone's List row carries far more padding than a Mac's, so the same
    // two lines of text stand in twice the height and a screenful holds five
    // conversations. nil keeps the desktop's own defaults, which are right.
    private var rowInsets: EdgeInsets? {
        isOS ? EdgeInsets(top: 2, leading: 8, bottom: 2, trailing: 8) : nil
    }

    private func history(_ items: [ConversationStore.Convo]) -> some View {
        List {
            if ConversationSearch.active(query) {
                ForEach(items) { convo in
                    historyRow(convo)
                        .listRowBackground(Color.clear)
                        .listRowInsets(rowInsets)
                        .deleteDisabled(model.busy)
                }
                .onDelete { offsets in delete(offsets, in: items) }
            } else {
                ForEach(Sidebar.groups(items)) { group in
                    Section(group.title) {
                        ForEach(group.items) { convo in
                            historyRow(convo)
                                .listRowBackground(Color.clear)
                                .listRowInsets(rowInsets)
                                .deleteDisabled(model.busy)
                        }
                        .onDelete { offsets in
                            delete(offsets, in: group.items)
                        }
                    }
                }
            }
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .modifier(TightSections())
    }

    // The conversation on screen, which is the live one once it has been
    // committed and a reopened one from the moment it is opened.
    private func isCurrent(_ convo: ConversationStore.Convo) -> Bool {
        convo.id == model.currentConversationId
    }

    private func historyRow(_ convo: ConversationStore.Convo) -> some View {
        HStack(spacing: 6) {
            Button { armedDelete = nil; onOpen(convo.id) } label: { row(convo) }
                .buttonStyle(.plain)
                .disabled(model.busy)
                .help(rowHelp)
            if !isOS {
                if armedDelete == convo.id {
                    Button { confirmArmed(convo) } label: {
                        Text("Delete")
                            .appFont(.caption)
                            .fontWeight(.semibold)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(Color.red, in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .disabled(model.busy)
                    .help("Click to delete; this cannot be undone")
                } else {
                    Button { requestDelete(convo) } label: {
                        Image(systemName: "trash")
                            .appFont(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .disabled(model.busy)
                    .help(trashHelp)
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isCurrent(convo) ? Color.accentColor.opacity(0.15)
                                     : .clear,
                    in: RoundedRectangle(cornerRadius: 7))
        .animation(.easeInOut(duration: 0.15), value: armedDelete)
        // Right-click on macOS, long-press on iOS, from one modifier.
        // Rename is offered mid-turn where Open and Delete are not: a title
        // touches neither the engine nor the live transcript.
        .contextMenu {
            Button { onRename(convo) } label: {
                Label("Rename\u{2026}", systemImage: "pencil")
            }
        }
    }

    private var trashHelp: String {
        model.busy ? "Available once this turn has finished"
                   : "Delete conversation"
    }

    // A running turn holds an index into the live transcript; opening
    // another conversation would redirect it there instead of stopping it.
    private var rowHelp: String {
        model.busy ? "Available once this turn has finished"
                   : "Open this conversation"
    }

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
        let reason = ConversationSearch.active(query)
            ? ConversationSearch.reason(convo, query) : nil
        let current = isCurrent(convo)
        return VStack(alignment: .leading, spacing: 2) {
            Text(convo.title)
                .lineLimit(1)
                .fontWeight(current ? .semibold : .regular)
                .foregroundStyle(current ? Color.accentColor : .primary)
            Text(reason ?? Sidebar.when(convo.updated))
                .appFont(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
    }

    // Snapshot the ids before deleting: it mutates the store's list, so
    // the offsets would otherwise index a shifting array.

    private func delete(_ offsets: IndexSet,
                        in items: [ConversationStore.Convo]) {
        let ids = offsets.map { i in items[i].id }
        for id in ids { model.deleteConversation(id) }
    }

    private var footer: some View {
        VStack(spacing: 18) {
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
