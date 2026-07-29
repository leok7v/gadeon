import SwiftUI

// Full-text search over saved conversations (titles + message text). A hit
// opens its conversation read-only.

struct SearchView: View {

    let onOpen: (UUID) -> Void
    let onClose: () -> Void
    @State private var query = ""

    struct Hit: Identifiable {
        let id: UUID
        let title: String
        let snippet: String
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            field
            Divider()
            content
        }
    }

    private var header: some View {
        HStack {
            Text("Search").font(.headline)
            Spacer()
            Button("Done", action: onClose)
                .keyboardShortcut(.defaultAction)
            Button("", action: onClose)
                .keyboardShortcut(.cancelAction)
                .hidden()
        }
        .padding(12)
    }

    private var field: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Search conversations", text: $query)
                .textFieldStyle(.plain)
            if !query.isEmpty {
                Button { query = "" } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    @ViewBuilder
    private var content: some View {
        let hits = results
        if query.trimmingCharacters(in: .whitespacesAndNewlines).count < 2 {
            hint("Type to search your conversations.")
        } else if hits.isEmpty {
            hint("No matches.")
        } else {
            List(hits) { hit in
                Button { onOpen(hit.id) } label: { row(hit) }
                    .buttonStyle(.plain)
            }
            .listStyle(.plain)
        }
    }

    private func hint(_ text: String) -> some View {
        VStack {
            Spacer()
            Text(text).foregroundStyle(.secondary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private func row(_ hit: Hit) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(hit.title).font(.headline).lineLimit(1)
            Text(hit.snippet)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
        .padding(.vertical, 2)
    }

    private var results: [Hit] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        var hits: [Hit] = []
        if q.count >= 2 {
            for convo in ConversationStore.shared.list {
                if let snippet = SearchView.match(convo, q) {
                    hits.append(Hit(id: convo.id, title: convo.title,
                                    snippet: snippet))
                }
            }
        }
        return hits
    }

    // A short snippet around the first (case-insensitive) match, nil if none.
    private static func match(_ convo: ConversationStore.Convo,
                              _ q: String) -> String? {
        var result: String? = nil
        var texts = [convo.title]
        texts.append(contentsOf: convo.messages.map { m in m.text })
        for text in texts where result == nil {
            if let r = text.range(of: q, options: .caseInsensitive) {
                result = snippet(text, r)
            }
        }
        return result
    }

    private static func snippet(_ text: String,
                                _ range: Range<String.Index>) -> String {
        let pad = 30
        let lo = text.index(range.lowerBound, offsetBy: -pad,
                            limitedBy: text.startIndex) ?? text.startIndex
        let hi = text.index(range.upperBound, offsetBy: pad,
                            limitedBy: text.endIndex) ?? text.endIndex
        var out = String(text[lo ..< hi])
            .replacingOccurrences(of: "\n", with: " ")
        if lo > text.startIndex { out = "\u{2026}" + out }
        if hi < text.endIndex { out += "\u{2026}" }
        return out
    }
}
