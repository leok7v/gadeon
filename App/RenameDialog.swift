import SwiftUI

// Centred over everything rather than edited in the row: on a phone the soft
// keyboard takes half the screen and an in-place editor lands under it, so
// the dialog is the one shape that needs no scrolling to stay visible.

struct RenameDialog: View {

    let title: String
    let onCommit: (String) -> Void
    let onCancel: () -> Void

    @State private var text = ""

    var body: some View {
        ZStack {
            Color.black.opacity(0.35)
                .ignoresSafeArea()
                .contentShape(Rectangle())
                .onTapGesture(perform: onCancel)
            card
        }
        .onAppear { text = title }
    }

    private var card: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Rename Conversation")
                .appFont(.headline)
            RenameField(text: $text, onSubmit: commit, onCancel: onCancel)
                .frame(height: 22)
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(.quinary, in: RoundedRectangle(cornerRadius: 7))
                .overlay {
                    RoundedRectangle(cornerRadius: 7)
                        .stroke(.separator, lineWidth: 0.5)
                }
            HStack(spacing: 10) {
                Spacer()
                Button("Cancel", action: onCancel)
                    .keyboardShortcut(.cancelAction)
                Button("Rename", action: commit)
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(maxWidth: 380)
        .background(.regularMaterial,
                    in: RoundedRectangle(cornerRadius: 14))
        .overlay {
            RoundedRectangle(cornerRadius: 14)
                .stroke(.separator, lineWidth: 0.5)
        }
        .padding(24)
    }

    // An empty name keeps the old one, which is the same answer Esc gives.

    private func commit() {
        let name = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if name.isEmpty { onCancel() } else { onCommit(name) }
    }
}
