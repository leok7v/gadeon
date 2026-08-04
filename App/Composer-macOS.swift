import SwiftUI
import UniformTypeIdentifiers

// The in-card model picker renders as a borderless label + chevron on
// macOS. .borderlessButton is a macOS-only menu style, so it lives in an
// SDK-split modifier (matched symbol-for-symbol by Composer-iOS.swift)
// rather than a #if os() branch in the shared Composer.

struct ComposerMenuStyle: ViewModifier {
    func body(content: Content) -> some View {
        content.menuStyle(.borderlessButton)
    }
}

// [+] attaches one image for the next turn via the Files / open panel. iOS
// (Composer-iOS.swift) adds the photo library alongside it.

struct AttachButton: View {
    let model: ChatModel
    @State private var showImporter = false

    var body: some View {
        Button { showImporter = true } label: {
            Image(systemName: model.attachGlyph)
                .frame(width: 22, height: 22)
        }
        .buttonStyle(.plain)
        .foregroundStyle(model.hasAttachments ? Color.accentColor
                                              : Color.secondary)
        .disabled(model.busy || !model.canAttachImages)
        .help(model.attachHelp)
        .fileImporter(isPresented: $showImporter,
                      allowedContentTypes: model.attachableTypes,
                      allowsMultipleSelection: true) { result in
            if case let .success(urls) = result {
                model.handleDrop(urls, at: model.caret)
            }
        }
    }
}
