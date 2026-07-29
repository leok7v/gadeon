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
            Image(systemName: model.attachedImages.isEmpty ? "plus"
                                                           : "photo.fill")
                .frame(width: 22, height: 22)
        }
        .buttonStyle(.plain)
        .foregroundStyle(model.attachedImages.isEmpty ? Color.secondary
                                                      : Color.accentColor)
        .disabled(model.busy || !model.modelSupportsVision)
        .help(model.modelSupportsVision ? "Attach an image"
                                        : "This model cannot see images")
        .fileImporter(isPresented: $showImporter,
                      allowedContentTypes: [.image],
                      allowsMultipleSelection: true) { result in
            if case let .success(urls) = result {
                for url in urls {
                    let scoped = url.startAccessingSecurityScopedResource()
                    if let data = try? Data(contentsOf: url) {
                        model.attachImage(data, name: url.lastPathComponent,
                                          at: model.caret)
                    }
                    if scoped { url.stopAccessingSecurityScopedResource() }
                }
            }
        }
    }
}
