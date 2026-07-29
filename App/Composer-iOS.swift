import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

// iOS counterpart of the macOS borderless picker style. .borderlessButton
// is macOS-only; the default menu style already renders the custom label as
// an inline tappable control, so this is the identity that keeps the shared
// Composer compiling on iOS. The iOS layout pass will refine it later.

struct ComposerMenuStyle: ViewModifier {
    func body(content: Content) -> some View {
        content.menuStyle(.automatic)
    }
}

// [+] on iOS attaches one image for the next turn from the photo library
// (PhotosPicker -- privacy-preserving, no permission prompt) or the Files app.

struct AttachButton: View {
    let model: ChatModel
    @State private var showImporter = false
    @State private var showPhotos = false
    @State private var photos: [PhotosPickerItem] = []

    // A PhotosPicker placed directly as a Menu item never presents (SwiftUI
    // only renders plain buttons in a menu), so a menu Button arms the picker
    // through the .photosPicker(isPresented:) modifier instead.
    var body: some View {
        Menu {
            Button("Photo Library") { showPhotos = true }
            Button("Files") { showImporter = true }
        } label: {
            Image(systemName: model.attachedImages.isEmpty ? "plus"
                                                           : "photo.fill")
                .frame(width: 22, height: 22)
        }
        .buttonStyle(.plain)
        .foregroundStyle(model.attachedImages.isEmpty ? Color.secondary
                                                      : Color.accentColor)
        .disabled(model.busy || !model.modelSupportsVision)
        .photosPicker(isPresented: $showPhotos, selection: $photos,
                      maxSelectionCount: ChatModel.maxImages,
                      matching: .images)
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
        .onChange(of: photos) { _, items in load(items) }
    }

    // Picked library items -> image Data -> the model's pending attachments.

    private func load(_ items: [PhotosPickerItem]) {
        if !items.isEmpty {
            Task { @MainActor in
                for item in items {
                    if let data = try? await item.loadTransferable(
                        type: Data.self) {
                        model.attachImage(data, name: "image", at: model.caret)
                    }
                }
                photos = []
            }
        }
    }
}
