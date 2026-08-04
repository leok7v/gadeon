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
            Button(model.canAttachVideo ? "Photos & Videos"
                                        : "Photo Library") {
                showPhotos = true
            }
            Button("Files") { showImporter = true }
        } label: {
            Image(systemName: model.attachGlyph)
                .frame(width: 22, height: 22)
        }
        .buttonStyle(.plain)
        .foregroundStyle(model.hasAttachments ? Color.accentColor
                                              : Color.secondary)
        .disabled(model.busy || !model.canAttachImages)
        .photosPicker(isPresented: $showPhotos, selection: $photos,
                      maxSelectionCount: ChatModel.maxImages,
                      matching: model.canAttachVideo
                          ? .any(of: [.images, .videos]) : .images)
        .fileImporter(isPresented: $showImporter,
                      allowedContentTypes: model.attachableTypes,
                      allowsMultipleSelection: true) { result in
            if case let .success(urls) = result {
                model.handleDrop(urls, at: model.caret)
            }
        }
        .onChange(of: photos) { _, items in load(items) }
    }

    // Picked library items -> image Data -> the model's pending attachments.

    // A picked VIDEO must not be handed to attachImage: with videos now
    // selectable, its bytes would arrive as Data like a photo's and be
    // patchified as a still. The item's own supported types say which it is,
    // and a movie is copied out to a URL because a clip is far too large to
    // carry as Data for the life of the conversation.
    private func load(_ items: [PhotosPickerItem]) {
        if !items.isEmpty {
            Task { @MainActor in
                for item in items {
                    let movie = item.supportedContentTypes.contains { t in
                        t.conforms(to: .movie)
                    }
                    if movie {
                        if let url = try? await item.loadTransferable(
                            type: PickedMovie.self) {
                            model.attachClip(url.url, isVideo: true,
                                             at: model.caret)
                        }
                    } else if let data = try? await item.loadTransferable(
                        type: Data.self) {
                        model.attachImage(data, name: "image", at: model.caret)
                    }
                }
                photos = []
            }
        }
    }
}

// PhotosPicker hands a movie over as a file it owns and then deletes; this
// copies it somewhere with our lifetime so the decode at send still finds it.
// Transferable is the only way to receive one -- loadTransferable(type:
// URL.self) yields a path that is already gone by the time it is read.
struct PickedMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .movie) { received in
            let copy = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + "-"
                                        + received.file.lastPathComponent)
            try? FileManager.default.removeItem(at: copy)
            try FileManager.default.copyItem(at: received.file, to: copy)
            return PickedMovie(url: copy)
        }
    }
}
