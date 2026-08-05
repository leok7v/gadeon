import Foundation
import MD
import SwiftUI
import UniformTypeIdentifiers

// A prepared export's bytes for SwiftUI's .fileExporter (the macOS Save
// panel). Write-only: the transcript is never read back from a file.
private struct ExportFile: FileDocument {
    static let readableContentTypes: [UTType] = []
    static let writableContentTypes: [UTType] = [.pdf, .html]
    let data: Data

    init(data: Data) { self.data = data }
    init(configuration: ReadConfiguration) throws { data = Data() }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}

// The transcript actions as a row of title-bar glyph buttons, revealed by the
// trailing chevron: Plain Text toggle, Find, Copy, Share, (macOS) Download,
// and For Nerds. Share and Download are the only dropdowns (PDF|HTML), so no
// per-format chevrons. A row of glyphs reads better in the bar than a single
// nested menu.

struct TranscriptActions: View {

    let document: Markdown.Document
    let title: String
    @Binding var renderMarkdown: Bool
    let onFind: () -> Void
    // nil hides the trace button entirely. It rides with the status line:
    // both are the diagnostic surface, and offering the numbers is the same
    // decision as offering what produced them.
    let onDebug: (() -> Void)?
    @State private var pdfURL: URL?
    @State private var htmlURL: URL?
    // The Save panel (macOS): the picked format's bytes + type + suggested
    // name, armed by the Download menu.
    @State private var exportFile: ExportFile?
    @State private var exportType: UTType = .pdf
    @State private var showExporter = false
    // Brief checkmark after a copy, matching the in-block copy buttons.
    @State private var copied = false
    @State private var copiedReset: Task<Void, Never>?

    var body: some View {
        HStack(spacing: 14) {
            Button { renderMarkdown.toggle() } label: {
                Image(systemName: renderMarkdown ? "doc.plaintext"
                                                 : "doc.richtext")
            }
            .help(renderMarkdown ? "Show as plain text" : "Show as Markdown")
            Button(action: onFind) {
                Image(systemName: "magnifyingglass")
            }
            .help("Find in chat")
            Button(action: copy) {
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
            }
            .help("Copy transcript")
            Menu {
                shareRow(pdfURL, kind: "PDF")
                shareRow(htmlURL, kind: "HTML")
            } label: {
                Image(systemName: "square.and.arrow.up")
            }
            .menuIndicator(.hidden)
            .help("Share as PDF or HTML")
            // Download = save the file to a chosen location, distinct from the
            // system share sheet. macOS only: on iOS the Share sheet already
            // offers Save to Files, so a second control would be redundant.
            if !isOS {
                Menu {
                    saveRow("Save as PDF", pdfURL, .pdf)
                    saveRow("Save as HTML", htmlURL, .html)
                } label: {
                    Image(systemName: "square.and.arrow.down")
                }
                .menuIndicator(.hidden)
                .help("Save as PDF or HTML")
            }
            // Last, and deliberately behind the chevron: the trace is a
            // developer surface, and it belongs with the other things done TO
            // a transcript rather than beside Settings, which is where a user
            // goes to change how the app behaves.
            if let onDebug {
                Button(action: onDebug) {
                    Image(systemName: "ladybug")
                }
                .help("For Nerds")
            }
        }
        .task(id: document) { await prepareExports() }
        .fileExporter(isPresented: $showExporter, document: exportFile,
                      contentType: exportType,
                      defaultFilename: sanitizedFilename(title)) { _ in }
    }

    // A Save row for one format; disabled until its export has been prepared.
    @ViewBuilder
    private func saveRow(_ label: String, _ url: URL?,
                         _ type: UTType) -> some View {
        Button(label) { beginSave(url, type) }
            .disabled(url == nil)
    }

    private func beginSave(_ url: URL?, _ type: UTType) {
        if let url, let data = try? Data(contentsOf: url) {
            exportFile = ExportFile(data: data)
            exportType = type
            showExporter = true
        }
    }

    @ViewBuilder
    private func shareRow(_ url: URL?, kind: String) -> some View {
        if let url {
            ShareLink(item: url) {
                Label("Share as \(kind)", systemImage: "square.and.arrow.up")
            }
        } else {
            Button { } label: {
                Label("Preparing \(kind)\u{2026}",
                      systemImage: "square.and.arrow.up")
            }
            .disabled(true)
        }
    }

    private func copy() {
        setClipboard(Markdown.plainText(document))
        copied = true
        copiedReset?.cancel()
        copiedReset = Task { @MainActor in
            try? await Task.sleep(for: .seconds(1.2))
            if !Task.isCancelled { copied = false }
        }
    }

    private func prepareExports() async {
        let safe = sanitizedFilename(title)
        async let pdf = MarkdownPDF.export(document, title: title)
        async let html = Markdown.htmlPrefetching(document, title: title)
        let (pdfData, htmlText) = await (pdf, html)
        pdfURL = pdfData.flatMap { data in
            writeTemp(data, name: "\(safe).pdf")
        }
        htmlURL = writeTemp(Data(htmlText.utf8), name: "\(safe).html")
    }

    private func sanitizedFilename(_ s: String) -> String {
        let base = s.isEmpty ? "Conversation" : s
        return base.replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: ":", with: "_")
    }

    private func writeTemp(_ data: Data, name: String) -> URL? {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("TranscriptTools", isDirectory: true)
        var result: URL?
        do {
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
            let url = dir.appendingPathComponent(name)
            try data.write(to: url, options: .atomic)
            result = url
        } catch {
            result = nil
        }
        return result
    }
}
