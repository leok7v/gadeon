import Foundation

// The public seam over the vendored converter in Pdf2md.swift, and the ONLY
// file that may adapt it. That file is copied in from a separate effort and
// must stay byte-identical so a re-import is a copy rather than a merge, so
// everything a caller needs is expressed here instead. See
// LLM/fixtures/pdf2md/ORIGIN.md.

public enum Pdf2md {
    // `.geometry` rather than `.vision`: it reads the page's OWN text layer,
    // whose word boxes are exact where recognition only estimates them, and it
    // falls back to recognizing the page by itself when the layer explains too
    // little of the ink. So a born-digital PDF costs no OCR and a scan still
    // works, without a caller having to know which it holds.
    //
    // `onPage` reports each page as it lands, which is the only progress a
    // conversion can offer: a long document is minutes of work, and the
    // converter has no other moment to speak.

    public static func markdown(of url: URL,
                                onPage: (@Sendable (Int) -> Void)? = nil)
        async throws -> String {
        var converter = Converter()
        converter.mode = .geometry
        if let onPage {
            converter.onAudit = { page, _ in onPage(page) }
        }
        return try await converter.markdown(of: url)
    }
}
