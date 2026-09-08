import Foundation

// The public seam over the vendored reader in Docs2md.swift, and the ONLY
// file that may adapt it. That file is copied in from a separate effort and
// must stay byte-identical so a re-import is a copy rather than a merge, so
// everything a caller needs is expressed here instead. See
// LLM/fixtures/pdf2md/ORIGIN.md.

public enum Docs2md {
    // SYNCHRONOUS, and a large document is real work: a caller on the main
    // actor must hand this to a detached task the way an attachment encode
    // already does. There is no progress to report -- unlike a PDF these
    // formats state their structure, so there are no pages to count through.

    public static func markdown(of url: URL) throws -> String {
        try DocsConverter().markdown(of: url)
    }

    // What a picker may offer. `htm` and `xhtml` are read as html but are not
    // formats of their own, so they are named here rather than derived.
    public static let readable: [String] =
        Format.allCases.map { format in format.rawValue } + ["htm", "xhtml"]
}
