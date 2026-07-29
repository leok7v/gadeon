import Foundation

// An attachment shows both as a chip above the composer and as an inline
// "@name" reference in the prompt text, bracketed by two invisible sentinels so
// it renders as plain text yet stays a detectable, removable unit. Deleting the
// reference (chip X or editing it out of the text) drops the attachment; at send
// each doc reference is replaced in place by the file's content.

enum AttachmentRefs {
    // Default-ignorable code points: invisible in the field, near-impossible in
    // real input, so they delimit a reference without showing.
    private static let open: Character = "\u{2063}"    // invisible separator
    private static let close: Character = "\u{2064}"   // invisible plus

    static func token(_ name: String) -> String {
        "\(open)@\(name)\(close)"
    }

    // Insert a reference for `name` into `text` at UTF-16 offset `offset`,
    // padded with a space on each side that is not already whitespace, so it
    // neither fuses onto a neighbouring word nor doubles an existing space.
    // Returns the new text and the caret (UTF-16) just after the reference.

    static func insert(_ name: String, into text: String, at offset: Int)
        -> (text: String, caret: Int) {
        let clamped = max(0, min(offset, text.utf16.count))
        let at = String.Index(utf16Offset: clamped, in: text)
        let before = at > text.startIndex ? text[text.index(before: at)] : " "
        let after = at < text.endIndex ? text[at] : "x"
        let lead = " \n".contains(before) ? "" : " "
        let trail = " \n".contains(after) ? "" : " "
        let piece = lead + token(name) + trail
        var out = text
        out.insert(contentsOf: piece, at: at)
        return (out, clamped + piece.utf16.count)
    }

    // The name of every complete reference span, in order of appearance.

    static func names(in text: String) -> [String] {
        var result: [String] = []
        var pending: String? = nil
        for ch in text {
            if ch == open {
                pending = ""
            } else if ch == close {
                if let inner = pending {
                    result.append(String(inner.drop(while: { c in c == "@" })))
                }
                pending = nil
            } else if pending != nil {
                pending?.append(ch)
            }
        }
        return result
    }

    // The text with the sentinels removed, leaving the human-readable "@name"
    // references -- what the transcript shows.

    static func stripped(_ text: String) -> String {
        String(text.filter { c in c != open && c != close })
    }

    // Remove every trace of a reference to `name`: the whole token, and a
    // broken remnant (one stray sentinel still on "@name", left when the user
    // edits into the reference). A fully sentinel-less "@name" is real typed
    // text and is left alone.

    static func scrub(_ name: String, from text: String) -> String {
        var out = text
        for variant in [token(name), "\(open)@\(name)", "@\(name)\(close)"] {
            out = out.replacingOccurrences(of: variant, with: "")
        }
        return out
    }

    // The prompt to send: each complete span becomes replace(name); plain text
    // passes through and stray sentinels are dropped. A dangling open (mid-edit)
    // emits its buffered text verbatim.

    static func substitute(_ text: String,
                           _ replace: (String) -> String) -> String {
        var out = ""
        var pending: String? = nil
        for ch in text {
            if ch == open {
                pending = ""
            } else if ch == close {
                if let inner = pending {
                    out += replace(String(inner.drop(while: { c in c == "@" })))
                }
                pending = nil
            } else if pending != nil {
                pending?.append(ch)
            } else {
                out.append(ch)
            }
        }
        if let inner = pending { out += inner }
        return out
    }
}
