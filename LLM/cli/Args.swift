import Foundation

// Command-line helpers for main.swift: stderr output and the flag prologue.

func err(_ s: String) { FileHandle.standardError.write(Data(s.utf8)) }

func isModelFile(_ path: String) -> Bool {
    path.hasSuffix(".ggxf") || path.hasSuffix(".gguf")
}

// One in-place download progress line: the leading '\r' returns to column 0
// so each report overwrites the last instead of scrolling. The path keeps its
// directory (a set holds dozens of identically-named model.mil / weight.bin)
// and is clipped from the LEFT, which is the end that repeats; the field is a
// fixed width so a short line can never leave a long one's tail on screen.

func progressLine(_ done: Int64, _ total: Int64, _ file: String) -> String {
    let gb = 1_000_000_000.0
    let width = 50
    let pct = total > 0 ? Double(done) * 100 / Double(total) : 0
    var name = file
    if name.count > width { name = "..." + String(name.suffix(width - 3)) }
    return String(format: "\r  %5.1f%%  %.2f / %.2f GB  %@", pct,
                  Double(done) / gb, Double(total) / gb,
                  name.padding(toLength: width, withPad: " ", startingAt: 0))
}
