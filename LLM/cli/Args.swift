import Foundation

// Command-line helpers for main.swift: stderr output and the flag prologue.

func err(_ s: String) { FileHandle.standardError.write(Data(s.utf8)) }

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

// Strip "flag value" from args when the flag is present AND `parse` accepts
// the value, returning the parsed value. Otherwise args stay untouched: a
// flag with a missing or unparseable value remains in place, exactly like
// the inline firstIndex/removeSubrange scan it replaces.

func stripValue<T>(_ args: inout [String], _ flag: String,
                   _ parse: (String) -> T?) -> T? {
    let idx = args.firstIndex(of: flag)
    let val = idx.flatMap { i in
        i + 1 < args.count ? parse(args[i + 1]) : nil
    }
    if let i = idx, val != nil { args.removeSubrange(i ... i + 1) }
    return val
}

func stripValue(_ args: inout [String], _ flag: String) -> String? {
    stripValue(&args, flag) { value in value }
}

// Strip "flag a b" (two values), both required for the strip to happen.

func stripPair(_ args: inout [String], _ flag: String)
    -> (String, String)? {
    let idx = args.firstIndex(of: flag)
    let a = idx.flatMap { i in i + 1 < args.count ? args[i + 1] : nil }
    let b = idx.flatMap { i in i + 2 < args.count ? args[i + 2] : nil }
    var result: (String, String)? = nil
    if let i = idx, let a, let b {
        args.removeSubrange(i ... i + 2)
        result = (a, b)
    }
    return result
}
