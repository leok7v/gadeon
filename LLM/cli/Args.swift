import Foundation

// Command-line helpers for main.swift's flag prologue.

func err(_ s: String) { FileHandle.standardError.write(Data(s.utf8)) }

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
