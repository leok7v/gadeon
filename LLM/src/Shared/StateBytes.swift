import Foundation

// The wire format a parked conversation is written in: little-endian Int64
// counts and f32 payloads, so a state file means the same thing whatever an
// engine stores internally (the GPU pools are half, the CPU ones f32).
//
// One definition because three engines write it. A file is only ever read
// back by the same model -- ChatSession stamps the prompt it was cooked for
// and the path carries the model name -- so this carries no geometry of its
// own.

enum StateBytes {

    // A stale file from a build whose serializeState was the protocol's
    // empty-Data default would otherwise deserialize to pos 0 with no KV and
    // SUCCEED -- the session would believe it was primed and run with no
    // system prefix at all. So a state file names itself, and anything that
    // does not is refused and re-cooked.
    static let magic: [UInt8] = Array("GDNS".utf8)
    static let version = 1

    static func putHeader(_ out: inout Data) {
        out.append(contentsOf: magic)
        putInt(&out, version)
    }

    // Consumes the header when it is ours, leaving `p` on the payload.
    static func readHeader(_ b: [UInt8], _ p: inout Int) -> Bool {
        var out = false
        if b.count >= magic.count + 8, Array(b[0..<magic.count]) == magic {
            p = magic.count
            out = getInt(b, &p) == version
        }
        return out
    }

    static func putInt(_ out: inout Data, _ v: Int) {
        var x = Int64(v).littleEndian
        withUnsafeBytes(of: &x) { out.append(contentsOf: $0) }
    }

    static func putFloats(_ out: inout Data, _ v: [Float]) {
        putInt(&out, v.count)
        v.withUnsafeBufferPointer { b in
            out.append(UnsafeBufferPointer(start: b.baseAddress, count: b.count)
                .withMemoryRebound(to: UInt8.self) { raw in
                    Data(buffer: raw)
                })
        }
    }

    static func getInt(_ b: [UInt8], _ p: inout Int) -> Int {
        var x: Int64 = 0
        withUnsafeMutableBytes(of: &x) { dst in
            for i in 0..<8 where p + i < b.count { dst[i] = b[p + i] }
        }
        p += 8
        return Int(Int64(littleEndian: x))
    }

    static func getFloats(_ b: [UInt8], _ p: inout Int) -> [Float] {
        let n = getInt(b, &p)
        var out = [Float](repeating: 0, count: max(n, 0))
        if n > 0 && p + n * 4 <= b.count {
            out.withUnsafeMutableBytes { dst in
                for i in 0..<(n * 4) { dst[i] = b[p + i] }
            }
        }
        p += max(n, 0) * 4
        return out
    }
}
