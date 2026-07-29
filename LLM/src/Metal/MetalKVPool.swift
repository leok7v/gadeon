// Lazy pos-major paged KV cache for one attention layer on the GPU: fixed-size
// pages of P positions (kvDim = headDim*nHeadKV floats each), each a separate
// MTLBuffer allocated on first write -- so context grows toward 256K-1M without
// one giant contiguous buffer and without realloc-copies. The GPU counterpart of
// the SIMD KVCache page pool / the CoreML PagePool.
//
// The paged attention kernel gathers across pages via a bindless page table:
// kAddr/vAddr are small buffers of per-page gpuAddresses that the kernel reads as
// `device const float* const*`. Pages must be made resident (useResources) and a
// barrier ordered after the tail write, since bindless reads are not hazard-
// tracked against the directly-bound append.
//
// Append-only: completed pages are never mutated, so a bookmark shares them by
// reference and copies only the partially-filled tail page (O(tail), not
// O(context)). truncate() rewinds the tail. The GDN recurrence -- which cannot be
// paged -- is snapshotted whole by the engine alongside this.
import Foundation
import Metal

final class MetalKVPool {
    private let device: MTLDevice
    let P: Int
    let kvDim: Int
    private(set) var kPages: [MTLBuffer] = []
    private(set) var vPages: [MTLBuffer] = []
    private(set) var len = 0
    // Bindless page-address tables (the MSL KVTable: KV_MAXP device pointers each,
    // stored as raw gpuAddresses), rebuilt from the current pages before each
    // attention read. Fixed at maxPages -- P * maxPages bounds the context.
    static let maxPages = 2048   // must match KV_MAXP in the shader
    private(set) var kAddr: MTLBuffer
    private(set) var vAddr: MTLBuffer

    init(device: MTLDevice, P: Int, kvDim: Int) {
        self.device = device
        self.P = P
        self.kvDim = kvDim
        let bytes = MetalKVPool.maxPages * 8
        kAddr = device.makeBuffer(length: bytes, options: .storageModeShared)!
        vAddr = device.makeBuffer(length: bytes, options: .storageModeShared)!
    }

    // Pages are HALF. Decode cost is linear in cached positions, so past a few
    // hundred tokens the KV read is most of what a token moves, and on the
    // dense 1.7B the pages themselves (~112 MB per 512 positions at f32) are
    // what bounds context on a 3 GB phone. f32 bought precision the format
    // never needed: llama.cpp defaults to f16 KV, and the CoreML PagePool on
    // the ANE side has always stored fp16.
    private var pageBytes: Int { P * kvDim * MemoryLayout<Float16>.stride }

    private func newPage() -> MTLBuffer {
        device.makeBuffer(length: pageBytes, options: .storageModeShared)!
    }

    // The tail page + element offset to write this position into; allocates a
    // fresh page when the tail lands on a page boundary. Call commitAppend after
    // the write kernel is encoded.
    func tailForAppend() -> (k: MTLBuffer, v: MTLBuffer, slot: Int) {
        if len % P == 0 { kPages.append(newPage()); vPages.append(newPage()) }
        return (kPages[kPages.count - 1], vPages[vPages.count - 1], len % P)
    }
    func commitAppend() { len += 1 }

    // Reserve N positions (allocating pages as the tail crosses boundaries) and
    // refresh the address table, so a batched append/attention kernel can write
    // and read positions basePos..basePos+N-1 through the page table.
    func appendBatch(_ n: Int) {
        for _ in 0..<n { _ = tailForAppend(); commitAppend() }
        refreshTable()
    }

    // Rebuild the bindless address tables from the current pages (cheap: one
    // uint64 per live page). The tables are fixed at maxPages (the shader's
    // KVTable is a fixed KV_MAXP-pointer struct, so they cannot grow); exceeding
    // that many pages is a precondition failure, not a reallocation.
    func refreshTable() {
        let n = kPages.count
        precondition(n <= MetalKVPool.maxPages,
                     "KV context exceeds P * \(MetalKVPool.maxPages) positions")
        let ka = kAddr.contents().assumingMemoryBound(to: UInt64.self)
        let va = vAddr.contents().assumingMemoryBound(to: UInt64.self)
        for i in 0..<n { ka[i] = kPages[i].gpuAddress; va[i] = vPages[i].gpuAddress }
    }

    // Every live page, for the encoder residency call before the attn dispatch.
    var residentPages: [MTLResource] { kPages + vPages }

    // Roll the tail back to `newLength`, dropping now-empty pages (rewind).
    func truncate(to newLength: Int) {
        len = newLength
        let need = (newLength + P - 1) / P
        while kPages.count > need { kPages.removeLast(); vPages.removeLast() }
    }

    // A bookmark: share completed pages (immutable), DUPLICATE the partial tail
    // so the snapshot and the pool never write the same page.
    struct Snapshot: @unchecked Sendable {
        let kPages: [MTLBuffer]
        let vPages: [MTLBuffer]
        let len: Int
    }
    func snapshot() -> Snapshot {
        var kp = kPages, vp = vPages
        if len % P != 0, !kp.isEmpty {
            kp[kp.count - 1] = dup(kp[kp.count - 1])
            vp[vp.count - 1] = dup(vp[vp.count - 1])
        }
        return Snapshot(kPages: kp, vPages: vp, len: len)
    }
    func restore(_ s: Snapshot) {
        kPages = s.kPages
        vPages = s.vPages
        len = s.len
        // Give the pool its own tail so later appends do not mutate the
        // snapshot's tail (a snapshot may be restored more than once).
        if len % P != 0, !kPages.isEmpty {
            kPages[kPages.count - 1] = dup(kPages[kPages.count - 1])
            vPages[vPages.count - 1] = dup(vPages[vPages.count - 1])
        }
    }

    private func dup(_ b: MTLBuffer) -> MTLBuffer {
        let c = device.makeBuffer(length: b.length, options: .storageModeShared)!
        memcpy(c.contents(), b.contents(), b.length)
        return c
    }
}
