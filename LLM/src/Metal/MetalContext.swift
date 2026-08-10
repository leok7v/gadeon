// The Metal device context for the Bonsai GPU backend: one MTLDevice + command
// queue, the build-time-compiled kernel library (metal/Kernels.metal ->
// default.metallib), a pipeline-state cache, and the no-copy MTLBuffers that
// wrap the mmap'd GGUF (tensors are addressed by byte offset into a window, so
// no per-tensor buffer and no setBuffer offset-alignment constraint). Shared
// with the CoreML backend's process; the two never touch the same buffers.
import Foundation
import Metal

// Where a weight lives on the GPU: the window buffer holding it and its byte
// offset inside that buffer. Only MetalContext.window builds one, so a
// dispatch cannot bind one window and address into another -- above this seam
// an offset is FILE-relative and means nothing to a kernel.
struct WeightRef {
    let buf: MTLBuffer
    let local: UInt64

    // Advancing to a row or a slice INSIDE a tensor stays in the same window,
    // because a tensor is wholly inside the one its base resolves to.
    static func + (base: WeightRef, delta: UInt64) -> WeightRef {
        WeightRef(buf: base.buf, local: base.local + delta)
    }
}

// PUBLIC so a caller outside the module can HOLD one and hand it on; every
// member stays internal, so it can only be obtained from the engine that owns
// the mapping. A second context over the same file would recompute the same
// cut, re-walk the straddle precondition, and hand out WeightRefs into
// different buffers over the identical bytes.
public final class MetalContext {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]
    // Whether this GPU has the matrix units the simdgroup_float8x8 /
    // simdgroup_half8x8 kernels compile against (q2_0_gemm_mm and friends,
    // f16w_gemm_mm). Apple7 is the A14 / M1 generation; on anything older --
    // an A13 iPhone SE -- building one of those pipelines takes the shader
    // compiler service down (XPC_ERROR_CONNECTION_INTERRUPTED) rather than
    // returning an error, so the paths that encode them must be skipped
    // entirely. Every seq=1 decode kernel is plain SIMD and runs anywhere.
    let matrixUnits: Bool

    private struct Window {
        let buf: MTLBuffer
        let start: Int
        let end: Int
    }
    private let windows: [Window]

    // How many buffers the mapping is split across, at minimum.
    //
    // maxBufferLength is a HARD per-device ceiling (2048 MB on an A14), and a
    // 2.5 GB set in one buffer cannot be mapped there on any amount of free
    // memory. Splitting is therefore not optional. Splitting UNCONDITIONALLY,
    // at min(that ceiling, total/N), is what makes it testable: on every
    // machine this repo runs, the total/N term is the smaller one, so a Mac
    // with a 12 GB buffer limit walks the SAME boundaries as an A14 with 2 GB.
    // The goldens captured on the Mac then gate the phone's path instead of
    // standing in for it.
    //
    // N is FINE rather than the 4 the ceiling alone needs, because referencing
    // one tensor makes its WHOLE window resident: coarse windows drag
    // unrelated weights along with every dispatch. Computed over
    // gemma-4-E2B's own tensor offsets (2541 MB, 35 layers, both embedding
    // tables CPU-gathered so their windows are never bound), the bytes one
    // command buffer REFERENCES are --
    //
    //     windows | whole chunk | 5 layers | 1 layer per buffer
    //           4 |      993 MB |   993 MB |             993 MB
    //          32 |      786 MB |   540 MB |                   -
    //          64 |      714 MB |   298 MB |              78 MB
    //
    // -- at 4 the layers span two windows totalling 993 MB and no split can
    // go below that, since L30's own tensors sit in both.
    //
    // READ THAT AS REFERENCES, NOT AS RESIDENCY. It is what the driver is
    // ASKED to make resident, and the driver does not oblige per buffer:
    // MEASURED on macOS by sampling vm_stat's wired count around a prefill,
    // 4 reps each, the delta is 1008 MB at one layer per buffer against
    // 1090 MB for the whole chunk on one -- i.e. it stays at the ~993 MB
    // every bindable window sums to, whichever way the work is cut. So the
    // fine cut is nearly free and buys nearly nothing HERE. It is kept
    // because a device under real pressure is a different regime (iOS was
    // measured releasing 1859 -> 801 MB the moment a prefill ended) and this
    // is the only lever that could pay there.
    // The buffers are views of one read-only mapping, so 32 of them cost 32
    // objects and nothing else.
    //
    // Env-readable because a 4.3% prefill drop was once attributed to
    // splitting the mapping and never A/B'd. This settles it: 1 is the unsplit
    // mapping, wherever maxBufferLength allows one. MEASURED on gemma-4-E2B,
    // ABBA with a cooldown, pp521 t/s over 4 reps each -- 1 window 425.0,
    // 4 windows 423.8, 64 windows 422.6. So the split costs 0.6% at the very
    // most, against a 0.7-1.1% spread WITHIN each group, and the 4.3% was
    // never its doing. 0.6% of prefill for 993 -> 78 MB of residency is not a
    // trade worth thinking about twice.
    static let minWindows: Int = {
        let raw = ProcessInfo.processInfo.environment["LLM_METAL_WINDOWS"]
        return max(1, raw.flatMap { v in Int(v) } ?? 64)
    }()

    init(_ gguf: GGUF) throws {
        let dev = MTLCreateSystemDefaultDevice()
        let q = dev?.makeCommandQueue()
        let limit = dev?.maxBufferLength ?? 0
        let ranges = MetalContext.cuts(gguf, limit: limit)
        let widest = ranges.map { r in r.count }.max() ?? 0
        let mapped = MetalContext.mapped(gguf, ranges, dev, limit)
        if let fault = MetalContext.fault(dev, q, widest: widest,
                                          limit: limit, mapped: mapped.count,
                                          want: ranges.count) {
            throw fault
        }
        device = dev!
        queue = q!
        library = try device.makeDefaultLibrary(bundle: MetalContext.bundle)
        windows = mapped
        matrixUnits = device.supportsFamily(.apple7)
        // The invariant every dispatch leans on, checked once per context
        // rather than trusted. Without it a mis-cut is silent: the goldens
        // touch a handful of tensors and would still MATCH while the forward
        // read a straddling one off the end of its buffer.
        for t in gguf.tensors.values {
            let r = window(UInt64(t.base - gguf.map))
            precondition(Int(r.local) + t.byteCount <= r.buf.length,
                         "\(t.name) straddles a weight window")
        }
    }

    // default.metallib is built from metal/Kernels.metal (Xcode for the
    // framework target, the MetalBuild plugin for SwiftPM); each build style
    // lands it in a different bundle.
    private static var bundle: Bundle {
        #if SWIFT_PACKAGE
        return Bundle.module
        #else
        return Bundle(for: MetalContext.self)
        #endif
    }

    // The window byte ranges. No window SPLITS A TENSOR: every kernel reads a
    // whole tensor (often a row deep inside one) through a single binding, so
    // a straddling tensor could not be addressed at all.
    //
    // Windows therefore OVERLAP by up to a page. A GGUF's general.alignment
    // can be as small as 32 -- the ternary files use it, and not one of their
    // tensors starts on a page -- while bytesNoCopy needs page-aligned bounds,
    // so a window rounds OUTWARD at both ends to hold whole tensors and the
    // same page can end one window and begin the next. They are views of one
    // read-only mapping, so the duplication costs nothing.
    //
    // Greedy: keep filling the open window until the next tensor would carry
    // it past the target, then start the next at that tensor's page. A tensor
    // bigger than the target on its own (gemma's 1260 MB per-layer table
    // against a 40 MB target) cannot be cut down, so the `> open` test leaves
    // it a window of its own rather than looping.
    //
    // `last > open` says the open window HOLDS something, and it is what stops
    // the very first tensor from cutting an EMPTY window ahead of itself: the
    // file starts with a header, so open is 0 while the first tensor begins
    // past it, and a first tensor wider than the target satisfies every other
    // term. Metal returns nil for a zero-length bytesNoCopy buffer, so the
    // whole context then failed to build with "Metal buffer allocation
    // failed" and nothing pointing at the cut.
    //
    // mmap returns a page-aligned base and the file mapping is rounded up to a
    // page, so a page-rounded total stays inside the allocation.
    private static func cuts(_ g: GGUF, limit: Int) -> [Range<Int>] {
        let page   = Int(getpagesize())
        let total  = (g.mapSize + page - 1) / page * page
        let target = min(limit, (total + minWindows - 1) / minWindows)
        let spans = g.tensors.values
            .map { t in (start: t.base - g.map,
                         end: t.base - g.map + t.byteCount) }
            .sorted { a, b in a.start < b.start }
        var out: [Range<Int>] = []
        var open = 0
        var last = 0
        for s in spans {
            let next = s.start / page * page
            if s.end - open > target && next > open && last > open {
                out.append(open ..< min((last + page - 1) / page * page,
                                        total))
                open = next
            }
            last = max(last, s.end)
        }
        out.append(open ..< total)
        return out
    }

    // One no-copy buffer per range over the mapping, which is only READ here
    // -- the caller owns it and must outlive this context. A range wider than
    // `limit` is skipped rather than asked for: Metal aborts on an oversized
    // bytesNoCopy instead of returning nil, which would take the diagnosis
    // below down with it. A short result is what the caller reads as failure.
    private static func mapped(_ g: GGUF, _ ranges: [Range<Int>],
                               _ dev: MTLDevice?, _ limit: Int) -> [Window] {
        let base = UnsafeMutableRawPointer(mutating: g.map)
        return ranges.compactMap { r in
            let buf = r.count <= limit
                ? dev?.makeBuffer(bytesNoCopy: base + r.lowerBound,
                                  length: r.count,
                                  options: .storageModeShared,
                                  deallocator: nil)
                : nil
            return buf.map { b in
                Window(buf: b, start: r.lowerBound, end: r.upperBound)
            }
        }
    }

    // Which acquisition failed, or nil. `tooBig` outranks `noBuffer` because
    // it names a ceiling no amount of free memory lifts, where noBuffer reads
    // as one that does -- and a window is at least one whole tensor, so a
    // tensor past the ceiling is unmappable however the cuts fall.
    private static func fault(_ dev: MTLDevice?, _ queue: MTLCommandQueue?,
                              widest: Int, limit: Int, mapped: Int,
                              want: Int) -> MetalErr? {
        var out: MetalErr? = nil
        if dev == nil {
            out = .noDevice
        } else if queue == nil {
            out = .noQueue
        } else if widest > limit {
            out = .tooBig(widest, limit)
        } else if mapped < want {
            out = .noBuffer
        }
        return out
    }

    // The window sizes in MB, for the harnesses to report: a golden that
    // MATCHes says nothing about the split unless the run also says how many
    // buffers it used.
    var windowMB: [Int] { windows.map { w in (w.end - w.start) / 1_048_576 } }

    // The buffer holding this FILE byte offset, and the offset inside it.
    // The LAST window that starts at or before `off`, not the first that
    // contains it: where two windows overlap, only the later one is guaranteed
    // to hold the whole tensor.
    func window(_ off: UInt64) -> WeightRef {
        var i = 0
        while i + 1 < windows.count && UInt64(windows[i + 1].start) <= off {
            i += 1
        }
        return WeightRef(buf: windows[i].buf,
                         local: off - UInt64(windows[i].start))
    }

    // The kernels written against simdgroup_float8x8 / simdgroup_half8x8.
    // Listed here because it is a property of the MSL, not of any one caller.
    static let matrixKernels: Set<String> = [
        "q2_0_gemm", "q2_0_gemm_mm", "q2_0_gemm_mm_h", "q8_0_gemm_mm_h",
        "q4_0_gemm_mm_h", "q8_0_gemm_mm", "q4_0_gemm_mm",
        "f16w_gemm_mm", "attn_batch_mm",
    ]

    // Build every pipeline this GPU can, before any encoding starts. After
    // this `pipeline` is a dictionary hit that cannot fail, which is what
    // makes the dispatch surface's non-throwing shape sound: no forward ever
    // compiles a shader mid-encode, so a compiler-service failure (its own
    // crash, or a jetsam under memory pressure) cannot land inside a command
    // encoder where there is nothing to do but die. A failure here throws
    // from the caller's init instead, reaching the app's load-failure screen.
    // Kernels the hardware cannot build are skipped rather than attempted --
    // building one does not return an error, it takes the compiler service
    // down with it.
    func prewarm() throws {
        for name in library.functionNames
        where matrixUnits || !MetalContext.matrixKernels.contains(name) {
            _ = try pipeline(name)
        }
    }

    // A cached compute pipeline for a named kernel.
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        var state = pipelines[name]
        if state == nil, let fn = library.makeFunction(name: name) {
            let made = try device.makeComputePipelineState(function: fn)
            pipelines[name] = made
            state = made
        }
        if state == nil { throw MetalErr.noFunction(name) }
        return state!
    }

    // A shared-storage f32 buffer of `count` elements (unified memory: the CPU
    // reads/writes the same pages the GPU does, so no blit for activations).
    func makeF32(_ count: Int) -> MTLBuffer {
        device.makeBuffer(length: max(count, 1) * MemoryLayout<Float>.stride,
                          options: .storageModeShared)!
    }

    // A shared-storage f32 buffer seeded from a Swift array.
    func makeF32(_ values: [Float]) -> MTLBuffer {
        values.withUnsafeBytes { src in
            device.makeBuffer(bytes: src.baseAddress!,
                              length: max(src.count, 1),
                              options: .storageModeShared)!
        }
    }

    // A shared-storage u32 buffer of `count` elements, zeroed. Metal hands
    // back whatever was in the page, so the clear is the caller's.
    func makeU32(_ count: Int) -> MTLBuffer {
        let b = device.makeBuffer(
            length: max(count, 1) * MemoryLayout<UInt32>.stride,
            options: .storageModeShared)!
        memset(b.contents(), 0, b.length)
        return b
    }

    // The all-empty block table, for every batched attention with no vision
    // blocks: an empty range leaves attn_end at the plain causal bound, so a
    // caller that has none binds this instead of branching in the kernel.
    // Shared because it is read-only and always zero, and grown rather than
    // sized up front because the batch width is the engine's to choose.
    private var blank: MTLBuffer?

    func noBlocks(_ rows: Int) -> MTLBuffer {
        let need = rows * 2 * MemoryLayout<UInt32>.stride
        if blank == nil || blank!.length < need { blank = makeU32(rows * 2) }
        return blank!
    }
}

enum MetalErr: Error, CustomStringConvertible {
    case noDevice, noQueue, noBuffer
    case tooBig(Int, Int)
    case noFunction(String)

    var description: String {
        let out: String
        switch self {
        case .noDevice: out = "no Metal device"
        case .noQueue: out = "no Metal command queue"
        case .noBuffer: out = "Metal buffer allocation failed"
        case let .tooBig(want, cap):
            out = "one weight tensor needs \(want / 1_048_576) MB but this "
                + "GPU maps at most \(cap / 1_048_576) MB in one buffer"
        case let .noFunction(n): out = "no Metal kernel named \(n)"
        }
        return out
    }
}

// f32 view over a shared MTLBuffer's contents (CPU side of unified memory).
extension MTLBuffer {
    func f32(_ count: Int) -> UnsafeMutableBufferPointer<Float> {
        UnsafeMutableBufferPointer(
            start: contents().assumingMemoryBound(to: Float.self), count: count)
    }

    func u32(_ count: Int) -> UnsafeMutableBufferPointer<UInt32> {
        UnsafeMutableBufferPointer(
            start: contents().assumingMemoryBound(to: UInt32.self),
            count: count)
    }
}
