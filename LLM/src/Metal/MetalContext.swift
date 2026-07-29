// The Metal device context for the Bonsai GPU backend: one MTLDevice + command
// queue, the build-time-compiled kernel library (metal/Kernels.metal ->
// default.metallib), a pipeline-state cache, and the single no-copy MTLBuffer
// that wraps the whole mmap'd GGUF (tensors are addressed by byte offset into
// it, so no per-tensor buffer and no setBuffer offset-alignment constraint).
// Shared with the CoreML backend's process; the two never touch the same
// buffers.
import Foundation
import Metal

final class MetalContext {
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
    // The whole GGUF mapping as one device buffer (no copy: Apple GPUs share the
    // page-aligned mmap). Weight tensors are (byteOffset) views into it.
    let weights: MTLBuffer

    init(mapBase: UnsafeRawPointer, mapSize: Int) throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw MetalErr.noDevice
        }
        guard let q = dev.makeCommandQueue() else { throw MetalErr.noQueue }
        // default.metallib is built from metal/Kernels.metal (Xcode for the
        // framework target, the MetalBuild plugin for SwiftPM); each build
        // style lands it in a different bundle.
        #if SWIFT_PACKAGE
        let bundle = Bundle.module
        #else
        let bundle = Bundle(for: MetalContext.self)
        #endif
        let lib = try dev.makeDefaultLibrary(bundle: bundle)
        // bytesNoCopy needs a page-aligned pointer and length; mmap returns a
        // page-aligned base, and the file mapping is rounded up to a page, so a
        // page-rounded length stays inside the allocation.
        let pageSize = Int(getpagesize())
        let length = (mapSize + pageSize - 1) / pageSize * pageSize
        let mutableBase = UnsafeMutableRawPointer(mutating: mapBase)
        guard let buf = dev.makeBuffer(
            bytesNoCopy: mutableBase, length: length,
            options: .storageModeShared, deallocator: nil) else {
            throw MetalErr.noBuffer
        }
        device = dev
        queue = q
        library = lib
        weights = buf
        matrixUnits = dev.supportsFamily(.apple7)
    }

    // The kernels written against simdgroup_float8x8 / simdgroup_half8x8.
    // Listed here because it is a property of the MSL, not of any one caller.
    static let matrixKernels: Set<String> = [
        "q2_0_gemm", "q2_0_gemm_mm", "q2_0_gemm_mm_h", "f16w_gemm_mm",
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
        if state == nil {
            guard let fn = library.makeFunction(name: name) else {
                throw MetalErr.noFunction(name)
            }
            let made = try device.makeComputePipelineState(function: fn)
            pipelines[name] = made
            state = made
        }
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
}

enum MetalErr: Error {
    case noDevice, noQueue, noBuffer
    case noFunction(String)
}

// f32 view over a shared MTLBuffer's contents (CPU side of unified memory).
extension MTLBuffer {
    func f32(_ count: Int) -> UnsafeMutableBufferPointer<Float> {
        UnsafeMutableBufferPointer(
            start: contents().assumingMemoryBound(to: Float.self), count: count)
    }
}
