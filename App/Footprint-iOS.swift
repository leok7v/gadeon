import Foundation

// How much footprint is left before this process is jetsammed. iOS answers it
// directly, and it is the only number that says whether a device is close to
// the edge -- footprint alone cannot, because the limit varies by device and
// by what else is resident.
extension Footprint {
    static func availableBytes() -> UInt64? {
        UInt64(os_proc_available_memory())
    }
}
