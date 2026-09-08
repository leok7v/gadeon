import Foundation

final class MetalStopSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var raised = false
    func raise() { lock.lock(); raised = true; lock.unlock() }
    func clear() { lock.lock(); raised = false; lock.unlock() }
    var raisedNow: Bool {
        lock.lock()
        defer { lock.unlock() }
        return raised
    }
}

public final class BackgroundGate: @unchecked Sendable {
    public static let shared = BackgroundGate()
    private let lock = NSLock()
    private var backgrounded = false

    // Set from the app's scenePhase (iOS): true when not .active.
    public func setBackgrounded(_ v: Bool) {
        lock.lock(); backgrounded = v; lock.unlock()
    }

    private var isBackgrounded: Bool {
        lock.lock(); defer { lock.unlock() }; return backgrounded
    }

    public var parked: Bool { isBackgrounded }

    // Park the calling thread while the app is backgrounded, so no GPU submit
    // fires there. A 1s poll (mirrors the reference im.ai gate); returns at once
    // in the foreground, and always at once on macOS (the flag is never raised).
    func waitForForeground() {
        while isBackgrounded { Thread.sleep(forTimeInterval: 1) }
    }
}
