import Foundation

// A diagnostics sink the host app wires (App -> the shared Diag), so MD render
// timing lands in the same pullable log. nil (default) costs nothing, and the
// timing is frame-budget gated so steady rendering stays silent -- only the
// stalls surface.

public enum MarkdownDiag {

    nonisolated(unsafe) public static var report: (@Sendable (String) -> Void)?

    // Time `body`, reporting to the host sink only when it overruns `over` ms
    // (a dropped frame). The label is an autoclosure so its string is built
    // only when the span is actually slow.
    @discardableResult
    static func timed<T>(_ label: @autoclosure () -> String,
                         over: Double = 12, _ body: () -> T) -> T {
        let t0 = DispatchTime.now()
        let result = body()
        if let report {
            let ms = Double(DispatchTime.now().uptimeNanoseconds
                - t0.uptimeNanoseconds) / 1e6
            if ms >= over {
                report(String(format: "[md] %.1fms %@", ms, label()))
            }
        }
        return result
    }
}
