import Foundation
import PackagePlugin

// Compiles metal/Kernels.metal into default.metallib for the SwiftPM build
// (the artifact lands in Bundle.module, where MetalContext loads it); the
// Xcode framework target compiles the same file natively instead. macOS SDK
// only: the SwiftPM products (gadeon-cli) are macOS -- the app's iOS build goes
// through Xcode, never this plugin.

@main
struct MetalBuild: BuildToolPlugin {
    func createBuildCommands(context: PluginContext,
                             target: Target) throws -> [Command] {
        let dir = context.package.directoryURL
        let src = dir.appendingPathComponent("metal/Kernels.metal")
        // The two included files are inputs, not translation units: an edit
        // to either has to re-run this command.
        let inc = ["metal/IQTables.metal", "metal/IQKernels.metal",
                   "metal/AssistKernels.metal"]
            .map { dir.appendingPathComponent($0) }
        let out = context.pluginWorkDirectoryURL
            .appendingPathComponent("default.metallib")
        return [.buildCommand(
            displayName: "metal Kernels.metal -> default.metallib",
            executable: URL(fileURLWithPath: "/usr/bin/xcrun"),
            arguments: ["-sdk", "macosx", "metal", "-O2",
                        "-o", out.path, src.path],
            inputFiles: [src] + inc,
            outputFiles: [out])]
    }
}
