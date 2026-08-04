// swift-tools-version: 6.0
import PackageDescription

// LLM is the pure-Swift engine (tokenizer + chat template + sampler + the
// backend-agnostic ChatSession seam) with per-backend compute under
// src/{Shared,CoreML,Metal,SIMD,Slugs}. gadeon-cli (cli/) is a macOS
// command-line harness for driving and testing it without the UI. The SwiftUI
// app links LLM from the Xcode project; see config/platform.xcconfig.
// Binds cblas_sgemm$NEWLAPACK. Without the define Swift binds the LEGACY
// cblas_sgemm -- a DIFFERENT entry point, and nothing guarantees the two
// agree in the last bits at every shape. The reference C the speech engine is
// compared against is built with it, so the comparison is only meaningful
// while this matches.
let accelerateNewLapack = ["-Xcc", "-DACCELERATE_NEW_LAPACK"]

let package = Package(
    name: "gadeon",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "LLM", targets: ["LLM"]),
        .executable(name: "gadeon-cli", targets: ["gadeon-cli"]),
    ],
    targets: [
        .target(
            name: "LLM",
            path: "src",
            // The Slugs MiniLM+index model and the TTS speech set ship INSIDE
            // the package so every consumer (app framework, SwiftPM clients)
            // resolves one copy -- WikiSlugs.bundledModel and Speech.bundled
            // find them in either bundle form.
            resources: [
                .copy("Slugs/minilm.gguf"),
                .copy("TTS/kitten_full.gguf"),
                .copy("TTS/voices.safetensors"),
                .copy("TTS/en_rules"),
                .copy("TTS/en_list"),
            ],
            swiftSettings: [.unsafeFlags(accelerateNewLapack)],
            plugins: ["MetalBuild"]
        ),
        .plugin(
            name: "MetalBuild",
            capability: .buildTool(),
            path: "plugins/MetalBuild"
        ),
        .executableTarget(
            name: "gadeon-cli",
            dependencies: ["LLM"],
            path: "cli",
            exclude: ["Info.plist"],
            // Embed an Info.plist so NSBundle.mainBundle.bundleIdentifier is set —
            // this is the namespace the ANE e5rt cache uses (else it falls back to
            // the executable name "gadeon-cli"). Gives the CLI its own cache dir at
            // ~/Library/Caches/io.github.leok7v.gadeon.cli/.
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "cli/Info.plist",
                ], .when(platforms: [.macOS])),
            ]
        ),
        .testTarget(name: "LLMTests", dependencies: ["LLM"], path: "tests"),
    ]
)
