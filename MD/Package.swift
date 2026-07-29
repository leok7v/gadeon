// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MD",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [.library(name: "MD", targets: ["MD"])],
    targets: [
        .target(
            name: "MD",
            path: "src",
            resources: [.process("Resources")]
        ),
        .testTarget(
            name: "MDTests",
            dependencies: ["MD"],
            path: "tests"
        ),
    ]
)
