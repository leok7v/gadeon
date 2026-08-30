import Foundation
import Testing
@testable import LLM

struct ModelCatalogTests {

    private static let name = "gemma-4-E2B"

    private func store() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("catalog-" + UUID().uuidString)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func setDir(_ dest: URL) throws -> URL {
        let set = try #require(
            ModelCatalog.localSet(ModelCatalogTests.name, in: dest))
        try FileManager.default.createDirectory(
            at: set, withIntermediateDirectories: true)
        return set
    }

    private func legacy(_ set: URL, _ want: String) -> URL {
        set.appendingPathComponent(
            (want as NSString).deletingPathExtension + ".gguf")
    }

    @Test func everyCatalogedFileIsGgxf() throws {
        for (_, file) in ModelCatalog.ggufFiles {
            #expect(file.hasSuffix(".ggxf"))
        }
        for (_, src) in ModelCatalog.sources {
            for file in src.files ?? [] {
                #expect(file.hasSuffix(".ggxf"))
            }
        }
    }

    @Test func aDownloadedGgufIsAdopted() throws {
        let fm = FileManager.default
        let dest = try store()
        defer { try? fm.removeItem(at: dest) }
        let set = try setDir(dest)
        let want = try #require(
            ModelCatalog.ggufFiles[ModelCatalogTests.name])
        let was = legacy(set, want)
        try Data("weights".utf8).write(to: was)
        let path = try #require(
            ModelCatalog.ggufPath(ModelCatalogTests.name, in: dest))
        #expect(path.hasSuffix(".ggxf"))
        #expect(fm.fileExists(atPath: path))
        #expect(!fm.fileExists(atPath: was.path))
        #expect(try Data(contentsOf: URL(fileURLWithPath: path))
                == Data("weights".utf8))
    }

    @Test func aPresentGgxfWins() throws {
        let fm = FileManager.default
        let dest = try store()
        defer { try? fm.removeItem(at: dest) }
        let set = try setDir(dest)
        let want = try #require(
            ModelCatalog.ggufFiles[ModelCatalogTests.name])
        let was = legacy(set, want)
        try Data("stale".utf8).write(to: was)
        try Data("real".utf8).write(to: set.appendingPathComponent(want))
        let path = try #require(
            ModelCatalog.ggufPath(ModelCatalogTests.name, in: dest))
        #expect(try Data(contentsOf: URL(fileURLWithPath: path))
                == Data("real".utf8))
        #expect(fm.fileExists(atPath: was.path))
    }

    @Test func nothingIsInventedWhenNeitherExists() throws {
        let fm = FileManager.default
        let dest = try store()
        defer { try? fm.removeItem(at: dest) }
        _ = try setDir(dest)
        let path = try #require(
            ModelCatalog.ggufPath(ModelCatalogTests.name, in: dest))
        #expect(!fm.fileExists(atPath: path))
    }
}
