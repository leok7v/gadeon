import Foundation

enum TestWeights {

    static let repoRoot: String = {
        let cwd = FileManager.default.currentDirectoryPath
        return cwd.hasSuffix("/LLM") ? String(cwd.dropLast(4)) : cwd
    }()

    static func find(_ variable: String, named: String? = nil) -> String? {
        let given = ProcessInfo.processInfo.environment[variable]
        var out = given.flatMap { p in readable(p) }
        if out == nil, let given {
            out = readable(repoRoot + "/" + given)
        }
        if out == nil, let named {
            out = search(named)
        }
        return out
    }

    private static func readable(_ path: String) -> String? {
        FileManager.default.isReadableFile(atPath: path) ? path : nil
    }

    private static var roots: [String] {
        [repoRoot + "/tmp", repoRoot + "/models",
         NSHomeDirectory() + "/Models"]
    }

    private static func search(_ named: String) -> String? {
        var out: String? = nil
        for root in roots where out == nil {
            out = firstMatch(root, named)
        }
        return out
    }

    private static func firstMatch(_ root: String,
                                   _ named: String) -> String? {
        let fm = FileManager.default
        var out = readable(root + "/" + named)
        let level1 = (try? fm.contentsOfDirectory(atPath: root)) ?? []
        for a in level1 where out == nil {
            let branch = root + "/" + a
            out = readable(branch + "/" + named)
            let level2 = (try? fm.contentsOfDirectory(atPath: branch)) ?? []
            for b in level2 where out == nil {
                out = readable(branch + "/" + b + "/" + named)
            }
        }
        return out
    }
}
