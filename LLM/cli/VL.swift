import CoreML
import Foundation
import LLM

// The vision drive modes: the offline fixture gate (--vl-gate), the
// single-shot real-image path (--vl-image), and the multi-turn
// carry gate (--vl-chat). Dispatched from main.swift.

// Load the VL fixture, run the fused vision prefill, and gate the first token
// against the HF reference. The engine builds the 3D M-RoPE positions itself
// from (ids, imageStart, grid), so this exercises the whole Swift port.
@MainActor
func runVLGate(_ dir: String) async throws {
    let base = URL(fileURLWithPath: dir)
    let raw = try Data(contentsOf: base.appendingPathComponent("vl_ref.json"))
    let ref = try JSONSerialization.jsonObject(with: raw) as! [String: Any]
    let ids = (ref["ids"] as! [Int]).map { Int32($0) }
    let grid = ref["grid"] as! [Int]
    let imageStart = ref["image_start"] as! Int
    let refArg = Int32(ref["ref_argmax"] as! Int)
    let refCont = (ref["ref_cont"] as! [Int]).map { Int32($0) }
    let pdata = try Data(contentsOf: base.appendingPathComponent("patches.bin"))
    let nPatch = grid[0] * grid[1] * grid[2]
    let patchDim = (pdata.count / 2) / nPatch
    let patches = try MLMultiArray(
        shape: [NSNumber(value: nPatch), NSNumber(value: patchDim)],
        dataType: .float16)
    pdata.withUnsafeBytes { s in
        patches.withUnsafeMutableBytes { d, _ in
            _ = memcpy(d.baseAddress!, s.baseAddress!, pdata.count)
        }
    }
    err("[vl-gate] prompt \(ids.count) tok, image at \(imageStart), "
        + "grid \(grid), patches \(nPatch)x\(patchDim)\n")
    var next = try await eng.prefillVision(ids, tiles: VisionTiles([patches]),
                                           imageStarts: [imageStart],
                                           gridH: grid[1], gridW: grid[2])
    var out: [Int32] = []
    while out.count < max(refCont.count, 6), next != tok.eosId {
        out.append(next)
        next = try await eng.decode(next)
    }
    let first = out.first ?? -1
    let n = min(out.count, refCont.count)
    var match = 0
    for i in 0 ..< n where out[i] == refCont[i] { match += 1 }
    print("VL   : \(tok.decode(out))")
    print("REF  : \(tok.decode(refCont))")
    err(first == refArg
        ? "[vl-gate] first-token MATCH \(first) (\(tok.decode([first]))) "
          + "vs HF · continuation \(match)/\(n)\n"
        : "[vl-gate] MISMATCH first=\(first) ref=\(refArg)\n")
}

// Real image path: Swift-preprocess the PNG, build the VL prompt, fuse-prefill,
// and stream the answer -- the same path the app's [+] drives.
@MainActor
func runVLImage(_ png: String, _ text: String) async throws {
    let data = try Data(contentsOf: URL(fileURLWithPath: png))
    let grid = await eng.visionGrid() ?? VisionGrid.canonical
    let tiles = try VisionPreprocess.imageSet(data, tiled: !fitImage,
                                              grid: grid)
    let built = try VLPrompt.buildTiles(tok, template: template, text,
                                        tiles.count, grid.mergedTokens)
    err("[vl-image] \(png): \(tiles.count) tiles @\(grid.side)px, "
        + "prompt \(built.ids.count) tok\n")
    print("\nUSER: [image] \(text)\nASSISTANT: ", terminator: ""); fflush(stdout)
    // Sample with penalties (not raw greedy), so a long OCR does not degenerate;
    // --greedy switches to temperature-0 (argmax) WITH repeat/presence penalties
    // -- deterministic (no RNG) yet loop-free, so an A/B has the tower as its
    // only variable without argmax degenerating into a repeat loop.
    let vlConfig = greedyDecode
        ? SamplerConfig(temperature: 0, repeatPenalty: 1.3,
                        presencePenalty: 1.5,
                        setMask: [.temperature, .repeatPenalty,
                                  .presencePenalty])
        : activePresets.select(thinking: enableThinking, vision: true)
    await eng.useSampler(Sampler(vocabSize: tok.vocabCount, config: vlConfig))
    var next = try await eng.prefillVision(
        built.ids, tiles: VisionTiles(tiles), imageStarts: built.imageStarts,
        gridH: grid.gridH, gridW: grid.gridW)
    var out: [Int32] = []
    var stream = StreamDecoder()
    while out.count < maxTokens, next != tok.eosId {
        out.append(next)
        let piece = stream.step(out, tok)
        if !piece.isEmpty { print(piece, terminator: ""); fflush(stdout) }
        next = try await eng.decode(next)
    }
    print()
}

// Multi-turn vision: turn 1 through replyVision (image prefilled + carried in
// the KV), later turns through reply (text; the image stays resident). A turn
// of the form "img:PATH the question" attaches a NEW image mid-conversation
// (replyVision again -> the no-reset vision-extend carry), so turn 2+ can carry
// the first image OR bring a second one -- the mid-conversation carry gate.
@MainActor
func runVLChat(_ png: String, _ turns: [String]) async throws {
    let grid = await eng.visionGrid() ?? VisionGrid.canonical
    func speakVision(_ text: String, _ path: String) async throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let tiles = try VisionPreprocess.imageSet(data, tiled: !fitImage,
                                                  grid: grid)
        err("[vl-chat] \(path): \(tiles.count) tiles @\(grid.side)px\n")
        print("\nUSER: [image] \(text)\nASSISTANT: ", terminator: "")
        fflush(stdout)
        let stream = session!.replyVision(
            text, tiles: tiles, gridH: grid.gridH, gridW: grid.gridW,
            tokensPerImage: grid.mergedTokens, onReasoning: { r in err(r) })
        for await piece in stream { print(piece, terminator: ""); fflush(stdout) }
        print()
    }
    // "img:PATH rest" -> (PATH, rest); nil for a plain text turn.
    func laterImage(_ turn: String) -> (path: String, text: String)? {
        var result: (path: String, text: String)? = nil
        if turn.hasPrefix("img:") {
            let body = turn.dropFirst("img:".count)
            let cut = body.firstIndex(of: " ") ?? body.endIndex
            let path = String(body[..<cut])
            let text = cut < body.endIndex
                ? String(body[body.index(after: cut)...]) : ""
            result = (path, text.isEmpty ? VLPrompt.defaultPrompt : text)
        }
        return result
    }
    if let session, let firstTurn = turns.first {
        try await speakVision(firstTurn, png)
        for turn in turns.dropFirst() {
            if let img = laterImage(turn) {
                try await speakVision(img.text, img.path)
            } else {
                print("\nUSER: \(turn)\nASSISTANT: ", terminator: "")
                fflush(stdout)
                for await piece in session.reply(turn,
                                                 onReasoning: { r in err(r) }) {
                    print(piece, terminator: ""); fflush(stdout)
                }
                print()
            }
        }
    }
}

// Several images in ONE turn, numbered by the template itself
// (add_vision_id -> "Picture 1: "), so a question can refer to them: "is
// Picture 1 inside Picture 2". Fit mode is what makes the numbering line up
// with the PICTURES -- tiling emits one placeholder per TILE, so a tiled
// image would consume several numbers on its own.
@MainActor
func runVLImages(_ pngs: [String], _ question: String) async throws {
    let grid = await eng.visionGrid() ?? VisionGrid.canonical
    var tiles: [MLMultiArray] = []
    for png in pngs {
        let data = try Data(contentsOf: URL(fileURLWithPath: png))
        tiles.append(contentsOf: try VisionPreprocess.imageSet(
            data, tiled: false, grid: grid))
    }
    err("[vl-images] \(pngs.count) image(s) -> \(tiles.count) tile(s) "
        + "@\(grid.side)px\n")
    if let session {
        print("\nUSER: [\(pngs.count) images] \(question)\nASSISTANT: ",
              terminator: "")
        fflush(stdout)
        let stream = session.replyVision(
            question, tiles: tiles, gridH: grid.gridH, gridW: grid.gridW,
            tokensPerImage: grid.mergedTokens, numberImages: true,
            onReasoning: { r in err(r) })
        for await piece in stream { print(piece, terminator: ""); fflush(stdout) }
        print()
    }
}
