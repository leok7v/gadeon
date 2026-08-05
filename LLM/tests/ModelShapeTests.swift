import Foundation
import Testing
@testable import LLM

// ModelShape is what the app's status line says about a model before any turn
// has produced a number, so everything in it is read from the file. The two
// ways that can go quietly wrong are checked here against the tensors: a
// metadata width that no longer matches the embedding table it describes, and
// a tower rule that drops a tensor prefix on the floor when an emit adds one.
//
// Weight-gated on GADEON_GEMMA_GGUF (see needsGemmaWeights), so without the
// file this reports as SKIPPED rather than passing on an empty body.
struct ModelShapeTests {

    @Test(needsGemmaWeights) func shapeAgreesWithTheTensors() throws {
        let g = try GGUF(path: gemmaGgufPath!)
        let shape = ModelShape(gguf: g)
        // The stated width IS the embedding table's row length.
        #expect(shape.embedding == g.tensor("token_embd.weight").dims[0])
        // E2B and E4B state the trained context only inside the origin
        // config.json they carry; the 12B keys it directly. Either way the
        // file answers.
        #expect(shape.trainedContext > 0)
        // Every tensor lands in exactly one tower, so the towers account for
        // the whole weight payload rather than for whatever the prefix rule
        // happened to recognise.
        let payload = g.tensors.values.reduce(0) { sum, t in
            sum + t.byteCount
        }
        let towers = shape.towers.reduce(0) { sum, t in sum + t.bytes }
        #expect(towers == payload)
        // The trunk leads, and outweighs any sense grafted onto it.
        #expect(shape.towers.first?.name == "text")
        let trunk = shape.towers.first?.bytes ?? 0
        #expect(shape.towers.dropFirst().allSatisfy { t in t.bytes < trunk })
    }
}
