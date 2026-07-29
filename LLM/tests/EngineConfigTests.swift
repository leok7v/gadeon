import XCTest
@testable import LLM

// Pure gates (no model) for the config.json seam: rope_theta reads from both
// the origin HF nesting and the retired flat form, and configMismatches is
// the witness check -- an origin config over the 0.8B's I/O-derived geometry
// verifies clean, any doctored field reports, and a config without the field
// verifies nothing.
final class EngineConfigTests: XCTestCase {

    // The 0.8B origin config's geometry, as Engine derives it from model I/O.
    private func mismatches(_ obj: [String: Any]) -> [String] {
        Engine.configMismatches(obj, dim: 1024, vocab: 248_320, kvHeads: 2,
                                gqaGroup: 4, dhAttn: 256, rot: 64,
                                convTaps: 3, nProg: 7)
    }

    private var origin: [String: Any] {
        [
            "model_type": "qwen3_5_vl",
            "text_config": [
                "hidden_size": 1024,
                "vocab_size": 248_320,
                "num_key_value_heads": 2,
                "num_attention_heads": 8,
                "head_dim": 256,
                "num_hidden_layers": 24,
                "full_attention_interval": 4,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": [
                    "rope_theta": 10_000_000,
                    "partial_rotary_factor": 0.25,
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                ],
            ],
            "vision_config": [
                "spatial_merge_size": 2,
                "patch_size": 16,
                "temporal_patch_size": 2,
            ],
        ]
    }

    func testRopeThetaBothForms() {
        XCTAssertEqual(Engine.ropeTheta(origin), 10_000_000)
        XCTAssertEqual(
            Engine.ropeTheta(["model_type": "qwen3_5",
                              "rope_theta": 10_000_000.0]),
            10_000_000)
        XCTAssertNil(Engine.ropeTheta(["model_type": "qwen3_5"]))
        XCTAssertNil(Engine.ropeTheta([:]))
    }

    func testOriginConfigVerifiesClean() {
        XCTAssertEqual(mismatches(origin), [])
    }

    // The retired minimal form has no geometry fields, so it checks nothing.
    func testMinimalConfigChecksNothing() {
        XCTAssertEqual(
            mismatches(["model_type": "qwen3_5",
                        "rope_theta": 10_000_000.0]), [])
    }

    func testDoctoredFieldsReport() {
        var t = origin["text_config"] as! [String: Any]
        t["hidden_size"] = 2048
        t["num_attention_heads"] = 16
        var doctored = origin
        doctored["text_config"] = t
        let bad = mismatches(doctored)
        XCTAssertEqual(bad.count, 2, "\(bad)")
        XCTAssertTrue(bad.contains { s in s.hasPrefix("hidden_size") })
        XCTAssertTrue(bad.contains { s in
            s.hasPrefix("num_attention_heads")
        })
    }

    func testProgramCountAndRotChecks() {
        var t = origin["text_config"] as! [String: Any]
        t["full_attention_interval"] = 3
        var r = t["rope_parameters"] as! [String: Any]
        r["partial_rotary_factor"] = 0.5
        r["mrope_section"] = [16, 8, 8]
        t["rope_parameters"] = r
        var doctored = origin
        doctored["text_config"] = t
        let bad = mismatches(doctored)
        XCTAssertEqual(bad.count, 3, "\(bad)")
    }

    func testVisionConstantsReport() {
        var doctored = origin
        doctored["vision_config"] = ["spatial_merge_size": 4]
        let bad = mismatches(doctored)
        XCTAssertEqual(bad.count, 1, "\(bad)")
        XCTAssertTrue(bad[0].hasPrefix("vision_config.spatial_merge_size"))
    }
}
