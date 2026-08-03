# Gadeon

A pure-Swift, universal-CoreML chat app for Qwen3.5 / QwenPaw hybrid models,
running inference on the Apple Neural Engine. macOS and iOS from one codebase.

It is a proof of concept for running Gated DeltaNet hybrid models on the
Neural Engine through CoreML, not a finished product or a benchmark leader.

Prefill and decode both run on the Neural Engine. Prefill is measured ~2-2.4x
faster and ~4.4x lower energy per token than llama.cpp Metal on a 0.8B / M3,
GPU left free. Decode is state-carry on the Neural Engine, with an optional
greedy-lossless MTP self-speculative mode via the shipped MTP head.

Ternary (Q2_0) models run instead on a pure-Swift Metal GPU backend, behind the
same session and tokenizer seams.

Pure Swift only: no C, no Python, no FFI. CoreML is the one framework dependency.

## Build

Requirements: an Apple Silicon Mac and Xcode 16+ (Swift 6, macOS 15 SDK). The
app project also needs two Homebrew tools:

    brew install xcodegen xcode-build-server

### Engine and CLI (SwiftPM)

The engine and command-line tools build straight from `LLM/Package.swift`, with
no Xcode project needed. Run from the repo root, as with every command below:

    swift build -c release --package-path LLM
    LLM/.build/release/gadeon-cli Qwen3.5-0.8B "what is an interest rate swap?"

Debug builds (`swift build` default, `-Onone`) are ~100x slower; quote only
Release numbers.

### Tests

    swift test --package-path LLM --skip MultiTurnTests

Green with no model on disk: the gates that need one skip. Two small files from
the public origin switch them on, and neither is a weight:

    mkdir -p models/Qwen3.5-0.8B
    hf download Qwen/Qwen3.5-0.8B tokenizer.json chat_template.jinja \
      --local-dir models/Qwen3.5-0.8B

12 MB of tokenizer and 8 KB of chat template, between them gating the
byte-level BPE, the jinja render, and the append-only continuation. Either
layout resolves: flat as above, or the `{sha}` subdirectory a set download
stages into.

`MultiTurnTests` is excluded above because it loads a real CoreML set on the
Neural Engine. It is the continuation-fidelity oracle, and it needs a complete
`models/Qwen3.5-0.8B/{sha}/`, so run it deliberately where one exists.

### App (Xcode)

`Gadeon.xcodeproj` is NOT committed: it is generated from `project.yml` (the
source of truth) by xcodegen. So build settings live in `project.yml`, not the
Xcode UI, and there are no `project.pbxproj` merge conflicts. Regenerate it
after cloning and after any `project.yml` change:

    xcodegen generate

Then build from the command line, or open `Gadeon.xcodeproj` in Xcode (Run, or
Product > Archive):

    xcodebuild -scheme Gadeon -destination 'platform=macOS' build

The app is Apple-Silicon-only (arm64): the Neural Engine and `Float16` do not
exist on Intel, so `project.yml` excludes `x86_64`.

### Editor tooling (SourceKit-LSP)

`xcode-build-server` writes `buildServer.json` so SourceKit-LSP resolves against
the real Xcode build. Rerun it after generating, then restart the language
server:

    xcode-build-server config -scheme Gadeon -project Gadeon.xcodeproj

## Performance

Throughput on a MacBook Air (M3, 24 GB), Release build, same protocol as
`llama-bench -p 512 -n 128`: a 512-token prefill (`pp512`) then 128 decoded
tokens (`tg128`), reported tokens/sec.

| Model            | Backend       | Prefill pp512 | Decode tg128 |
|------------------|---------------|--------------:|-------------:|
| Qwen3.5-0.8B     | Neural Engine |   3065 tok/s  |   66.9 tok/s |
| QwenPaw-2B       | Neural Engine |   2157 tok/s  |   36.1 tok/s |
| QwenPaw-4B       | Neural Engine |    846 tok/s  |   17.0 tok/s |
| QwenPaw-9B       | Neural Engine |    456 tok/s  |    9.9 tok/s |
| Bonsai-27B Q2_0  | GPU (Metal)   |     48 tok/s  |    8.2 tok/s |

All five use the same `gadeon-cli <model> --bench`. The ternary Bonsai models
run on the GPU by design: the Neural Engine expands a 2-bit weight to fp16
before the DMA, so a ternary model pays fp16 bandwidth there and the 2-bit
format buys nothing. That is a property of the format, not of the model size.

Decode cost grows with cached positions, so `--ctx N` benches at a realistic
context rather than at the ~640 the plain bench reaches:

    gadeon-cli <model.gguf> --metal --bench --ctx 4096

Prefill speed is the number that matters most in agentic use: when the model
searches the web or reads an article, every fetched byte is prompt to ingest,
not text to generate.

## Append-only context, rollback, recurrent state

Three quarters of the layers are Gated DeltaNet: their memory is a fixed-size
recurrent state, a lossy fold of everything ingested so far. Unlike a KV cache
that state is not addressable by prefix, so the engine owns its session state
rather than reconstructing it per request:

- **Append-only continuation.** Each turn renders only the delta (the previous
  stripped answer plus the new user turn) through the model's own chat template
  and appends it. Total work over a conversation is O(conversation), and
  per-turn latency does not grow with history.
- **Marks and rollback.** Before each turn's generation prompt the engine drops
  a mark: a deep snapshot of the recurrent state plus the paged KV. The next
  turn rewinds to it, which is how transient bytes leave the context: raw
  `<think>` reasoning is dropped, and a tool exchange is re-laid in the
  template's canonical form instead of the model's raw emission.
- **Park / resume / persist.** The same snapshot primitive serializes, so whole
  conversations park and resume over one loaded model.

## Layout

- `LLM/` - the engine package: `src/{Shared,CoreML,Metal,SIMD,Slugs}`
  (tokenizer, chat template, sampler, ChatSession, per-backend compute).
- `LLM/cli/` - gadeon-cli, the macOS command-line harness for the engine.
- `App/` - SwiftUI app (macOS + iOS), no `#if os` (SDK file split).
- `MD/` - Markdown transcript rendering package.
- `models/` - downloaded model sets, kept local (not committed).
- `config/` - entitlements and the SDK-scoped source split for the app target.

## Models

The app ships no weights. Model sets download on demand from their pinned
Hugging Face commits (`ModelCatalog` + `HubFetch`: sha-pinned, digest-verified,
resumable) into the app's container, once; later launches are offline.
`gadeon-cli <name>` fetches the same sets into `./models/` (gitignored), or
takes a path to any local set directory.

## Model caches

Two separate things end up on disk:

- `models/*.mlmodelc`: the model sets in MIL format (`model.mil` +
  `weights/weight.bin`), the ahead-of-time `coremlc` bake.
- **Compiled Neural Engine programs**: the OS compiles each model for this
  specific chip the first time it loads (the one-time ~30 s "first launch"
  wait), then reuses the result instantly on later launches. Cached per app
  bundle-id at:

      ~/Library/Caches/<bundle-id>/com.apple.e5rt.e5bundlecache/

  This can reach a few hundred MB. Deleting that directory forces a one-time
  cold recompile on the next launch and reclaims the disk.

## Credits

- **Qwen team** for the Qwen3.5 / Qwen3.6 models and the Gated DeltaNet hybrid
  design this engine implements.
  [Qwen on Hugging Face](https://huggingface.co/Qwen)
- **PrismML** for the ternary Bonsai builds and the Q2_0 weight format.
  [prismml.com](https://prismml.com/)
- **llama.cpp** (ggml-org), MIT: the Metal `q2_0_gemm_mm` kernel is a port of
  its `kernel_mul_mm`, the vision tower is transcribed from
  `clip_graph_qwen3vl`, and the SIMD engine is validated against it end to end.

## License

Gadeon is **GPLv3 or later**. That is arithmetic, not preference -- it is the
strongest obligation among the parts it is built from:

| part | source | licence |
|---|---|---|
| speech model + voices | [KittenTTS](https://github.com/KittenML/KittenTTS) by KittenML | Apache-2.0 |
| English pronunciation data (`en_rules`, `en_list`) | [eSpeak NG](https://github.com/espeak-ng/espeak-ng) | **GPLv3 or later** |
| everything else here | this repo | Copyright (C) 2026 Leo Kuznetsov |

The pronunciation data is copyleft, and eSpeak NG grants **no exception for a
program's output** -- so shipping those files, or a lexicon derived by running
them, carries the same terms rather than escaping them. Apache-2.0 is
one-way compatible with GPLv3 (it may be combined into a GPLv3 work, though
not GPLv2), which is why the result is GPL **v3** specifically and cannot be
anything more permissive.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. See [LICENSE](LICENSE).

It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU General Public License for more details.

### Credits

- **[KittenML / KittenTTS](https://github.com/KittenML/KittenTTS)** -- the
  speech model and its eight voices, Apache-2.0.
- **[eSpeak NG](https://github.com/espeak-ng/espeak-ng)** -- the English
  letter-to-sound rules and exception dictionary, Copyright (C) 2005-2014
  Jonathan Duddington and Copyright (C) 2016-2017 Reece H. Dunn, GPLv3+.

Model weights are covered by their own upstream licences, not by this repo's.
