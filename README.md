# Gadeon

A pure-Swift, universal-CoreML chat app for Qwen3.5 / QwenPaw hybrid models,
running inference on the Apple Neural Engine. macOS and iOS from one codebase.

It is a proof of concept for running Gated DeltaNet hybrid models on the
Neural Engine through CoreML, not a finished product or a benchmark leader.

Prefill and decode both run on the Neural Engine. Prefill is measured ~2-2.4x
faster and ~4.4x lower energy per token than llama.cpp Metal on a 0.8B / M3,
GPU left free. Decode is state-carry on the Neural Engine, with an optional
greedy-lossless MTP self-speculative mode via the shipped MTP head.

Pure Swift only: no C, no Python, no FFI. CoreML is the one framework dependency.

## Build

Requirements: an Apple Silicon Mac and Xcode 16+ (Swift 6, macOS 15 SDK). The
app project also needs two Homebrew tools:

    brew install xcodegen xcode-build-server

### Engine and CLI (SwiftPM)

The engine and command-line tools build straight from `LLM/Package.swift`, with
no Xcode project needed. Run from the repo root, as with every command below:

    swift build --package-path LLM
    LLM/.build/debug/gadeon-cli models "what is an interest rate swap?"

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
    xcodebuild -scheme Gadeon -destination 'generic/platform=iOS Simulator' build

The app is Apple-Silicon-only (arm64): the Neural Engine and `Float16` do not
exist on Intel, so `project.yml` excludes `x86_64` (the Mac App Store accepts an
Apple-Silicon-only app). Xcode Run/Build (Debug) and Product > Archive (Release)
are arm64; only a forced-universal command-line build needs an extra
`EXCLUDED_ARCHS=x86_64`, because a SwiftPM package does not inherit the exclusion
in a plain build action.

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

All five use the same `gadeon-cli <model> --bench` (raw 512-token prefill, 128
greedy decodes, one warmup pass): the four Qwen3.5 / QwenPaw models on the Neural
Engine, and the ternary Bonsai-27B through the pure-Swift Metal (GPU) backend
(the default for a GGUF; `--cpu` opts out to the SIMD engine, and `--metal` is
still accepted as a no-op). The ternary Bonsai models run on the GPU by design:
the Neural Engine expands a 2-bit weight to fp16 before the DMA, so a ternary
model pays fp16 bandwidth there and the 2-bit format buys nothing. That is a
property of the format, not of the model size, so it holds for the whole ternary
lineage (measured on the 1.7B, `scripts/probes/bonsai17b_stream_test.py`;
see ROADMAP.md). They are kept as a comparison point: the same harness runs both
backends, so a ternary model's answer quality and speed can be read directly
against the Neural Engine models.
Debug builds (`swift build` default, `-Onone`) are ~100x slower; every number
here is `swift build -c release`.

Energy: on the 0.8B, Neural Engine prefill was measured at ~2-2.4x faster and
~4.4x less energy per token than llama.cpp Metal, with the GPU left free. That
is the one energy figure on record (prefill, 0.8B); per-model and decode energy
are not benchmarked here.

<sub>MacBook Air (M3): 16-core Neural Engine rated 18 TOPS; 10-core GPU; 8-core
CPU (4 performance + 4 efficiency); 100 GB/s unified-memory bandwidth; 24 GB
RAM.</sub>

Prefill speed is the number that matters most in agentic use: when the model
searches the web, reads a Wikipedia article, or pulls in a news story, every
fetched byte is prompt to ingest, not text to generate. A tool round routinely
prefills 4-16 KB of page text to decode a two-sentence conclusion, so the
reading rate, not the talking rate, bounds how many sources a turn can afford.

## Append-only context, rollback, recurrent state

The hybrid trunk forces a session design that stateless servers never need,
and it is worth being explicit about why.

Three quarters of the layers are Gated DeltaNet: their memory is a fixed-size
recurrent state, a lossy fold of everything ingested so far. Unlike a KV
cache, that state is not addressable by prefix (there is no "reuse the first
N tokens" shortcut), and the only way to recompute it is to replay the entire
conversation through the model. So the engine owns its session state rather
than reconstructing it per request:

- **Append-only continuation.** Each turn renders only the delta (the
  previous stripped answer plus the new user turn) through the model's own chat
  template and appends it to the live state. Nothing is re-prefilled; total
  work over a conversation is O(conversation), and per-turn latency does not
  grow with history. The whole-history re-render plus common-prefix diff (the
  stateless-server pattern) is deliberately absent.
- **Marks and rollback.** Before the generation prompt of every turn the
  engine drops a mark: a deep snapshot of the recurrent state plus the paged
  KV (cheap, since completed KV pages are shared copy-on-write). The next turn
  rewinds to it, which is how transient bytes leave the context: raw
  `<think>` reasoning is dropped and the turn re-appends the stripped answer;
  a tool exchange is re-laid in the template's canonical form instead of the
  model's raw emission. Stop during prefill restores the pre-turn snapshot
  entirely, so the turn never happened.
- **Park / resume / persist.** The same snapshot primitive serializes: whole
  conversations park and resume over one loaded model, and the rendered
  system plus tools prefix is precooked to disk once and restored at launch,
  skipping most of the time-to-first-token.

## Layout

- `LLM/` - the engine package: `src/{Shared,CoreML,Metal,SIMD,Slugs}`
  (tokenizer, chat template, sampler, ChatSession, per-backend compute).
- `LLM/cli/` - gadeon-cli, the macOS command-line harness for the engine.
- `App/` - SwiftUI app (macOS + iOS), no `#if os` (SDK file split).
- `MD/` - Markdown transcript rendering package.
- `models/` - downloaded model sets, kept local (not committed).
- `config/platform.xcconfig` - the SDK-scoped source split for the app target.

## Models

The app ships no weights. Model sets download on demand from their pinned
Hugging Face commits (`ModelCatalog` + `HubFetch`: sha-pinned, digest-verified,
resumable) into the app's container, once; later launches are offline.
`gadeon-cli <name>` fetches the same sets into `./models/` (gitignored), or
takes a path to any local set directory.

## Model caches

Two separate things end up on disk:

- `models/*.mlmodelc`: the models shipped with the app, in MIL format
  (`model.mil` + `weights/weight.bin`), the ahead-of-time `coremlc` bake.
- **Compiled Neural Engine programs**: the OS compiles each model for this
  specific chip the first time it loads (the one-time ~30 s "first launch" wait),
  then reuses the result instantly on later launches. They are cached per app
  bundle-id at:

      ~/Library/Caches/<bundle-id>/com.apple.e5rt.e5bundlecache/

  i.e. `io.github.leok7v.gadeon` (app) and `io.github.leok7v.gadeon.cli` (CLI).
  This can reach a few hundred MB. Deleting that directory forces a one-time
  cold recompile on the next launch and reclaims the disk:

      rm -rf ~/Library/Caches/io.github.leok7v.gadeon/com.apple.e5rt.e5bundlecache

  This is the Neural Engine (`.e5`) cache, distinct from the classic CPU/GPU
  `model.espresso.*` caches other CoreML apps produce.

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

---

### Footnote: how the gemma-4-12B repack scores

The 12B GGUF this app downloads is a *recovery* of the quantization-aware
training's own 4-bit codes, not a fresh quantization of the released weights.
Google publishes a second 4-bit build of the same checkpoint, so the two can be
scored against the bf16 they both come from.

Relative weight error against that checkpoint:

| tensor | [ours](https://huggingface.co/leok7v/gemma-4-12b-it-qat) | [w4a16-ct](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct) |
|---|---|---|
| `gate_proj` layer 0 | **1.03e-03** | 6.66e-02 |
| `q_proj` layer 0 | **1.07e-03** | 6.67e-02 |
| `down_proj` layer 30 | **1.10e-03** | 6.66e-02 |

About 65x closer to the trained weights, in a file 6.35 GiB against 9.56 GiB.

Not a better search: the int4 **codes agree 99.40%** between the two files, so
both recover the same trained grid, and Google's build is an independent
witness that the recovery is right. The **scales** differ, theirs a median
1.0645x larger, because a min/max observer takes the block scale from the
block's extreme where this repack refits it by least squares over the settled
codes.

Their build is better on one tensor: it leaves the embedding table in bf16 and
therefore exact, at 1.88 GiB against 0.53 GiB here. Both leave the same modules
unquantized -- the vision patch dense, both multimodal projections and the
position table.

The measurement is reproduced by the converter tooling in the development
repo. Sources:
[base checkpoint](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized),
[Google's 4-bit build](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct),
[this repack](https://huggingface.co/leok7v/gemma-4-12b-it-qat).
