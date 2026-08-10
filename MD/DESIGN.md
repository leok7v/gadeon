# Markdown - design

This package is a synthesis of two existing sibling projects: the
feature-complete `md.too` document viewer and the streaming Markdown path
inside `im.ai`. Neither on its own fits Gadeon. `md.too` re-parses the
whole document on every change (fine for a static file, wrong for a live
token stream). `im.ai` streams well but has a thin block set and no
export. This package keeps `md.too`'s block richness and rendering while
adopting `im.ai`'s incremental sealing, and it delegates the actual block
construction back to the batch parser so the two never drift.


## Provenance

The sibling repos are cloned locally under `~/github.com/leok7v/`:

- `md.too`: `~/github.com/leok7v/md.too.devs/`
  (GitHub: https://github.com/leok7v/md.too.devs/)
- `im.ai`:  `~/github.com/leok7v/im.ai/`
  (GitHub: https://github.com/leok7v/im.ai/)

What we adapt from where. File paths are under each repo's `src/`.

- Block model + batch parser -> md.too `MarkdownParser.swift`
- Incremental seal / open block -> im.ai `utils/Makrdown.swift`
  (`Markdown.Streamer`)
- Stream cache + render cadence -> im.ai `model/Session.swift`
  (`MarkdownCache`)
- SwiftUI block views -> md.too `BlockViews.swift`
- Single selectable surface -> md.too `DocumentText.swift`
  (+ `-iOS` / `-macOS`)
- Selection + atomic units -> md.too `SelectableText.swift`,
  `Bridges-*.swift`
- Syntax highlighting -> md.too `Highlight.swift` (+ `highlights.ini`)
- TeX-style math -> md.too `TeX.swift`
- Table metrics -> md.too `TableMetrics.swift`
- Image prefetch + aspect fit -> md.too `ImagePrefetch.swift`
- PDF via CoreText -> md.too `PDFRenderer.swift`, `PDFExport.swift`
- HTML / plain export -> md.too `HtmlExport.swift`, `PlainExport.swift`
- Platform font / color / image -> md.too `Platform-*.swift`,
  `FontRole.swift`

Gadeon consumer touch-points (where this package plugs in):

- `App/ChatModel.swift` - `Message` (`text`, `reasoning`), and the
  `sendText` / `sendVision` `for await piece in stream` loops. This is the
  `append` site.
- `App/ContentView.swift` - `bubble(_:)` / `answerText(_:)`, where
  `Text(answer)` becomes `MarkdownView(document:)`.


## Layers

1. Model. `Block`, `ListItem`, `Markdown.Document`. Pure values, no UI.
2. Parse. `Markdown.parse` (batch, whole document) and `MarkdownStream`
   (incremental). Both produce the same `[Block]`.
3. Render. `MarkdownView` (SwiftUI blocks) and the single-surface
   NSAttributedString view. Both consume `[Block]`.
4. Subsystems. Highlight, TeX, table metrics, image prefetch, selection.
   Foundation / CoreText level, no SwiftUI.
5. Export. HTML, PDF, plain text, from `[Block]`.
6. Platform. Typealiases and shims (`PlatformFont` / `PlatformColor` /
   `PlatformImage`) in SDK-split files, so layers 1-5 carry no
   `#if os(...)`.

Layer 3-5 are lifted almost verbatim from `md.too`. The new work is
layer 2's `MarkdownStream` and the block-identity contract that lets
layer 3 stay cheap under streaming.


## The block model

Same shape as `md.too:src/MarkdownParser.swift`:

```
enum Block {
    case heading(level: Int, text: AttributedString)
    case paragraph(AttributedString)
    case code(language: String?, text: String)
    case quote([Block])
    case list(items: [ListItem], tight: Bool)
    case table(headers: [String], rows: [[String]])
    case rule
    case image(alt: String, url: URL, width: CGFloat?, height: CGFloat?)
}
```

`Markdown.Document` wraps `[Block]` with a stable id per block:

```
struct Document { let blocks: [(id: Int, block: Block)] }
```

The id is what makes streaming cheap in the view (see below). Inline
styling stays delegated to Apple's `AttributedString(markdown:)` with
`.inlineOnlyPreservingWhitespace` plus our own reference-link
substitution, `<u>` handling, and TeX segmentation, exactly as
`md.too` does in `inline(_:)`.


## Batch parsing

`Markdown.parse` is `md.too`'s parser: strip reference link definitions in
a first pass (respecting fenced regions), then parse blocks, resolving
reference links in both directions. Used for static documents, table
cells (each cell is re-parsed), and export. It is the ground truth: the
streaming parser must produce byte-identical blocks for the same input
once the stream is complete.


## Streaming parser (the centerpiece)

The requirement: tokens arrive at up to ~30 per second, and the last
block's geometry changes on each token (a sentence grows, a row is added
to a table, a list gains an item, a paragraph wraps differently). We must
not re-parse or re-measure the whole message per token.

### Sealed prefix + one open block

`MarkdownStream` keeps:

```
sealed:   [Block]          // final, never touched again
open:     accumulated lines of the block currently growing
partial:  the trailing line with no '\n' yet
deferred: one line held for table lookahead
refs:     reference link definitions seen so far
```

This mirrors `im.ai`'s `Markdown.Streamer` (`sealed`, `openLines`,
`partial`, `deferred`) in `src/utils/Makrdown.swift`, generalized to
`md.too`'s larger block set.

`append(_ chunk:)`:

1. `partial + chunk`, split on `\n`. The last piece becomes the new
   `partial`; each completed line is classified.
2. Classification decides whether the line continues the open block, seals
   it, or starts a new block. This is line-level work only: the same
   `isFence` / `isHeading` / `isHR` / `isTableSeparator` / `isQuoteStart`
   / `isListStart` / `isIndentedCode` predicates `md.too` already has.
3. When the open block seals, it moves to `sealed` with a fresh id and is
   never revisited.

`snapshot()` returns `sealed` plus the open block rendered in full
(including `partial`). `finish()` flushes `partial` and `deferred`, seals
the last open block, and returns the final document.

### Delegation: the open block is built by the batch parser

Boundary detection is incremental and cheap. Block construction is not
reimplemented. The open block's accumulated lines are handed to the same
`Markdown.parse` (over that small line window only) to produce the real
`Block`. This is the key move: nested lists, task markers, tight / loose,
tables with alignment, and inline styling all come from one code path, so
the streaming and batch results cannot diverge. Cost is proportional to
the open block, not the document.

`im.ai` half-did this: its `Streamer.renderAs` re-implemented list / quote
/ table rendering separately from the one-shot path, which is exactly the
drift we avoid by delegating.

### Complexity

- Per token: O(size of the open block). Sealed blocks contribute nothing.
- Per snapshot in the view: only the open block re-renders and re-measures
  (see identity, below). Sealed blocks keep their cached layout.

A long transcript is therefore flat in cost as it grows, which is the
whole point.

### Why only the last block re-lays-out

`Markdown.Document.blocks` carries a stable id per block. Sealed blocks
keep their id forever; the open block sits at id `sealed.count`. In
SwiftUI, `ForEach(document.blocks, id: \.id)` sees the sealed ids
unchanged (SwiftUI keeps their views and measured heights) and the open
block's id constant across snapshots (SwiftUI updates that one view in
place and re-measures only it). When the open block seals and a new one
opens at the next id, the just-sealed block's final render equals its last
open render, so there is no visible reflow at the seam.

The view is dumb: there is no separate "streaming view". The same
`MarkdownView(document:)` serves static and streaming; the intelligence is
entirely in `MarkdownStream` and the id contract.

### Cadence and coalescing

The package has no timer. The App calls `snapshot()` when it wants to
draw. `im.ai` coalesced to a render tick (`~0.1s` while pinned to the
bottom, slower when scrolled up); Gadeon's `ChatModel` already has a
`statsTicker` / stream loop that can do the same. Snapshotting per token
works; coalescing to ~10 Hz just keeps SwiftUI relaxed.

### Streaming correctness caveats

- Reference links resolve backward only while streaming: a definition
  arriving after a block sealed does not retro-update it. Rare in LLM
  output (inline links dominate). Static `parse` is unaffected.
- One-line table lookahead: a `|`-bearing line is held in `deferred` until
  the next line decides table-vs-paragraph. So the open block can flip
  from paragraph to table when the separator row lands. Matches `im.ai`.
- Incomplete inline markup renders literally and heals when it closes,
  because the open block is re-parsed in full each snapshot.
- An unterminated code fence stays a code block until `finish()`.


## View layer

`MarkdownView` is `md.too`'s `BlockView` switch (`src/BlockViews.swift`):
per-block SwiftUI, tables measured through a width preference then laid out
natural-vs-wrap via `TableMetrics`, code in a horizontal scroll with
highlight and a copy button, images loaded async with a placeholder.

The single-surface renderer is `md.too`'s `DocumentText` path: the whole
document flattened into one `NSAttributedString` shown in a selectable
native text view, with `NSTextTable` (macOS) or tab stops (iOS) for
tables, and atomic-unit attributes so drag-selection snaps around whole
code / table / image units (`src/SelectableText.swift`,
`src/Bridges-*.swift`). Gadeon likely wants per-block for the live turn
(cheap incremental updates) and can offer single-surface for a finished
message or an export preview.

One deliberate improvement over `md.too`: fonts, colors, and spacing move
out of hardcoded literals into a `MarkdownStyle` value, so the App can
match Gadeon's look and its light / dark handling instead of the
viewer's fixed palette.


## Subsystems (reused from md.too)

- Highlight: regex spans from a bundled `highlights.ini`, mask-tracked to
  avoid overlaps, adaptive light / dark (`src/Highlight.swift`). Shipped as
  a package resource.
- TeX: `$...$` / `$$...$$` split, then LaTeX-to-Unicode substitution
  (Greek, operators, super / sub scripts, simple `\frac`), rendered italic
  (`src/TeX.swift`). Not a layout engine; documented as such.
- TableMetrics: column count, char widths, sqrt-weighted point widths with
  minimums, monospaced serialization for copy. One source of table width
  math shared by every renderer and export (`src/TableMetrics.swift`).
- ImagePrefetch: `aspectFit` sizing plus concurrent `TaskGroup` fetch and
  decode (`src/ImagePrefetch.swift`).


## Export

HTML (`src/HtmlExport.swift`): self-contained, inline styles, images as
base64 data URIs with sniffed mime. PDF (`src/PDFRenderer.swift`,
`src/PDFExport.swift`): CoreText framesetter with manual pagination,
headers / footers / page numbers, column-fit tables with numeric
word-joiner protection, A4 vs Letter by region, forced-light appearance.
Plain (`src/PlainExport.swift`): round-trips `[Block]` back to Markdown
text. All three consume the batch `[Block]`, so export always uses the
full-fidelity parse, never a streaming snapshot.


## Platform split

Following `md.too` and Gadeon's own `App/`: no `#if os(...)` in logic.
Platform typealiases and shims (`PlatformFont`, `PlatformColor`,
`PlatformImage`, font trait merging, clipboard, image decode, adaptive
color) live in `Platform-iOS.swift` / `Platform-macOS.swift`. UI bridges
(`NativeText` representable, selection coordinator) live in
`Bridges-iOS.swift` / `Bridges-macOS.swift`. Every other file is
platform-neutral.


## Proposed package layout

```
MD/
  Package.swift            # product "MD", one target, macOS 15 / iOS 18
  README.md
  DESIGN.md
  Sources/MD/
    Block.swift            # Block, ListItem, Document
    Markdown.swift         # parse namespace, inline, normalizeBreaks
    MarkdownStream.swift    # incremental parser (the new code)
    MarkdownView.swift      # SwiftUI block renderer
    BlockViews.swift        # per-block views
    DocumentText.swift      # single-surface + -iOS / -macOS
    SelectableText.swift    # + Bridges-iOS / Bridges-macOS
    Highlight.swift
    TeX.swift
    TableMetrics.swift
    ImagePrefetch.swift
    FontRole.swift
    Style.swift            # MarkdownStyle / theme
    HtmlExport.swift
    PdfExport.swift        # PDFRenderer + PDFExport
    PlainExport.swift
    Platform-iOS.swift
    Platform-macOS.swift
    Resources/highlights.ini
```

- One product, one target for v1 (simplest). If a non-SwiftUI consumer
  ever appears (a CLI PDF exporter), split into `MarkdownCore` (model /
  parse / export / subsystems) and `MarkdownUI` (SwiftUI). Not now.
- No external dependencies. Highlight, TeX, tables, and PDF are all
  in-package, consistent with this repo's dependency discipline.
- Wiring: the App adds this package to its `project.yml` package list (and
  its SwiftPM manifest if the CLI ever renders Markdown). Not done here.


## Coding discipline

This lives in the `coreml.ui` repo, which carries the AGENTS discipline
(`../AGENTS.md`, `../okf/rules/cardinal-rules.md`). The
implementation will hold to it: single entry / single exit, no invented
`bool ok` flags, WHY-only comments, 79-column Swift, match neighbouring
style, no new dependency. The `md.too` sources already read close to this,
so most lifted code needs light touch-up, not rewriting. `im.ai`'s
`Streamer` uses early `return` / `break` in a few spots and will be
restructured to single-exit as it moves in.


## Out of scope for the package

These are app concerns and stay in the consuming target, not the library:

- `MarkdownDocument` (FileDocument), the DocumentGroup, and the file
  picker (`md.too:src/App-*.swift`, `MarkdownDocument.swift`).
- `FileWatcher` live reload (`md.too:src/FileWatcher.swift`).
- The Quick Look extension (`md.too:src/QuickLook.swift`).
- Save panels, share sheets, toolbars (`md.too:src/Toolbar*.swift`).

The package renders and exports; it does not own documents, windows, or
the file system.


## Open decisions for you

1. Single target now, or core / UI split from the start? Recommendation:
   single target.
2. Module named `Markdown` with a `Markdown` parse namespace inside it.
   Call sites are `Markdown.parse(...)`, which compiles cleanly. If the
   module / type same name ever bites, the fallback is module `MDKit`
   with a `Markdown` namespace. Preference?
3. Reasoning (`<think>`) rendering: same `MarkdownView` with a muted style,
   or keep it plain `Text`? Affects whether the App holds one stream or two
   per turn (`im.ai` streams content and reasoning separately, see
   `MarkdownCache.reasonStreamer`).
4. Table alignment: `md.too` tolerates `:` in the separator but does not
   apply alignment. Match that (ignore), or render alignment properly as an
   enhancement?
5. Style surface: how much of Gadeon's look should `MarkdownStyle`
   expose on day one (fonts and colors only, or spacing and code theme
   too)?
