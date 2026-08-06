# Pdf2md: where it comes from, and the contract that keeps it cheap

`LLM/src/Shared/Pdf2md.swift` is VENDORED. It is developed in a separate
effort against its own benchmark, and it arrives here by copy. Re-import is
`cp`, and it must stay that way.

## The contract

DO NOT EDIT THE VENDORED FILE. Specifically, do not:

- rename its types (`Converter`, `Page`, `Element`, `Mode`, `Fragment`,
  `Grid`, `Ruling`, `Audit`, `ConversionError`),
- add `public` to anything in it,
- nest its types to tidy the namespace,
- reflow, reorder, or "while I am here" it.

Every one of those is tempting and every one converts the next re-import
from a copy into a merge, against a file whose constants are still being
swept upstream. None of them buys anything: the names collide with nothing
in this module (checked), and `internal` is sufficient because the public
surface lives in a SEPARATE file beside it. Put every adaptation there.

If the file needs a change to be usable, ASK UPSTREAM FOR IT rather than
making it locally. That worked once already: the span audit was `private`
and folded into a trace string, and upstream exposed it as data on request
(`Audit` plus the `onAudit` sink). A local patch would have been faster and
would have cost every re-import after it.

## The gate

`onAudit` is the correctness assertion available here. It fires once per
page whatever `trace` says, and the counts need no ground truth:

    lost == 0 && unaccounted == 0

`lost` counts recognized text that never reached the output; `unaccounted`
counts output text that was never recognized. The failure it catches is the
quiet one -- a span inside a table's rows but outside its columns simply
vanishes, and nothing else in the pipeline notices.

The F1 bench (cell F1, token F1, against LaTeX-derived ground truth) is the
QUALITY question and stays upstream, where the corpus and the harness live.
What runs here is a REGRESSION gate: did this re-import break something the
app depends on. Two different jobs; do not try to host the first one.

## Why the fixtures here are generated, not borrowed

The upstream benchmark is `piushorn/pdf-parse-bench`:

- https://huggingface.co/datasets/piushorn/pdf-parse-bench
- https://github.com/phorn1/pdf-parse-bench

It is MIT (Copyright 2025 Pius Horn) and its PDFs are produced by pdfTeX
from generated LaTeX, so there is no scraped-document problem. Borrowing a
few would have been defensible. It was declined anyway, for two reasons.

FIRST, the content is not as synthetic as the description suggests. The
DOCUMENTS are generated with randomized geometry, and the prose between the
tables is word salad -- but the tables themselves carry real harvested data.
Sampled from the corpus: one page scores AnyLoc, MegaLoc, EgoNN, BEVPlace++
and AutoPlace on nuScenes and Boreas; another scores Deepfakes, FaceShifter,
InfoSwap, SimSwap and BlendFace on ID similarity and FVD; a third is a
coherent algae-growth parameter set. That is lifted from published papers,
or sampled from something derived from them. Upstream's LICENSE does carry a
data note, but it covers the FORMULAS set (Wikipedia, CC BY-SA 4.0) and says
nothing about tables. Numerical results are thin copyright and the risk is
small, but this repo publishes to a public mirror, and a small unresolved
question is still an unresolved question.

SECOND, and this is the better reason: fixtures we generate come with
GROUND TRUTH. We place every cell, so a test can assert cell-level
correctness rather than smoke and anchor strings. Borrowed PDFs would have
given a weaker gate AND a licence footnote.

The cost is honest and worth stating: generated fixtures exercise what we
THINK is hard. The upstream corpus is adversarial in ways we would not
invent. That difficulty is not lost, it is simply not hosted here -- it is
what the upstream bench is for.
