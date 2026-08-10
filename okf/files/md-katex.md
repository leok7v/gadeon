---
type: File
title: MD/src/KaTeX.swift
description: A self-contained TeX math layout engine. One file, one font, no
  WebView and no JavaScript.
sources:
  - resource: MD/src/KaTeX.swift
tags: [orientation, vendored]
timestamp: 2026-08-10T03:30:00Z
---

Font metrics, big-operator sizes, stretchy delimiter recipes and the TeX
layout constants all come from the OpenType MATH table of STIXTwoMath, which
this package carries in Resources. Parse and lay out a fragment, then draw or
rasterize it.

Its output is GEOMETRY, so a change here breaks nothing that compiles and
shows up in no diff. `MD/tests/KaTeXGoldenTests.swift` is the gate.

Carried in from another project, and it now obeys the house rules rather
than claiming the exemption it used to. Four divergences from upstream must
survive a re-import: the font resolves through Bundle.module, the value types
carry Sendable, the entry point holds a lock across a whole layout, and the
file is single-exit throughout.
See `two-engines-for-one-formula` and `a-layout-engine-can-obey-the-rules`.
