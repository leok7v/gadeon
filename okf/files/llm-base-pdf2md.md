---
type: File
title: LLM/Base/Pdf2md.swift
description: Vendored. A PDF read to Markdown through its own geometry.
sources:
  - resource: LLM/Base/Pdf2md.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Copied in byte-identical from another repo and MUST NOT be edited here, so a
re-import stays a copy rather than a merge; its own okf pointers refer to that
repo's bundle. Adapt through Pdf2mdPublic.swift and see
LLM/fixtures/pdf2md/ORIGIN.md.

Gated by `Pdf2mdTests` on pages GENERATED in the test, prose, a ruled table
and two columns, each with a real text layer so `.geometry` reads the word
boxes and nothing is recognised; that is what keeps the runs fast and
identical everywhere, and `spansMatchTheTextLayer` catches a silent fall
back to recognition. Quality on real documents is measured upstream.
