---
type: File
title: LLM/Base/Docs2md.swift
description: Vendored. The formats that state their own structure, read to
  Markdown.
sources:
  - resource: LLM/Base/Docs2md.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

docx, xlsx, pptx, epub and html. Copied in byte-identical from another repo
and MUST NOT be edited here, so a re-import stays a copy rather than a merge;
its own okf pointers refer to that repo's bundle. Adapt through
Docs2mdPublic.swift and see LLM/fixtures/pdf2md/ORIGIN.md.

Gated by `Docs2mdTests` on containers BUILT in the test, so ground truth is
what was put in them: the reader accepts stored zip entries, so a fixture is
a few XML parts, nothing is licensed and nothing published. That says whether
a re-import still reads the structure the app depends on, not whether the
reader is good on documents in the wild, which is measured upstream.
