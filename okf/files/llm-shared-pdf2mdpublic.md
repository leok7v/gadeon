---
type: File
title: LLM/src/Shared/Pdf2mdPublic.swift
description: The public seam over the vendored PDF converter.
sources:
  - resource: LLM/src/Shared/Pdf2mdPublic.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

One entry point: a URL in, Markdown out, with an optional per-page progress
callback. It fixes the converter's mode and hides every other knob.

It stops at adaptation, because the vendored file must stay byte-identical.
See `pdf-text-layer-before-ocr`.
