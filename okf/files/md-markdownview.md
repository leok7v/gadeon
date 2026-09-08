---
type: File
title: MD/src/MarkdownView.swift
description: The per-block SwiftUI renderer.
sources:
  - resource: MD/src/MarkdownView.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Identity driven: each block carries a stable id, so feeding a changed
document re-renders only the block whose value changed. It cannot select
across blocks, which is why the transcript does not use it.
