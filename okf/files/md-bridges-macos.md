---
type: File
title: MD/src/Bridges-macOS.swift
description: The AppKit side of the native text surface.
sources:
  - resource: MD/src/Bridges-macOS.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The representable, its delegate, the resizing text view and the copy
overlays. Find highlighting uses layout-manager TEMPORARY attributes, which do
not mutate storage, so code and table backgrounds survive and the incremental
attribute diff is undisturbed. See `md-single-surface-transcript`.
