---
type: File
title: MD/src/Bridges-iOS.swift
description: The UIKit side of the native text surface.
sources:
  - resource: MD/src/Bridges-iOS.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The representable, the resizing text view, and the copy button pinned over an
atomic run. Find highlighting uses REAL background colour with save and
restore, because UIKit's layout manager has no temporary attributes.
See `md-single-surface-transcript` and
`uitextview-init-skips-swift-ivars`.
