---
type: File
title: App/TitleBarClick-macOS.swift
description: A click in the title bar, which no SwiftUI view covers.
sources:
  - resource: App/TitleBarClick-macOS.swift
tags: [orientation]
timestamp: 2026-08-30T01:00:00Z
---

The drawer's scrim is content and stops at `contentLayoutRect`, so nothing
reachable from SwiftUI sits over the title bar and a click there had no view
to land on. A local `.leftMouseDown` monitor answers it, returning the event
UNCONSUMED so window dragging still works, and testing y against the window's
own `contentLayoutRect` rather than a guessed bar height.
