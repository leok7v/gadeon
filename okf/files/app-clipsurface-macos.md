---
type: File
title: App/ClipSurface-macOS.swift
description: A clip's playback surface on macOS.
sources:
  - resource: App/ClipSurface-macOS.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

An AVPlayerView at minimal controls with a play/pause overlay, not SwiftUI's
VideoPlayer, whose fixed inline control bar carries a volume flyout that is
unsatisfiable inside AppKit and logs the conflict every time it opens. Losing
that control means losing the whole bar, which is a deliberate trade for a
ten-second clip in a chat bubble.
