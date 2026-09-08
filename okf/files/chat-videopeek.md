---
type: File
title: Chat/VideoPeek.swift
description: One downsized video frame, carried from a soft turn's tower
  encode to the transcript.
sources:
  - resource: Chat/VideoPeek.swift
tags: [orientation]
timestamp: 2026-09-03T00:00:00Z
---

Moved out of `App/FilmFrame.swift` (which keeps the SwiftUI `FilmFrame`
view): the driver's `TurnEvent.looking` case and `ClipAttachment.spans` both
need the type, and `Chat` cannot depend on `App`.
