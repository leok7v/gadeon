---
type: File
title: Chat/Session.swift
description: The session driver -- model build/switch, the ChatSession
  lifecycle, and one turn.
sources:
  - resource: Chat/Session.swift
tags: [orientation]
timestamp: 2026-09-05T00:30:00Z
---

Everything `App/ChatModel.swift` used to do below the view layer: build and
switch the GGUF backend, keep one `ChatSession` alive across turns, and run
`sendText`/`sendSoft`/`sendSpoken`, reporting each turn as an
`AsyncStream<TurnEvent>`. Every attachment goes through the model's
`MediaEncoder`, and the modalities it offers are the encoder's. It holds
no view state. After a build it reads the model file once so the first
prompt faults no weights in from disk, when the file fits in memory
(`the-first-prefill-paid-the-page-in`).
See `the-driver-is-not-the-view-model` and
`one-media-seam-over-every-model`.
