---
type: File
title: App/AudioSession-iOS.swift
description: The one audio session, and the category each use needs of it.
sources:
  - resource: App/AudioSession-iOS.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Record and playback are mutually exclusive on iOS, so the microphone and the
voice take turns and barge-in stops the speech BEFORE claiming the input. Who
is using it is tracked because the two ends do not arrive in order.
See `voice-conversation-landed`.
