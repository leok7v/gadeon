---
type: File
title: App/VoicePlayer.swift
description: Synthesis and playback for a speaking session.
sources:
  - resource: App/VoicePlayer.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Text segments in, sound out, in order, with a stop that takes effect
immediately. Synthesis runs on its own serial queue because it is synchronous
and holds an arena for a whole call, and playback schedules buffers into a
player node so consecutive sentences abut with no gap and no temporary file.
See `voice-conversation-landed`.
