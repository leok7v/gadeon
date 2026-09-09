---
type: File
title: App/Conversations.swift
description: The view model's thin wrapper around conversation persistence.
sources:
  - resource: App/Conversations.swift
tags: [orientation]
timestamp: 2026-09-03T00:00:00Z
---

Gathers the current transcript and hands it to the driver
(`Chat/Conversations.swift`, an extension of `Chat.Session`), which owns the
`ConversationStore` translation -- Message/ToolRound to and from
`ConversationStore.Msg`/`Round` -- then applies what comes back to this
model's own `@Observable` state. Only the DISPLAY transcript persists: text,
reasoning, tool rounds, the 640 px JPEG previews of attached images and the
poster frame of each clip, never the clip file and never the KV state, so a
saved chat is kilobytes and reopening it is view-only. A reopened turn
cannot re-send its clip, and that is accepted: re-prefill is not an option
at these rates, and a continuation, if one ever lands, saves the engine
state rather than the inputs.
