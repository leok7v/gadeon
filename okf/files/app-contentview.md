---
type: File
title: App/ContentView.swift
description: Every screen the app has, and which one is showing.
sources:
  - resource: App/ContentView.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Onboarding, download and optimize progress, the failure view, the chat shell
with its bars and drawer, the transcript and its bubbles, the find bar, and
the routes into Settings and the debug view. Full-screen views are hosted
here rather than presented as sheets.

It stops at state. Everything it reads and writes belongs to ChatModel.
See `feedback-host-views-in-content`, `toolcall-strip-landed` and
`thinking-ticker-static-tail`.
