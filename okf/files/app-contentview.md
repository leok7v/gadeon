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

The drawer is 300 pt scaled by the text zoom, except on an iPad in the
regular width class, where it takes half the window if that is more.

It stops at state. Everything it reads and writes belongs to ChatModel.
