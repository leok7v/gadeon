---
type: File
title: App/ChatModel.swift
description: The app's whole observable state, and the turn it runs.
sources:
  - resource: App/ChatModel.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Model resolution, download, ANE or GPU build, session lifetime, attachments
of every kind, the settings that persist, and the transcript the views read.
Everything a view binds to is here, which is also why it is the largest file
in the app.

It stops at the engine boundary. Prompt rendering, KV state and decoding
belong to ChatSession in LLM.
See `one-backend-many-sessions` and `never-reprefill`.
