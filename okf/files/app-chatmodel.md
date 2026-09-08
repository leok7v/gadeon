---
type: File
title: App/ChatModel.swift
description: The view model -- transcript, input, attachments and every
  setting a view binds to.
sources:
  - resource: App/ChatModel.swift
tags: [orientation]
timestamp: 2026-09-03T00:00:00Z
---

Messages and their Markdown streams, input/caret/attachments, zoom, theme,
speech, whimsical phrase state and every derived property the views read.
It owns one `Chat.Session` (the driver) and applies what it reports; it
never touches the engine directly.

See `the-driver-is-not-the-view-model` for the seam between this file and
`Chat/Session.swift`, and `one-backend-many-sessions` for why a session
replacement must drain the old one before the new one touches the engine.
