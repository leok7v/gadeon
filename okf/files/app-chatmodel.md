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

A session replacement drains the old one before the new one touches the
engine.
