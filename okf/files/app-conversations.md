---
type: File
title: App/Conversations.swift
description: Projecting the live transcript to the store's shape and back.
sources:
  - resource: App/Conversations.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Only the DISPLAY transcript persists: text, reasoning, tool rounds and tiny
thumbnails, never the KV state. So a saved chat is kilobytes and reopening it
is view-only.
