---
type: File
title: App/ConversationStore.swift
description: Conversation persistence, one JSON file per conversation.
sources:
  - resource: App/ConversationStore.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Under Application Support, with an in-memory index kept in sync on every save
and delete rather than re-scanned. It stores what it is given and knows
nothing about transcripts.
