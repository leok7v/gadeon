---
type: File
title: Chat/ConversationStore.swift
description: Conversation persistence, one JSON file per conversation.
sources:
  - resource: Chat/ConversationStore.swift
tags: [orientation]
timestamp: 2026-09-03T00:00:00Z
---

Under Application Support, with an in-memory index kept in sync on every save
and delete rather than re-scanned. It stores what it is given and knows
nothing about transcripts.
