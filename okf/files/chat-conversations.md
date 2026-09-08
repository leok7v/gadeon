---
type: File
title: Chat/Conversations.swift
description: The driver's side of conversation persistence -- the
  Message/ConversationStore.Msg translation.
sources:
  - resource: Chat/Conversations.swift
tags: [orientation]
timestamp: 2026-09-03T00:00:00Z
---

An extension of `Session`: `commitCurrent`/`openConversation` translate
between the live transcript and `ConversationStore.Convo`, and
`sweepAttachments` deletes any kept file nothing cites any more. Takes
messages/trace as parameters and returns values -- see `App/Conversations
.swift` for the view model's thin caller.
