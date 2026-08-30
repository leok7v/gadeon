---
type: File
title: LLM/src/Shared/TopicTitle.swift
description: The title a conversation gets when the model will not give one.
sources:
  - resource: LLM/src/Shared/TopicTitle.swift
tags: [orientation]
timestamp: 2026-08-30T00:30:00Z
---

Word frequency over the transcript, stop-worded and read at title time so
search scoring is untouched. A word must repeat to count as a subject, which
is what keeps "Hi / Hello / How are you" from earning a title at all; the
winners are then ordered by first appearance so the result reads as a phrase
rather than a ranking, and title-cased.

Falls to a locale-short timestamp, never to a truncated prompt.
See `a-closed-empty-think-block-makes-gemma-narrate`.
