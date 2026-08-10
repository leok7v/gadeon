---
type: File
title: LLM/src/CoreML/Engine.swift
description: The Neural Engine trunk, and the state carried between
  prefill and decode.
sources:
  - resource: LLM/src/CoreML/Engine.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

An actor owning position, the KV pool, the compiled programs and the
bookmarks a conversation parks and restores, plus the vision prefill and the
speculative verify path.

It stops at tokens in and a token out. Templates, tools and turn structure
belong to ChatSession.
See `prefill-launch-compile` and `h12-ane-decode-unsupported`.
