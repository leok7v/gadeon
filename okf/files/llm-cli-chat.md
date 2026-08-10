---
type: File
title: LLM/cli/Chat.swift
description: The chat drive modes.
sources:
  - resource: LLM/cli/Chat.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The shipping ChatSession turn loop, the raw token-by-token references, and
the ternary GGUF main that serves a .gguf argument end to end. The default
bench prompt is exactly 512 tokens under two tokenizers and must not be
reflowed, because the count is what makes it comparable.
