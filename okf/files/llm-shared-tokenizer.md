---
type: File
title: LLM/src/Shared/Tokenizer.swift
description: GPT-2 byte-level BPE, reconstructed from tokenizer.json.
sources:
  - resource: LLM/src/Shared/Tokenizer.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Swift Regex covers the pretokenizer pattern, so there is no hand-written
Unicode scanner. Merges are ranked, which is the canonical rule.
See `qwen35-tokenizer-merges-verified`.
