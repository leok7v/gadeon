---
type: File
title: LLM/Base/Tokenizer.swift
description: GPT-2 byte-level BPE, out of the GGUF's own metadata or a
  tokenizer.json.
sources:
  - resource: LLM/Base/Tokenizer.swift
tags: [orientation]
timestamp: 2026-09-05T12:00:00Z
---

Swift Regex covers the pretokenizer pattern, so there is no hand-written
Unicode scanner. Merges are ranked, which is the canonical rule. The GGUF
stores only the pretokenizer's NAME, not its regex, so the Qwen pattern is a
constant here, verified byte-identical to the one the Qwen3.5 tokenizer.json
carries.
