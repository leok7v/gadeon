---
type: File
title: LLM/src/Shared/GemmaTokenizer.swift
description: The SentencePiece-flavoured tokenizer gemma ships.
sources:
  - resource: LLM/src/Shared/GemmaTokenizer.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Driven ENTIRELY by the GGUF's own metadata rather than by a model name:
metaspace, byte fallback, ignore_merges, token types, and the fact that gemma
stops on three ids rather than one. A second implementation beside
Tokenizer.swift, not a branch of it. See `gemma4-port`.
