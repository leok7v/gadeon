---
type: File
title: LLM/src/Shared/Grammar.swift
description: A byte-level NFA that constrains what the sampler may pick.
sources:
  - resource: LLM/src/Shared/Grammar.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

A small program of byte-matching instructions run as a Thompson NFA. Byte
level on purpose, since byte-BPE tokens are byte sequences and a codepoint
range is an alternation over its encodings. A fraction of GBNF, covering
tool-call and JSON masking. See `tool-grammar-sampler-perf`.
