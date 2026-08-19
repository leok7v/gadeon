---
type: File
title: LLM/cli/Q2E8Convert.swift
description: The command line over the safetensors-to-GGUF converter.
sources:
  - resource: LLM/cli/Q2E8Convert.swift
tags: [orientation]
timestamp: 2026-08-18T21:32:34Z
---

`gadeon-cli x --q2e8-convert SRC OUT [--trunk T] [--embd T] [--wide LIST]
[--gate REF]` picks the trunk block type, the token-embedding type and the
tensors held wider than the trunk, and optionally reports every written tensor
against a reference GGUF. See `hf-to-gguf-in-swift`.
