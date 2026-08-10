---
type: File
title: LLM/src/Shared/GGUF.swift
description: A GGUF v3 reader over the mapped file.
sources:
  - resource: LLM/src/Shared/GGUF.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Header, KV metadata and the tensor directory, exposing each tensor's dims,
ggml type and base pointer into the mapped region. It reads and never writes.

The descriptor and the pages are a separate object on purpose, so that a
parse which throws still frees them. See `a-throwing-init-runs-no-deinit`
and `bonsai-gguf-arch`.
