---
type: File
title: LLM/src/Slugs/MiniLM.swift
description: A sentence encoder, the embedder behind the semantic index.
sources:
  - resource: LLM/src/Slugs/MiniLM.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

A pure-Swift port that parses the GGUF, dequantizes, tokenizes with WordPiece,
runs the encoder blocks and returns a unit vector. An EMBEDDING model, so no
KV cache, no causal mask and no sampling.
