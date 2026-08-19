---
type: File
title: LLM/src/Shared/Q2E8Convert.swift
description: HuggingFace safetensors to a Qwen3.5 GGUF, driven by the source.
sources:
  - resource: LLM/src/Shared/Q2E8Convert.swift
tags: [orientation]
timestamp: 2026-08-18T21:32:34Z
---

Geometry from config.json, the tensor plan from the safetensors index, the
tokenizer from tokenizer.json plus tokenizer_config.json, and every
*_config.json carried verbatim as a qwen35.source.* key. A name with no
mapping is a hard error, which is what keeps the vision and MTP towers in the
file.

The mapping contracts it honours, and the per-tensor gate that catches each
one, are `hf-to-gguf-in-swift` and `q2x-block-and-qwen38-gguf`.
