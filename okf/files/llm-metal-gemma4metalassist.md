---
type: File
title: LLM/src/Metal/Gemma4MetalAssist.swift
description: The gemma-4 MTP drafting head on the GPU.
sources:
  - resource: LLM/src/Metal/Gemma4MetalAssist.swift
tags: [orientation]
timestamp: 2026-08-29T02:30:00Z
---

Four narrow layers that fold the trunk's last token and hidden through
pre_proj, attend the TRUNK's KV pools, and leave both a drafted token and a
3840-wide state for the next draft. It owns no cache and cannot run alone.
See `the-12b-assistant-is-an-mtp-head-not-a-draft-model`.
