---
type: File
title: LLM/src/Shared/HubFetch.swift
description: Model download from the Hugging Face Hub, with no dependency.
sources:
  - resource: LLM/src/Shared/HubFetch.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

Two plain HTTPS endpoints over URLSession. The revision pins to a commit sha
BEFORE the first byte, because pulling file by file from a branch can straddle
a push and mix two commits into a set that loads cleanly and generates
garbage. Completeness is a sentinel, never file presence.
See `hf-xet-breaks-hubfetch`.
