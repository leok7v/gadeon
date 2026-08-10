---
type: File
title: LLM/src/Slugs/Slugs.swift
description: On-device semantic search over a local corpus.
sources:
  - resource: LLM/src/Slugs/Slugs.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

A query never leaves the device; only the resolved article id is fetched. The
two seams are deliberately general, so an embedder and a vector index could
serve a second corpus. See `slugs-wikipedia-tool`.
