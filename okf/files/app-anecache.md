---
type: File
title: App/AneCache.swift
description: Ownership of the ANE compile cache, established by
  observation.
sources:
  - resource: App/AneCache.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

The system never collects inside an OS-build namespace and the cache's keys
are opaque, so ownership is worked out by diffing the entry list around each
set's build and persisting the claim per set and OS build. GC deletes only
unclaimed, aged entries, across the live cache AND the shadow, and only once
every installed non-GGUF set holds a non-empty claim. Deleting a model
releases its claim and whatever no surviving claim still covers.

It stops at the cache. It compiles nothing and loads nothing.
See `ane-e5bundlecache-forensics` and `ane-shadow-outlived-its-models`.
