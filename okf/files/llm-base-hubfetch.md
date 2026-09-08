---
type: File
title: LLM/src/Base/HubFetch.swift
description: Model download from the Hugging Face Hub, with no dependency.
sources:
  - resource: LLM/src/Base/HubFetch.swift
tags: [orientation]
timestamp: 2026-09-05T12:00:00Z
---

Two plain HTTPS endpoints over URLSession and JSONSerialization: the tree
listing at `api/models/{repo}/tree/{sha}?recursive=true` and the blob at
`{repo}/resolve/{sha}/{path}`, which answers with one redirect to the CDN.
Public repos, no auth. The revision pins to a commit sha BEFORE the first
byte, because pulling file by file from a branch can straddle a push and mix
two commits into a set that loads cleanly and generates garbage.

Files download IN PLACE at their final `{sha}/` paths: each blob streams to
a part file in ranged lanes, is digest-checked against the tree's oid (sha256
for LFS, the git blob sha1 otherwise), and only then moves atomically to its
destination, so a landed file is always whole and verified. An interrupted
download resumes: survivors that re-verify are kept and only the rest
re-download. Completeness is the `.complete` sentinel, never file presence;
`ModelCatalog.isComplete` gates on it, so a partial tree is never a set.
See `hf-xet-breaks-hubfetch`.
