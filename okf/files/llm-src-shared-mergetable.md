---
type: File
title: LLM/src/Shared/MergeTable.swift
description: The merge-rank BPE both tokenizers run, with the split that
  keeps it linear.
sources:
  - resource: LLM/src/Shared/MergeTable.swift
timestamp: 2026-08-31T15:00:00Z
---

Symbols are interned to Int32 at init, so the merge loop hashes a packed
pair of integers and allocates nothing; Strings reappear only for the
symbols it returns. `segments` cuts the input where no merge rule could ever
join the two adjacent characters, which is what bounds the loop's input and
is exact rather than heuristic.

Shared because this loop was copy-pasted into both tokenizers, and a fix
landed on one of them and missed the other.
