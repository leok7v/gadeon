---
type: File
title: LLM/src/Shared/SpecialIndex.swift
description: Finds the earliest control token in one pass instead of one
  whole-text search per special.
sources:
  - resource: LLM/src/Shared/SpecialIndex.swift
timestamp: 2026-08-31T15:00:00Z
---

Holds the specials sorted longest first, plus the set of their first
characters. A scan tests candidates only where one of those characters
occurs, so the earliest and longest match is found without searching the
whole string once per special.

Both tokenizers use it. The per-special search it replaces was 93% of the
gemma tokenizer's time on a 1 MB input.
