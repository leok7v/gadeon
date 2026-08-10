---
type: File
title: LLM/cli/main.swift
description: The CLI entry point and its mode dispatch.
sources:
  - resource: LLM/cli/main.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

A multi-turn chat over the state-carrying decode trunk, with turns from
arguments or stdin, reporting rates to stderr and text to stdout.

Fixture defaults resolve from the REPO ROOT, not from the package directory.
See `cli-runs-from-repo-root`.
