---
type: File
title: LLM/src/Shared/E8P.swift
description: QuIP#'s padded E8 codebook, 2^16 points over 8 weights.
sources:
  - resource: LLM/src/Shared/E8P.swift
tags: [orientation]
timestamp: 2026-08-19T21:00:00Z
---

The 256-entry table of |D8hat| absolute values (227 at norm^2 <= 10 plus 29
padding at 12), the parity rule that forces the 8th sign, and an exact
nearest-point search that costs 2 * 256 rather than 2^16 because for a fixed
entry the best signs are sign(y) and a wrong parity is repaired at the
coordinate minimising s_i*|y_i|.

Exactly 2.000 bits per weight, so it lands on Q2_X's budget once the shared
fp16 scale is added. Reachable through `--e8p`, which routes into
[[llm-shared-e8]]'s LDLQ sweep with `padded` set. Numbers, the two gates that
caught a bad first reading, and the low-rank incompatibility are in
`e8-lattice-codebook`.
