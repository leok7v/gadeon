---
type: File
title: LLM/src/Shared/E8.swift
description: The E8 lattice codebook and its LDLQ vector-GPTQ sweep.
sources:
  - resource: LLM/src/Shared/E8.swift
tags: [orientation]
timestamp: 2026-08-19T18:00:00Z
---

Conway and Sloane's closest-point decoder for E8 = D8 union (D8 + 1/2),
bounded to the 56881 points inside norm^2 <= 10 so the codebook is honestly
2 bits per weight over a vector of 8. `gptq` is the LDLQ sweep: groups of 8
columns quantized jointly, whole-group residuals propagated to later columns
through the same Cholesky feedback `Q2GPTQ.sweep` uses.

Reconstruction only -- there is no packed format and no kernel, so it is
reachable only under `--reconstruct`, which `--e8` forces. Results and the
scale-collapse bug that made the first run void are in
`e8-lattice-codebook`.
