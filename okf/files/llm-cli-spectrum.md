---
type: File
title: LLM/cli/Spectrum.swift
description: Prints the eigen-spectrum summary of every collected Hessian.
sources:
  - resource: LLM/cli/Spectrum.swift
tags: [orientation]
timestamp: 2026-08-19T06:19:12Z
---

`gadeon-cli x --spectrum <hess-dir>` runs `Eigen.of` over each .h32 and
reports K, condition number, the share of the trace in the top 1% of
eigenvalues, the single largest share, and the participation ratio.

Those five numbers pick the next lever: energy concentrated in few directions
means a low-rank residual corrects it, energy spread flat means a rotation is
the only route. See `gptq-at-2b`.
