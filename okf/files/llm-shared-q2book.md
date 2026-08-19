---
type: File
title: LLM/src/Shared/Q2Book.swift
description: Sweeps a second 2-bit codebook per block and reports what it buys.
sources:
  - resource: LLM/src/Shared/Q2Book.swift
tags: [orientation]
timestamp: 2026-08-19T12:00:00Z
---

A probe, not a shipping path. For each Hessian-backed tensor in chosen
layers it runs GPTQ twice, once with the mid-riser grid alone and once with
`Q2Fit.wide` offering a second grid per block, and prints the change in
relative output error next to the FRACTION OF BLOCKS that chose the second
grid. The pick rate is the load-bearing column; see
`free-sign-bit-codebook` for why, and for the answer (about 1%).

Reaches into the converter's own `plans` / `apply` / `squares` so the tensor
it scores is byte-identical to the one a real emit would write, permutation
and norm offset included.
