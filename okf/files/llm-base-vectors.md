---
type: File
title: LLM/Base/Vectors.swift
description: The two reductions every engine and every gate needs, written
  once.
sources:
  - resource: LLM/Base/Vectors.swift
tags: [orientation]
timestamp: 2026-09-02T14:00:00Z
---

`argmax` had six identical copies across the four engines and the sampler,
and `cosine` had two spellings in the gemma gates plus two more written
inline in the ViT probe. The gates are why the cosine one matters: they
decide whether a port is correct, so two spellings of one formula is two
answers to that question.

Two properties are load-bearing rather than incidental. `argmax` breaks ties
on the FIRST index, which is what `vDSP_maxvi` does and what the sampler
documented; changing it would move a decode stream. And `cosine` accumulates
in Double whatever it is handed, which is what the probe's Float
accumulation used to get wrong.
