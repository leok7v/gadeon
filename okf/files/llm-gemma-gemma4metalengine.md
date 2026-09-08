---
type: File
title: LLM/src/Gemma/Gemma4MetalEngine.swift
description: The gemma-4 text forward on the GPU.
sources:
  - resource: LLM/src/Gemma/Gemma4MetalEngine.swift
tags: [orientation]
timestamp: 2026-09-07T03:30:00Z
---

Every kernel for one token is encoded onto ONE command buffer and committed
once, so a decoded token costs one dispatch stream and one sync; the
layer loop ends each layer with the next layer's input norm, fused with
the residual add and the layer scale, and a clamped projection is one
dispatch. The weights
are MIXED, so a dispatch names a tensor and lets the encoder pick the kernel
rather than assuming a type. The CPU engine is the oracle.

The KV is not one geometry: 11 sliding layers window at 512, L13 is sliding
but stays unwindowed because layers 15 and up read past its own window, the
full layers are 512 wide, and 20 layers own no pool at all; the window is
applied at read time as an absolute-position bound, so a shared layer
windows its source's history exactly as HF does. The chunk scratch is sized
once at init and the capacity is enforced: raising the batch past it wrote
off the end of every chunk buffer and took WindowServer down. The capacity
is one on a GPU without matrix units, where the batched forward cannot be
built, so `extend` degrades to the token-by-token path rather than having
no path. The stop flag is the same lever the Qwen engine carries, cleared at
each extend so a prior turn's Stop cannot kill this one.
