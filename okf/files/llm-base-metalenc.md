---
type: File
title: LLM/Base/MetalEnc.swift
description: The typed one-token dispatch surface.
sources:
  - resource: LLM/Base/MetalEnc.swift
tags: [orientation]
timestamp: 2026-09-07T04:40:00Z
---

Each method encodes ONE kernel with the bindings its shader counterpart
expects. Kept apart from the engine so the forward loop reads as an op
sequence and the binding detail lives here.

`gemm` routes by WIDTH: N=1 to the per-type gemv, 1 < N <= narrowMax to a
per-type narrow kernel, wider to a simdgroup-matrix tile, per type where one
exists (`kqTileName` for the four K-quants, `denseTileName` for F16 and BF16
rows a multiple of 32 wide) and the generic `iq_gemm_mm_h` for the rest; F32
stays on `gemmRows`. Q4_0 and Q2_0 each keep their own narrow family; Q4_K,
Q5_K, Q6_K, IQ4_XS and the two 32-weight block types in `spanTypes` (Q8_0,
IQ4_NL) take `gemmNarrowIQ`, which also picks the kernel's SHAPE, one
simdgroup or eight sharing the staged activations, by width and row length
from a measured table, and at width 3 on the four K-quants the device-read
kernel; the other super-blocked types fall to `gemmRows`, N gemv calls. The
span types reach the same gemv family through `iqGemv`; `packedGemv` keeps
Q2_0, Q4_0 and the dense types.

An encoder opened concurrent gets a buffer barrier after every dispatch,
so the op sequence keeps its serial meaning; `parallel` withholds the
barrier between its members and puts one after them, so only dispatches
a caller has declared independent ever overlap.

`attnPaged` splits a head's keys over up to eight threadgroups from 256
keys and reduces the partials, so a long decode context is not one
threadgroup per head.

`try! ctx.pipeline(name)` is sound because `MetalContext.prewarm` builds
every buildable pipeline at engine init: the lookup is a dictionary hit and
can only fail on a name this library does not carry, a typo that fails on
the first dispatch of every run. The prefill tile can be forced regardless
of width for `QwenMetalKernelBench`'s A/B. The vision GEMM takes any K
because the kernel guards the tail; it is a simdgroup-matrix kernel, so
prewarm skips it where the hardware cannot build it, and the guard that
keeps it from ever being asked for is `QwenMetalBackend.tower`, which
refuses without matrix units.
