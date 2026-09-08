---
type: File
title: LLM/src/Base/SoftSpan.swift
description: One attachment resolved into everything a turn needs from it.
sources:
  - resource: LLM/src/Base/SoftSpan.swift
tags: [orientation]
timestamp: 2026-09-05T03:30:00Z
---

The placeholder the template emitted, the id block that replaces it, and the
tower rows spliced over that block. The template decides WHERE a modality
sits in the turn and only the expansion lives here, which is the same split HF
draws between jinja and the processor.
A span may also carry the patch grid its rows came from, which the LM
ropes it at, and the begin/end pair the template wrote around its
placeholder when the block brings its own, see
`one-media-seam-over-every-model`.
