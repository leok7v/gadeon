---
type: File
title: LLM/Base/Media.swift
description: The one seam every attachment crosses on its way into a turn.
sources:
  - resource: LLM/Base/Media.swift
tags: [orientation]
timestamp: 2026-09-05T00:30:00Z
---

`MediaEncoder` turns an image, a clip or a video into `Attached`, the
content parts and the `SoftSpan`s a turn carries, and says which modalities
it takes. `Gemma4Media` and `QwenMedia` are the two encoders behind it; the
driver, the transcript and the view model see only this. `VideoStrip`
accumulates a video's spans frame by frame for both encoders: a stamp, a
bracket of placeholders, the rows, and the grid; `span(wrap:)` says whether
the block replaces the template's own bracket.
