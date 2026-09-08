---
type: File
title: LLM/src/Base/Vision.swift
description: An image file becomes the CGImage the media encoders scale,
  and the size they scale it to.
sources:
  - resource: LLM/src/Base/Vision.swift
tags: [orientation]
timestamp: 2026-09-05T12:00:00Z
---

`VisionPreprocess` decodes with a cap on the long edge (4096, so a 100 MP
photo never materialises its full bitmap), makes the chip thumbnail with
the EXIF transform applied and the tower decode without it, and turns a
CGImage back into a JPEG for the stored transcript. `nativeSize` is HF's
smart_resize: sides to multiples of the factor, area brought under the
budget with the aspect kept; `native` resamples to it and normalises.
`VLPrompt.defaultPrompt` is the question an image-only turn asks. Both
encoders sit on it, see `one-media-seam-over-every-model`.
