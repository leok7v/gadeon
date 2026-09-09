---
type: File
title: LLM/cli/Gemma.swift
description: The gemma-4 gates, the say modes, and the loader that hands the
  shared chat a backend with its attachments already encoded.
sources:
  - resource: LLM/cli/Gemma.swift
tags: [orientation]
timestamp: 2026-09-04T09:30:00Z
---

`loadGemma` runs every gate (`--gemma-gate`, `--gemma-metal-gate`, the
tower, patch, mel, batch, park, image and unified gates) and every say mode
(`--gemma-audio-say`, `--gemma-mic`, `--gemma-image-say`,
`--gemma-video-say`), each of which exits, and otherwise returns the
`LoadedChat` the shared driver runs. `gemmaDrive`, `gemmaEars` and
`gemmaEyes` are the engine and frontend selection the say modes share. The
gates are the point: free-running text proves nothing, and every bug this
architecture hides shows as a diverging hidden state long before a bad word.
