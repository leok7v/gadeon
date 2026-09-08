---
type: File
title: LLM/src/Base/VideoFrames.swift
description: Frames sampled from a video, each with its timestamp.
sources:
  - resource: LLM/src/Base/VideoFrames.swift
tags: [orientation]
timestamp: 2026-09-05T03:30:00Z
---

The frame count is a parameter, because it belongs to a model's processor
rather than to video: gemma names a count, Qwen a rate with a floor and a
ceiling, and both spread the frames endpoint to endpoint. Fewer frames
than asked for is not an error here, since the soft-token count is already
variable per item. `poster` is the one display frame for a clip: the
file's own cover art when it carries any, else the frame a quarter of the
way in, since the opening frame is often black or a fade and the midpoint
can land on a cut. It scales in the generator, which the tower path must
never do (`composer-film-strip`), because this frame is for the chip
and the transcript, not the model.
