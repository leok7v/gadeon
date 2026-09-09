---
type: File
title: LLM/src/Base/JSONBytes.swift
description: The encoder for a blob that will be compared, hashed or diffed.
sources:
  - resource: LLM/src/Base/JSONBytes.swift
tags: [orientation]
timestamp: 2026-09-02T14:20:00Z
---

`JSONEncoder` does not order keys unless asked, so two encodes of one value
differ byte for byte. Named for the INTENT rather than the setting, because
the bug it prevents is someone reaching for a bare `JSONEncoder()` at a site
where the bytes are later compared.

Deliberately not used everywhere. A saved conversation
(`App/ConversationStore.swift`) keeps the plain encoder: nothing compares
those files, they are rewritten constantly, and key order is invisible to
the only reader they have.
