---
type: File
title: LLM/src/Shared/Calculator.swift
description: The engine behind the always-available calculator tool.
sources:
  - resource: LLM/src/Shared/Calculator.swift
tags: [orientation]
timestamp: 2026-08-10T00:10:24Z
---

A hand-written lexer and recursive-descent parser over a FIXED operator and
function table, so a model-authored expression never reaches NSExpression,
whose format grammar can invoke arbitrary selectors. Values are complex, so
the Euler-formula walk a model loves computes instead of erroring.
