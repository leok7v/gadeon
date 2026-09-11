---
type: File
title: LLM/Base/Calculator.swift
description: The engine behind the always-available calculator tool, and the
  shapes models actually send it that it tolerates on purpose.
sources:
  - resource: LLM/Base/Calculator.swift
tags: [orientation]
timestamp: 2026-09-04T11:00:00Z
---

A hand-written lexer and recursive-descent parser over a FIXED operator and
function table, so a model-authored expression never reaches NSExpression,
whose format grammar can invoke arbitrary selectors. Values are complex, so
the Euler-formula walk a model loves computes instead of erroring.

Every tolerance below was added after a model sent that exact shape and
burned rounds on the error, each with a test in CalculatorTests: `**` as
`^`; `;` as `,` between arguments; names case-folded (`POW`, `Math.PI`) and
the `math.` / `np.` namespaces dropped; a trailing `= ?` / `?` / `=`; `$`
signs; postfix `%` as a percentage where no operand follows; implicit
multiplication (`2pi`, `2sin(1)`); a reversed assignment (`350 * 0.4 = X1`);
`round(x, n)` to n decimals (Qwen3.5-9B, 2026-09-04, sent it to tidy a
result and got an error for the arity).
