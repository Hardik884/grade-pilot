---
name: physics-qa
description: Audit a module against the process physics reference and the episode schema.
argument-hint: [module-path]
disable-model-invocation: true
context: fork
---

Audit $ARGUMENTS against the `papermaking-process` and `episode-schema` skills.

Check and report as a table of findings, each marked pass or fail:

1. Every physical quantity uses the canonical unit and name from the reference.
2. No hardcoded magic numbers that should come from `meta.json` machine parameters.
3. Transport delay is treated as speed-dependent, not a fixed constant.
4. No module outside `src/sim/` reads `injected_faults` or `bw_true`.
5. Any recommendation path filters constraints before scoring, not after.
6. Episode columns match the contract exactly - no invented names.

Report findings only. Do not edit files.
