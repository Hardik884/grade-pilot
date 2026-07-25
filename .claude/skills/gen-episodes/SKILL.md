---
name: gen-episodes
description: Generate a fresh synthetic episode dataset and run the plausibility gate.
argument-hint: [count]
disable-model-invocation: true
allowed-tools: Bash(python -m src.sim.generate *) Bash(python -m tests.physics_check *) Bash(pytest *)
---

Generate $ARGUMENTS grade-change episodes (default 300 if no count given):

1. Run `python -m src.sim.generate --n $ARGUMENTS --out data/episodes --seed 42`
2. Run `python -m tests.physics_check data/episodes`
3. If the gate fails, report which rule failed and which episodes, then stop. Do not
   proceed to fix the data - fix the simulator.
4. On pass, report: episode count, off-spec rate, mean stabilisation time, and the
   distribution of injected fault types.

An off-spec rate outside 25-45% means the simulator is miscalibrated - flag it.
