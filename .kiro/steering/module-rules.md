# Module ownership rules

Which rules bind you depends on what you are editing. Ported from the project's
per-module agent definitions.

## `src/sim/`, `tests/physics_check` — Synthetic Mill (M1)

Physical plausibility outranks feature completeness. This output is the foundation every
other module depends on.

Priorities in order:

1. Correct mass balance and speed-dependent transport delay.
2. Planted causal structure with realistic lags, recoverable by causal discovery but not
   trivially obvious from a plain correlation.
3. Realistic failure modes and a 25-45% off-spec rate.
4. Variety across the grade catalogue, including sparse grade pairs.

Never leak simulator ground truth into anything but `meta.json`'s `injected_faults`
field. Always run the plausibility gate before declaring work complete.

## `src/causal/` — Lagged causal discovery (M3)

Produces a lagged causal graph across episodes and ranks loops by impact on breach risk
and stabilisation time.

- Lagged methods only. A contemporaneous correlation matrix is not an acceptable output.
- Every edge reports cause, effect, lag in seconds, strength, and sample size.
- Never read `injected_faults`. Validation against it happens in tests, not in the code
  path.
- Report recovery rate against planted structure as the headline metric.

## `src/twin/`, `src/forecast/`, `src/advisor/`, `src/evidence/` — Twin, forecaster, advisor, evidence (M4-M7)

- The twin is gray-box: physics baseline plus learned residual. Never a pure black box.
- Forecasts carry uncertainty and are always reported against a persistence baseline.
- Constraint filtering happens before candidate scoring. A candidate violating recipe
  limits or actuator rates must never be scored, ranked, or logged as an option.
- Every output is an evidence card. Weights must sum to 1.
- The narrator receives only the card.

## `src/ui/`, `src/feedback/` — Operator surface and feedback loop (M8-M9)

- Follow the `dashboard-language` conventions exactly, especially colour semantics.
- Accept and Reject are equally weighted. Reject requires a reason code.
- Never display a narration that fails the numeral validator.
- Capture screenshots at 1600x1000 into `docs/screenshots/` whenever the UI changes.
