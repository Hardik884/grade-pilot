# Grade Change Intelligence (GCI)

Advisory intelligence layer over a paper machine's MD Multivariable MPC. It does NOT
replace control. It predicts basis-weight excursions during grade transitions,
recommends constrained setpoint moves, and explains every suggestion with traceable
evidence.

## Non-negotiable rules

1. **Never emit a recommendation that violates recipe limits or actuator constraints.**
   Constraint filtering happens BEFORE scoring, not after. This is structural, not advisory.
2. **Every prediction and recommendation carries an evidence card.** No bare numbers
   reach the UI. See `#evidence-card`.
3. **All physical quantities carry explicit units.** See `#papermaking-process` for
   canonical units and valid ranges. Reject any value outside its physical range at the
   boundary of the module that produced it.
4. **The LLM narrator only rephrases the evidence card.** It never introduces a fact,
   number, or causal claim not present in the card.
5. **Every model result is reported against a named baseline.** Persistence baseline for
   forecasting, current-practice baseline for advisory. Never report a bare accuracy.

## Repo layout

```
src/
  sim/         M1  Synthetic Mill - physics simulator
  episodes/    M2  Segmentation + labelling + episode store
  causal/      M3  Lagged causal discovery
  twin/        M4  Gray-box forward model (physics + learned residual)
  forecast/    M5  Risk forecaster with uncertainty
  advisor/     M6  Constrained counterfactual search
  evidence/    M7  Provenance assembly + narration
  feedback/    M8  Accept/reject capture + trust calibration
  ui/          M9  Dashboard
data/          Generated episodes, causal graphs, feedback log (gitignored)
tests/         Physics sanity checks + module tests
docs/          Architecture doc + deck source
```

## Conventions

- Python 3.11+, `uv` or venv. Type hints on every public function.
- All time series are pandas DataFrames indexed by `t_sec` (float, seconds from episode start).
- All episode IDs are `EP-{grade_from}-{grade_to}-{seq:04d}`.
- Random seeds are explicit arguments, never global. Default seed 42.
- No module imports from `ui/`. `ui/` imports from everything else.
- Tests live beside the module they test: `tests/test_<module>.py`.

## Commands

```
pytest -q                                                 # test suite
python -m src.sim.generate --n 300 --out data/episodes    # generate dataset
python -m tests.physics_check data/episodes               # physical plausibility gate
python -m src.ui.server                                  # dashboard, port 8000
```

## Reference context

- `papermaking-process` and `episode-schema` load automatically when editing `src/**/*.py`.
- `dashboard-language` loads automatically when editing `src/ui/**`.
- `evidence-card` is on-demand: activate it before touching `src/advisor/`,
  `src/evidence/`, `src/feedback/`, or any UI component that renders a suggestion.

## Working style

- Plan before any change touching more than one module.
- Structural decisions go in `docs/decisions.md` as short dated entries.
- Prefer small, testable functions over notebook-style scripts.
