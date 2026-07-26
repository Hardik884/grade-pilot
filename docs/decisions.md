# Decision log

Short dated entries. One paragraph each: what was decided, and why.

## 2026-07-25 - Gray-box twin over black-box classifier
Basis weight follows a known mass balance. Modelling only the residual keeps the system
credible to process engineers and works with far less data than learning the full
process. Consequence: the simulator must be physically correct, or the residual absorbs
simulator error instead of process dynamics.

## 2026-07-25 - Lagged causal discovery over correlation matrix
Process variables are heavily cross-correlated, so a correlation heatmap cannot separate
cause from effect or recover lags. Lagged methods produce actionable edges with a lag in
seconds. Validation is against planted simulator structure.

## 2026-07-25 - Thick stock flow is m3/h, not L/min
The process reference tabled `stock_flow` as 500-4000 L/min, which is jointly infeasible
with its own mass balance: at 1000 m/min on a 6.4 m trim machine, maximum legal flows
yield 46 g/m2 only by running 53% ash against a 30% ceiling, and capping ash legally
gives 31 g/m2 - below the 40 g/m2 floor. The schema's own example grade of 82 g/m2
requires ~15,800 L/min of stock, 4x the stated ceiling, but 946 m3/h, which sits
mid-range. The numbers were right and the unit label was wrong. Both skills corrected.
Consequence: the mass balance must convert stock flow by 1000/60 before use; omitting
the conversion under-predicts basis weight by ~17x and would be silently absorbed by
the twin's learned residual.

## 2026-07-25 - Machine speed is a grade property
Basis weight is inversely proportional to speed, so a single machine speed cannot serve
a catalogue spanning 45-160 g/m2 within fixed actuator limits - a heavy high-ash grade
at full speed demands roughly twice the filler flow the recipe permits. Each grade
therefore carries its own nominal speed, light grades fast and heavy grades slow. This
matches real mill practice and makes the speed/basis-weight tradeoff emerge from the
catalogue rather than being asserted.

## 2026-07-26 - Basis-weight loop gain is scheduled on loop lag
A PI controller around a dead time is stable only while its gain stays small
relative to that dead time. The basis-weight gain was fixed across the catalogue,
but the slowest grade (505 m/min) carries about three times the loop lag of the
fastest, so one tuning could not serve both: the aggressive-loop fault that merely
overshot at 1400 m/min diverged at 505, driving basis weight 24 g/m2 past target and
still climbing after the ramp closed. Gains now follow the standard dead-time
relation, Kc proportional to 1/lag and Ti proportional to lag, against a reference
lag of 85 s. Consequence: fault severity is comparable across grades rather than
silently speed-dependent.

## 2026-07-26 - Overshoot fault sized against the rate limit
`overshoot_from_delay` was written on the assumption that the loop's own stability
margin caps the resulting excursion. It does not - past the margin a dead-time loop
diverges rather than overshooting. The original gain multipliers pushed basis weight
16-24 g/m2 beyond target, more than the entire transition on the narrower grade
pairs, and breached the 15 g/m2/min plausibility limit on six of 300 episodes. Sized
now to overshoot by a few percent: still well outside the 2.5% spec band, so the
episode is off-spec and worth advising on, but a physical excursion rather than a
divergence. Off-spec rate moved 42.0% -> 37.7%, still mid-band.

## 2026-07-26 - The zero-lag gate rule tests instantaneity, not dead time
The plausibility rule "basis weight responds with zero lag" cannot be tested by
cross-correlation on this data: the loop is closed, so every manipulated variable is
driven by the same setpoint ramp and correlates with basis weight at lag 0 by
construction, and the scanner's 20-45 s zero-order hold is coarser than the ~8 s
transport delay being looked for. Both a correlation estimator and an onset
comparison produced false failures on episodes whose measurement provably lagged the
headbox by 30-40 s. The rule now fits the mass balance against lag and rejects an
instantaneous fit tighter than the scanner noise floor, which is the gross defect it
can actually decide. The delay mechanism itself is pinned by unit test instead.
Consequence: the gate no longer claims to verify dead time - `tests/test_sim.py`
does, directly on the mechanism.

## 2026-07-25 - Constraint filter before scoring
Filtering candidates after ranking means an unsafe setpoint can exist as a scored option.
Filtering first makes unsafe recommendations structurally impossible rather than merely
unlikely.

## 2026-07-26 - Retrieval matches grade properties, not grade codes
30 of the grade pairs in the generated dataset occur exactly once, so code matching
returns zero comparable episodes precisely on the unusual transitions where an operator
most wants help. Episodes are embedded instead as `to_props - from_props` over
`[bw, ash, moist, caliper]`, min-max normalised across the catalogue, and neighbours are
the nearest vectors. `nominal_speed_m_min` stays out of the embedding: it is an operating
parameter, not a property of the paper. Consequence: a singleton pair still retrieves nine
usable neighbours, and the neighbourhood is drawn from transitions with comparable
property deltas rather than comparable labels.

## 2026-07-26 - Neighbour comparison is on fractional progress, not absolute setpoints
Manipulated variables cannot be compared across grades in absolute units: the catalogue
spans 505-1400 m/min and stock flows differ by more than 2x, so a neighbour's stock flow at
90 s carries no information about this episode's. The advisor compares
`(v(t) - v(0)) / v(0)` instead, which transfers across grades and is also how an operator
states it - "the ones that worked had stock 6% up by now, you are 2% up".

## 2026-07-26 - History picks the variable, physics picks the direction
Imitating successful neighbours is directionally blind. On the first end-to-end run the
advisor proposed raising filler on a sheet it had just called heavy, because the
neighbours happened to be running more filler at that point. Candidate directions are now
gated by the mass balance: a heavy sheet gets less delivered mass or more speed, whatever
history did. `steam_p` is excluded as a basis-weight lever entirely - it acts on moisture,
and its correlation with basis weight during a ramp is the co-movement the project's causal
work exists to distinguish from a cause.

## 2026-07-26 - The predictor's deviation is a magnitude; the advisor must re-sign it
`Predictor.predict` builds `predicted_max_dev_pct` from `max(abs(dev))` plus a residual
correction, so it is unsigned. Consumed as if signed, it inverted the corrective direction
on every light sheet in the dataset. The advisor takes the magnitude from the gray-box and
the sign from `PhysicsProjection.signed_dev_at_max_pct`, and the evidence card reads the
signed value off the suggestion rather than re-deriving it, so the claim and the action
cannot disagree about which way the sheet is off.

## 2026-07-26 - Physics attribution is computed over the validity window
Attributing the excursion to the whole remaining ramp produced numbers like "speed alone
takes 62.8% out of the sheet" on a G01 to G11 change, which is arithmetically true of a
1357 to 505 m/min ramp and useless as an explanation, because the setpoint is travelling
with it. Each variable is now advanced at its observed rate for as long as the open-loop
projection is valid - the composed measurement lag - and clamped at its target. The mass
term is also split into fibre and filler, because naming "mass" while quoting stock-flow
numbers is wrong on every high-ash to low-ash change, where filler is the stream moving.

## 2026-07-26 - A move too small to matter is not a suggestion
Neighbour imitation produces arithmetically real but useless proposals: "filler back
2 L/min" against a 3.2% excursion appeared on 9 of the first 91 cards. Candidates now carry
a mass-balance estimate of their effect on basis weight and must clear a quarter of the
spec band, 0.625%, to be scored. Across the dataset this drops 69 candidates and 24
would-be cards. An advisory system that issues noise gets switched off, so silence is the
better output.

## 2026-07-26 - Feedback rows carry the dominant source type
The question the feedback log exists to answer is which kind of evidence operators trust -
cards where physics carried the claim, or cards where the learned residual did. Recovering
that after the fact would mean retaining every card indefinitely, so the one derived field
is denormalised onto the decision row. Everything else is the contract as specified:
card, episode, decision, reason code, timestamp.

## 2026-07-26 - Dashboard is static HTML over a thin Flask API, not Streamlit
Streamlit re-runs the whole script on every widget change, which for a replay that steps
a clock is a full round trip per tick, and it owns the markup so the monochrome-chrome /
colour-only-in-charts rule from `dashboard-language` cannot be enforced. The dashboard is
now `src/ui/static/` served by `src/ui/server.py`: plain HTML, CSS and vanilla JS with
Chart.js from a CDN and a vendored fallback, no build step, and eight JSON endpoints that
only assemble calls into loader, predictor, advisor, evidence and the feedback store. The
predictor is fitted once at boot, roughly 3.5 s, and prediction plus advice are memoised
on the 5 s replay grid, so a step costs about 70 ms and the replay stays responsive.
Consequences: Flask is a new dependency, the replay clock lives in the browser rather than
in Python, and one process, `python -m src.ui.server`, serves both the API and the page.

## 2026-07-26 - The forecast drops its residual term before the 90 s horizon
The residual model's features are cut at the 90 s feature horizon, so handing them to the
predictor at an earlier replay position would let the forecast see data the operator does
not have. Before 90 s the dashboard requests physics only and labels the header
accordingly; the alternative, rebuilding features at each replay step, costs a full
feature pass per request and still would not be what the model was trained on.
Consequence: early-transition forecasts are weaker on purpose, and the panel says so
rather than quietly borrowing hindsight.

## 2026-07-26 - Amber is a forecast, red is measured sheet
The colour table assigns amber to at-risk and red to off-spec, which on a chart carrying
both a prediction and a measurement resolves to: predicted excursions are amber, sheet
that actually measured outside the band is red. The timeline also draws the open-loop
validity horizon as a labelled line, because the forecast is plotted 60 s past it for
context and a crossing to the right of that line is flagged but not claimed as a breach.
Consequence: the breach callout carries one of two texts depending on which side of the
validity horizon it falls on, and no excursion is ever annotated as a breach outside the
window where the mass balance is a statement about sheet that already exists.
