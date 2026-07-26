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
