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

## 2026-07-25 - Constraint filter before scoring
Filtering candidates after ranking means an unsafe setpoint can exist as a scored option.
Filtering first makes unsafe recommendations structurally impossible rather than merely
unlikely.
