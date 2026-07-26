---
inclusion: fileMatch
fileMatchPattern: 'src/**/*.py'
---

# Paper machine process reference

Authoritative for units, ranges, and governing equations in this repo. Any generated
value outside these ranges is a bug, not a disturbance.

## Canonical variables

| Symbol | Variable | Unit | Typical range | Role |
|---|---|---|---|---|
| `bw` | Basis weight | g/m² | 40–300 | Controlled quality (primary) |
| `moist` | Moisture | % | 4–9 | Controlled quality |
| `ash` | Ash content | % | 5–30 | Controlled quality |
| `caliper` | Caliper | µm | 60–400 | Controlled quality |
| `stock_flow` | Thick stock flow | m³/h | 500–4000 | Manipulated |
| `stock_cons` | Stock consistency | % | 2.5–4.5 | Slow-varying |
| `filler_flow` | Filler flow | L/min | 0–800 | Manipulated |
| `steam_p` | Dryer steam pressure | bar | 1–6 | Manipulated |
| `speed` | Machine speed | m/min | 400–1800 | Manipulated |
| `trim` | Trim width | m | 3–10 | Constant per machine |

## Governing relations

**Basis weight from mass balance** (the physics baseline for the twin):

```
stock_flow_L_min = stock_flow_m3_h * 1000 / 60

bw = (stock_flow_L_min * stock_cons * 10 + filler_flow * filler_cons * 10) / (speed * trim) * retention
```

where both flows enter the balance in L/min, consistencies are %, speed is m/min, trim
is m, and `retention` is the first-pass retention fraction (0.6–0.9). Result is g/m².
The factor 10 converts L/min × % to g/min assuming ~1 kg/L slurry density.

**Watch the units on `stock_flow`:** it is tabled in m³/h and *must* be converted to
L/min before it enters the balance. Skipping the conversion under-predicts basis weight
by ~17×. `filler_flow` is already L/min and needs no conversion. A 6.4 m trim machine at
1000 m/min running 946 m³/h of 3.5% stock and 404 L/min of 30% filler at 0.78 retention
makes 82 g/m² — use this as the arithmetic check on any implementation.

Key consequence: **basis weight is inversely proportional to machine speed.** A speed
ramp with no compensating stock ramp drives basis weight down. This is the dominant
grade-change dynamic and any simulator must reproduce it.

**Transport (dead) time** — the reason transitions overshoot:

```
theta_sec = 60 * distance_m / speed_m_per_min
```

The QCS scanner sits at the reel, typically 80–200 m downstream of the headbox. At
1000 m/min that is 5–12 s of pure delay; at 500 m/min it doubles. Delay is
speed-dependent and therefore **changes during the transition itself**. Model it as a
variable-lag buffer, never a fixed constant.

Additional lags before the delay: wet-end mixing and retention dynamics behave as a
first-order lag with time constant 20–60 s. Dryer section thermal response is slower,
60–180 s, which is why moisture recovers after basis weight.

**Scanner behaviour.** The QCS sensor traverses the sheet, so a reading is a diagonal
sample, not a point measurement. Scan period is 20–45 s. Report MD (machine-direction)
values as scan averages; do not pretend a continuous point measurement exists. Sensor
noise on basis weight is roughly 0.3–0.8 g/m² standard deviation.

## Grade change behaviour

A grade change is a coordinated ramp of setpoints from grade A to grade B, executed over
3–15 minutes. Realistic failure modes to reproduce in simulation:

- **Overshoot from delay** — controller reacts to a stale measurement and over-corrects.
- **Ramp desynchronisation** — filler flow ramps ahead of stock flow, so ash rises before
  basis weight recovers, dragging basis weight off target.
- **Speed-first transitions** — speed ramped before stock flow, causing an immediate
  basis-weight dip proportional to the speed ratio.
- **Consistency drift** — slow thick-stock consistency wander that biases the mass balance.
- **Steam-limited drying** — a large basis-weight increase without adequate steam headroom
  leaves moisture high and caliper off, extending stabilisation.

Off-spec definition for this project: `abs(bw - bw_target) / bw_target > 0.025`.

Stabilisation time: first instant after which `bw` stays within the spec band for a
continuous 120 s window.

## Physical plausibility gate

Reject generated or predicted data if any hold:

- any variable outside its range in the table above
- basis weight changes faster than 15 g/m² per minute
- machine speed changes faster than 100 m/min per minute
- basis weight responds to a manipulated variable with zero lag
- ash exceeds basis weight in absolute mass terms
- moisture and steam pressure move in the same direction at steady state

## Units discipline

Suffix every variable name with its unit when ambiguity is possible (`theta_sec`,
`speed_m_min`). Convert at module boundaries, never mid-calculation.
