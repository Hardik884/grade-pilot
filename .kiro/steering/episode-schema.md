---
inclusion: fileMatch
fileMatchPattern: 'src/**/*.py'
---

# Episode data contract

Every module reads and writes this shape. Do not invent alternative column names.

## On disk

```
data/episodes/
  EP-G12-G07-0001/
    series.parquet     # time series
    meta.json          # episode metadata + labels
  EP-G12-G07-0002/
  ...
  index.parquet        # one row per episode, flattened meta for fast queries
```

## series.parquet

Index: `t_sec` (float64, 0.0 at transition trigger, 1 Hz sampling, negative values for
pre-transition baseline, minimum 120 s of pre-roll).

| Column | dtype | Notes |
|---|---|---|
| `bw`, `moist`, `ash`, `caliper` | float64 | Measured (delayed + noisy) quality |
| `bw_true` | float64 | Ground-truth sheet value at the headbox, simulator only |
| `bw_sp`, `moist_sp`, `ash_sp` | float64 | Active setpoint, ramps during transition |
| `stock_flow`, `filler_flow`, `steam_p`, `speed` | float64 | Manipulated variables |
| `stock_cons` | float64 | Slow-varying disturbance |
| `theta_sec` | float64 | Instantaneous transport delay |
| `phase` | category | `pre`, `ramp`, `settle`, `steady` |
| `op_action` | string \| null | Operator intervention, null when none |
| `alarm` | string \| null | Active alarm tag, null when none |

## meta.json

```json
{
  "episode_id": "EP-G12-G07-0001",
  "grade_from": "G12",
  "grade_to": "G07",
  "grade_from_props": {"bw": 82.0, "ash": 18.0, "moist": 6.2, "caliper": 105.0},
  "grade_to_props":   {"bw": 64.0, "ash": 12.0, "moist": 5.8, "caliper": 88.0},
  "machine": {"trim_m": 6.4, "scanner_distance_m": 140.0, "retention": 0.78,
              "filler_cons_pct": 30.0},
  "recipe_limits": {
    "stock_flow": [600.0, 3600.0],
    "filler_flow": [0.0, 700.0],
    "steam_p": [1.2, 5.5],
    "speed": [500.0, 1500.0]
  },
  "actuator_rates": {"stock_flow": 120.0, "speed": 60.0, "steam_p": 0.4},
  "injected_faults": ["ramp_desync_filler_lead"],
  "seed": 42,
  "labels": {
    "off_spec": true,
    "max_dev_pct": 3.9,
    "breach_t_sec": 148.0,
    "stabilisation_t_sec": 412.0,
    "broke_tonnes": 2.4
  }
}
```

`injected_faults` is simulator ground truth. **It must never be read by M3–M6.** It
exists only so tests can verify that causal discovery recovers what was planted.

`actuator_rates` are maximum rates of change per minute, used by the advisor's
constraint filter.

`recipe_limits` and `actuator_rates` are in each variable's canonical unit from the
paper machine process reference — note that `stock_flow` is **m³/h**, not L/min.

`machine.filler_cons_pct` is the filler slurry consistency in %, needed by the mass
balance. `machine.retention` applies to both fibre and filler.

## Label definitions

- `off_spec`: true if `abs(bw - bw_sp) / bw_sp > 0.025` at any point after `t_sec = 0`.
- `max_dev_pct`: maximum of that quantity over the episode, as a percentage.
- `breach_t_sec`: first `t_sec` where the threshold is exceeded, null if never.
- `stabilisation_t_sec`: first `t_sec` after which `bw` remains inside the band for a
  continuous 120 s window. Null if it never stabilises within the episode.
- `broke_tonnes`: off-spec production mass, `sum(bw * speed * trim * dt) / 1e6` over
  off-spec samples, in tonnes.

## Grade catalogue

`data/grades.json` maps each grade code to its property vector plus a
`nominal_speed_m_min`. Speed is a property of the grade, not the machine: light grades
run fast (1200–1500 m/min) and heavy grades slow (450–700), because the actuator
envelope cannot otherwise cover the catalogue — a heavy, high-ash grade at full speed
demands more filler flow than the recipe limit allows.

The advisor's grade-space embedding uses `[bw, ash, moist, caliper]` normalised per
dimension; `nominal_speed_m_min` is an operating parameter, not part of the embedding.
Transitions are represented as the vector `to_props - from_props`, so unseen grade pairs
are retrievable by proximity in this space.

## Rules

- Write episodes atomically: build in a temp dir, then rename.
- `index.parquet` is derived, never hand-edited. Regenerate it from `meta.json` files.
- Any module that widens this schema must update this steering file in the same commit.
