# Results

Every number here is produced by the commands in the README against 300 generated
episodes at seed 42. Reproduce with `python -m src.analysis.impact` and
`python -m src.analysis.predictor`.

Dataset: 300 episodes, 240 train / 60 test, stratified on `off_spec`. Test breach rate
0.383, so 23 of the 60 held-out episodes contain a real breach. All predictions are made
from the **first 90 seconds** of each transition. Decision thresholds are calibrated on
the training split only and applied blind to test — choosing them on test would be
leakage.

## Breach prediction from 90 s of data

| Model | Accuracy | Precision | Recall | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| Naive (current deviation > 1.5%) | 0.650 | 0.750 | 0.130 | 3 | 1 | 20 | 36 |
| Physics only | 0.700 | 0.778 | 0.304 | 7 | 2 | 16 | 35 |
| **Gray-box (physics + residual)** | 0.767 | 0.667 | 0.783 | 18 | 9 | 5 | 28 |

Thresholds calibrated on train: gray-box 2.45%, physics-only 2.75%.

Max-deviation MAE: physics-only 1.163%, gray-box 0.939%.

### Why recall is the metric that matters

The two error types are not symmetric in cost. A false negative is a missed breach: the
sheet goes off-spec, and every second of off-spec production at ~1000 m/min on a 6.4 m
trim is broke that must be repulped. A false positive is an advisory the operator reads
and dismisses — it costs a few seconds of attention. Optimising accuracy on a dataset
where 62% of episodes are clean rewards a model for staying quiet, which is precisely the
failure mode that makes an advisory system worthless in a control room.

That trade is visible in the table. The naive rule has *better precision* than the
gray-box model (0.750 vs 0.667) and is nearly useless, because it finds 3 of 23 breaches.
The gray-box model gives up precision to find 18 of 23. Nine false positives across 60
episodes is roughly one spurious advisory per seven transitions, which is an acceptable
price for catching six times as many real failures.

Precision was not allowed to fall freely, though. A threshold recalibration targeting
recall directly was tested and **rejected**: it reached 0.945 mean recall but drove test
precision to 0.503, below the 0.6 floor, in 15 of 15 random splits. See "Tuning that was
tried and rejected" below.

### Why the physics-only column is the interesting one

Physics alone triples the naive recall (0.130 → 0.304) using no learned parameters at
all — just the mass balance evaluated on current setpoints instead of on the stale
measurement. The residual model then more than doubles it again (0.304 → 0.783) by
absorbing what 90 s of data cannot identify: the per-episode wet-end time constant,
moisture coupling into the QCS total-weight reading, and controller behaviour through the
settle.

The contribution split is **67% physics / 33% learned residual**, with physics mean
absolute deviation 1.966% against a mean correction of 1.012%. This is checked
programmatically on every run: if the learned term ever exceeds 40% of the answer, the
report prints `MODEL IS DOING TOO MUCH - check physics path` rather than quietly
reporting a good score. Residual mean absolute error is 1.097% with standard deviation
1.376%.

## Impact ranking with discovered lags

Spearman-style association between each manipulated variable and measured basis weight,
swept over lags 0–120 s in 5 s steps, averaged across 300 episodes. Target is measured
`bw`, scan-period smoothed and differenced.

| Rank | Variable | Discovered lag | Strength at best lag | Strength at zero lag | Gain from lag |
|---|---|---|---|---|---|
| 1 | `speed` | 40 s | 0.809 | 0.732 | +0.078 |
| 2 | `filler_flow` | 40 s | 0.662 | 0.593 | +0.070 |
| 3 | `steam_p` | 80 s | 0.581 | 0.455 | +0.125 |
| 4 | `stock_flow` | 80 s | 0.487 | 0.372 | +0.115 |
| 5 | `stock_cons` | 35 s | 0.210 | 0.147 | +0.063 |

**These lags recover the simulator's planted causal structure without being told it.**
The ranking code reads measured columns only; `injected_faults` and `bw_true` are
simulator ground truth and are never available to it. The structure it recovers:

- `speed` and `filler_flow` act on the sheet immediately, so their discovered lag of
  40 s is the composed measurement lag itself (median 41.4 s — see below). The sheet
  changed at once; the scanner took 41 s to say so.
- `stock_flow` acts through the wet-end mixing volume, so it carries the composed
  measurement lag *plus* the wet-end time constant. Discovered at 80 s against
  41.4 + ~40 = ~81 s expected.
- `steam_p` acts through the dryer section's slow thermal response, likewise at 80 s.

The separation between the 40 s group and the 80 s group is the physically meaningful
result: it tells an operator which loop acts first and how far ahead of the display each
one moves.

### Why a correlation heatmap cannot produce this

Raw level correlation against basis weight scores every loop between **0.351 and 0.967**.
Everything correlates with everything, because a grade change ramps all of them together
— the setpoint ramp is a common cause. A heatmap therefore ranks loops by how hard they
were ramped, not by how much they moved the sheet, and carries no timing information at
all. The lag sweep is what separates them, and the timing is the actionable part.

The dashboard shows both views on a toggle, which makes the difference concrete:

| Lagged discovery | Zero-lag correlation |
|---|---|
| ![lagged](screenshots/03-impact-ranking-lagged.png) | ![zero lag](screenshots/04-impact-ranking-zero-lag.png) |

Left: five loops cleanly separated, each labelled with the lag that produced its peak.
Right: the same five loops with no lag applied — bars near-saturated and the ordering
uninformative. Only the left panel tells an operator which loop to reach for.

## Lag decomposition

The claim the whole system rests on is that the operator's display is stale. This
quantifies it:

| Component | Median | Notes |
|---|---|---|
| Transport delay (`theta_sec`) | 7.16 s | Headbox to scanner at machine speed |
| Scanner traverse and hold | 33.5 s | QCS is a traversing sensor, not a point measurement |
| **Composed measurement lag** | **41.44 s** | p10 30.51 s, p90 51.10 s |

Transport delay alone understates staleness by roughly **5×**. Any evidence card quoting
`theta_sec` on its own would materially mislead the operator, so the composed figure is
what the UI shows and `theta_sec` is never quoted alone.

## Stabilisation-time ranking — directional only, not significant

The same lagged-association machinery was pointed at stabilisation time (how long the
mill takes to settle after the setpoint ramp completes). **No feature reaches
significance at p < 0.05.** Reported honestly:

| Feature | Spearman rho | p-value | Significant at 0.05 |
|---|---|---|---|
| `dev_at_90s` | −0.166 | 0.112 | No |
| `steam_p` | 0.081 | 0.439 | No |
| `speed_stock_desync` | 0.070 | 0.507 | No |
| `speed` | 0.069 | 0.509 | No |
| `effective_measurement_lag` | 0.058 | 0.579 | No |
| `stock_flow` | −0.042 | 0.687 | No |
| `stock_cons` | 0.041 | 0.696 | No |
| `bw_step_size` | −0.031 | 0.771 | No |
| `filler_flow` | 0.029 | 0.780 | No |

Fitted on 93 episodes, not 300. Stabilisation time is degenerate across the full dataset:
80% of episodes are already in-band when the setpoint ramp completes, so the value piles
up at zero. Tightening the band does not rescue it — a 0.5% band leaves 195 of 300
episodes never settling at all. The ranking is therefore fit on the off-spec subset only,
where the quantity is genuinely a recovery time, and 93 episodes is not enough to
establish these effects.

The strongest signal, `dev_at_90s` at rho −0.166 and p = 0.112, is *directionally*
sensible: episodes further off target at 90 s take longer to recover. It should be read as
a hypothesis worth testing on more data, not as a finding.

## Tuning that was tried and rejected

Recorded because a negative result is still a result.

**Residual model hyperparameter sweep.** A 27-point grid over `n_estimators`
{100, 200, 400} × `max_depth` {2, 3, 4} × `learning_rate` {0.03, 0.05, 0.1}, on the same
80/20 split. The best cell (200 / 2 / 0.10) appeared to beat the shipped configuration
(200 / 3 / 0.05) on all three metrics: accuracy 0.817 vs 0.767, precision 0.731 vs 0.667,
recall 0.826 vs 0.783.

Repeating both configurations across 15 different split seeds showed this was an artefact
of selecting on one test split. Averaged over 15 splits the "winner" is **worse**: mean
recall 0.765 vs 0.800, better on 2 splits, worse on 9, tied on 4. The shipped
configuration was kept unchanged.

**Recall-targeted threshold recalibration.** Replacing the accuracy-maximising cut point
with the lowest cut point whose *training* precision still cleared 0.6. Mean recall rose
from 0.800 to 0.945, but mean test precision fell to 0.503, and test precision landed
below the 0.6 floor in **15 of 15 splits** — the constraint held on train and did not
survive out of sample. Rejected.

Both experiments were run out-of-tree; `src/analysis/predictor.py` was never modified.
