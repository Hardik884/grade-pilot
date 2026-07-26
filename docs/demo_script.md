# Demo script

A five-minute walkthrough of the dashboard. Every screen named here is captured in
[screenshots/](screenshots/), so this reads equally well as a script for a live demo or
as a caption track for the images if no demo is possible.

**Setup** — the dashboard replays a *recorded* historical transition. It is not a live
feed, and it is labelled as such in the header. Nothing auto-applies; the operator is the
actuator.

```bash
python -m src.sim.generate --n 300 --out data/episodes --seed 42
python -m src.analysis.impact
python -m src.ui.server        # http://127.0.0.1:8000
```

---

## 1. The problem, on one screen (30 s)

**Show:** `01-header-timeline-at-risk.png`

Open on episode `EP-G10-G01-0001` at t = 90 s. The banner reads *"Basis weight is on
track to breach the low limit in 35 s."*

Point at the timeline: the measured (scanner) trace is still comfortably inside the ±2.5%
band. Nothing on it looks alarming yet. **That is the whole point** — the call is being
made before the measurement supports it.

## 2. Why the call is possible (60 s)

**Show:** the staleness panel lower on the same screenshot.

*"You are seeing paper made 56 seconds ago (transport 13 s + scanner hold 43 s)."*

Four figures:

| Transport delay | Scanner traverse and hold | Effective measurement lag | Headbox lead over the reading |
|---|---|---|---|
| 13.2 s | 43.0 s | 56.2 s | −5.60 g/m² |

Headbox basis weight is already **121.56 g/m²** against a scanner reading of
**127.15 g/m²**. The sheet is light; the screen has not caught up.

The line to land: *the mass balance is valid open-loop for 56 s, because until that sheet
reaches the scanner the loop has no information to react to. That window is what makes an
early call possible.* Across the dataset the composed lag has a median of **41.4 s**
(p10 30.5 s, p90 51.1 s) — transport alone would understate staleness by roughly 5×,
which is why the UI never quotes `theta_sec` on its own.

## 3. The recommendation, and its evidence (90 s)

**Show:** `02-suggestion-evidence-card.png`

The proposed move: **machine speed 680 → 644 m/min over the next 60 s** (−36 m/min/min).

Read the counterfactual straight off the card: peak deviation **1.83% with the move**
against **3.20% if left alone**, and a stabilisation gain of 260 s. Confidence 0.96.

Then scroll to the evidence card — **five sources, weights summing to 1.00**:

| Source | Weight | What it contributes |
|---|---|---|
| Physics | 0.395 | Mass balance over the valid window |
| Model | 0.241 | Residual moved the forecast by 2.633% |
| Causal | 0.168 | `speed → bw` at lag 40 s, strength 0.8094 |
| Recipe | 0.104 | Speed lower limit 420 m/min; proposal uses 3% of range |
| Historical | 0.092 | 9 nearest transitions in grade-property space |

Two things to say explicitly, because they are the differentiators:

- **The narration is not free text.** It is rephrased from the card only, and every
  numeral in it is machine-validated to appear in the card. The UI says so under the
  paragraph: *"Validated: every numeral above appears in the card."*
- **The constraint filter ran before scoring.** Expand it. A move violating recipe limits
  or actuator rates was never a scored candidate in the first place.

Expand **Retrieved episodes**: k = 9, split 4 held spec / 5 went off, each listed by
episode ID so the claim is checkable rather than asserted.

## 4. Why lag is the whole trick (60 s)

**Show:** `03-impact-ranking-lagged.png`, then toggle to `04-impact-ranking-zero-lag.png`

Lagged view — five loops cleanly separated, each bar labelled with the lag that produced
its peak: speed 40 s, filler 40 s, steam 80 s, thick stock 80 s, consistency 35 s.
Recovered from measured data alone; the ranking code never sees `injected_faults` or
`bw_true`.

Now flip the toggle. Every bar reads **"lag 0 s"**. Be precise about what changes: the
*ordering* survives, but the strengths compress and **the timing disappears entirely**.
Nothing on the zero-lag view tells an operator which loop acts first or how far ahead of
the display it moves — and that timing is the actionable part.

## 5. Closing the loop (45 s)

**Show:** `05-history-trust-economics.png`

Accept or reject the card. A rejection requires a **reason code** — the screenshot shows
one card rejected as *"Too aggressive"* and one accepted, each logged with its source mix.

Broke avoided: **1.12 t** on this transition, against 2.61 t realised with no advice — the
move takes peak deviation from 3.20% to 1.83%, a 43% shallower excursion, and tonnage is
scaled by that reduction.

Be honest about the trust chart: it is plotted over **n = 2 logged decisions**. It
demonstrates the mechanism, not a result. Say so before anyone asks.

## 6. The headline, if you have 30 s left

**Show:** `07-full-page.png`

Everything on one operator screen. Then the number that matters:

> From the first 90 seconds of a transition, the naive current-deviation rule catches
> **3 of 23** real breaches. This system catches **18 of 23** — recall 0.130 → 0.783 on
> 60 held-out episodes, at 0.783 accuracy and 0.692 precision.
>
> Physics alone, with zero learned parameters, already triples the naive recall
> (0.130 → 0.304). The residual more than doubles it again.

---

## Questions you should expect, and the honest answers

**"Is this real mill data?"** No. None was provided. Everything comes from a physics
simulator whose episodes pass a six-rule plausibility gate. The strongest honest claim is
that the method recovers structure it was not told about, in data whose generating process
is independently constrained. Recall of 0.783 should not be expected to transfer unchanged.

**"Why is precision only 0.692?"** Deliberately. A false negative is broke on the reel; a
false positive is an advisory the operator dismisses. The naive rule has *better*
precision (0.750) and is useless because it finds 3 of 23. A recall-targeted threshold was
tested and rejected — it drove test precision to 0.503, below the 0.6 floor, in 15 of 15
random splits.

**"Did you tune this until it looked good?"** Three attempts, all rejected: a 27-point
hyperparameter sweep, a recall-targeted threshold, and an alarm-history feature set. Each
looked like a win on one split; 15-split repeats showed all three were selection noise.
None was claimed as a gain. They are written up in [RESULTS.md](RESULTS.md).

**"How do you know the constraint filter works?"** By an adversarial test that tightens
limits until candidates become infeasible and confirms none survives to scoring. Across
the 300-episode run, 268 candidates were generated and **0** were discarded, because the
grade catalogue leaves 15% headroom on every actuator. The filter has never faced a
naturally binding limit — that is a limitation, and it is recorded as one.

**"Could it drive the machine?"** Not as built. It is advisory by design, with no path to
the controller. The failure modes of a closed-loop version are not the ones analysed here.
