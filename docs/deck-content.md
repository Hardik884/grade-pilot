# Submission deck — approved slide copy

**Written to be graded from the PDF alone** — no demo, no code review, no verbal
explanation. Every claim below is self-contained and numeric, and every number traces to
a file in this repo.

Sources: [RESULTS.md](RESULTS.md) (accuracy, lags, p-values), `data/impact_ranking.json`
(discovered lags), and an aggregate run of the advisor over all 300 episodes through the
dashboard's own code path (peak-deviation and evidence-card figures). Dataset is 300
episodes at seed 42, 240 train / 60 test.

Template: `docs/IDEA_Presentation_Format.pptx` (SIH format). Its slide 1 is an
instructions page the template itself says to delete; removing it leaves exactly six
slides, matching the stated maximum. The five template section headings are kept verbatim.

---

## Slide 1 — TITLE PAGE

Placeholders, left for the team to complete:

- Problem Statement ID — `<to be completed>`
- Problem Statement Title — `<to be completed>`
- Theme — `<to be completed>`
- PS Category — `<to be completed>`
- Student Name (registered on portal) — `<to be completed>`
- Student ID — `<to be completed>`

Stated on the slide:

**GRADE CHANGE INTELLIGENCE**
Catching basis-weight deviations before they exceed limits
*Advisory layer over the MD multivariable MPC — it never writes a setpoint; the operator is the actuator.*

---

## Slide 2 — IDEA TITLE

**The measurement is late. We compute the sheet that already exists.**

- The QCS reading is a median **41.4 s old** — 7.2 s transport + 33.5 s scanner traverse and hold
- A rule watching the displayed deviation is reacting to sheet that no longer exists
- GCI runs the mass balance on **current setpoints**, reading the sheet already at the headbox
- Inside that lag the control loop has no new information, so the breach is already committed — arithmetic, not a trend fit

### Evaluated against the five criteria

**Prediction accuracy**
- Catches deviations before they exceed limits in **18 of 23** real breaches, from the first 90 s of a transition; the naive current-deviation rule catches **3 of 23**
- Recall 0.130 → 0.783 on 60 held-out episodes; accuracy 0.783, precision 0.692

**Process optimization**
- Across **67 issued recommendations**, forecast peak deviation falls from a median **4.13% to 1.85%** — a 57.4% shallower excursion, shallower in **67 of 67**
- All **67 of 67** forecast excursions are brought back inside the ±2.5% spec band; model-estimated stabilisation gain median **66 s** (directional — see limitations)

**Explainability**
- **All 67 cards carry exactly 5 weighted sources summing to 1.000** — physics, model, causal, recipe, historical (e.g. 0.395 / 0.241 / 0.168 / 0.104 / 0.092)
- Narration is rephrased from the card only and machine-validated: every numeral it prints must appear in the card, enforced in the card path and covered by tests

**Usability**
- Working interactive dashboard (Flask + static frontend), screenshots on slides 2 and 5
- Every card has Accept / Reject with a structured reason code (e.g. "Too aggressive"), feeding a trust-calibration view and an avoided-broke figure in tonnes

**Historical-data use**
- Case-based retrieval over past transitions in grade-property space: **k = 9** neighbours returned split by outcome (4 held spec, 5 went off)
- Lagged causal discovery over **300 episodes** recovered the correct variable ordering and timing without being told it: speed 40 s, filler 40 s, steam 80 s, thick stock 80 s, consistency 35 s

Visual: `docs/screenshots/01-header-timeline-at-risk.png` — verified to show the breach
call, the transition timeline, and the staleness panel (headbox 121.56 g/m² against a
scanner reading of 127.15 g/m²).

---

## Slide 3 — TECHNICAL APPROACH

**Gray-box: the mass balance leads, the model only trims the residual**

*Pipeline*
- simulator → episodes → 90 s feature window → physics projection → learned residual → constrained advisor → evidence card → dashboard

*Method*
- Mass balance is not fitted — it is the relation the mill obeys
- Gradient-boosted residual corrects only what 90 s of data cannot identify (wet-end time constant, moisture coupling, controller behaviour through the settle)
- Contribution split **67.4% physics / 32.6% model**, asserted on every run — the report fails loudly if the learned term ever exceeds 40%
- Constraint filter runs **before** scoring, so a setpoint violating recipe limits or actuator rates is never a scored candidate

*Stack*
- Python 3.11 · NumPy / pandas / SciPy · scikit-learn · Flask + static frontend
- 300-episode physics simulator · **90 tests** · six-rule physical plausibility gate on all generated data

**Lagged discovery, not a correlation heatmap** — speed 40 s (0.809) · filler 40 s (0.662)
· steam 80 s (0.581) · thick stock 80 s (0.487) · consistency 35 s (0.210). Raw level
correlations sit at **0.351–0.967**, so every loop looks connected and the ordering is
uninformative; the lag sweep is what separates them.

Visual: cropped `docs/screenshots/03-impact-ranking-lagged.png` — verified to show the
five ranked bars, each labelled with the lag that produced it, and the note that raw
correlations span 0.35–0.97.

---

## Slide 4 — FEASIBILITY AND VIABILITY

**Measured from 90 s of each transition — 60 held-out episodes, 23 real breaches**

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Naive (current deviation > 1.5%) | 0.650 | 0.750 | 0.130 |
| Physics only | 0.700 | 0.778 | 0.304 |
| **Gray-box (shipped)** | **0.783** | **0.692** | **0.783** |

- Physics alone **triples** naive recall with zero learned parameters
- Max-deviation MAE improves 1.163% → 0.910%

*Why recall is the metric*
- A false negative is broke on the reel; a false positive is an advisory the operator dismisses
- The naive rule has **better** precision (0.750) and is useless — it finds 3 of 23

*Risks, stated*
- Validated on synthetic physics-based data — no real mill data was provided
- 60-episode test set: one episode moves recall by 0.043
- Stabilisation ranking is directional only — best p = 0.112 (rho −0.166, n = 93); nothing significant at 0.05
- Constraint filter: **0 of 268** generated candidates were ever rejected by it, so it is proven by an adversarial test, not by naturally binding data

*Method we can show*
- Six-rule physical plausibility gate on every episode before use
- Decision thresholds calibrated on train only, never on test
- Three tuning attempts — a 27-point hyperparameter sweep, a recall-targeted threshold, and an alarm-history feature set — each looked like a win on one split; 15-split repeats showed all three were selection noise, and **none was claimed as a gain**

---

## Slide 5 — ARTIFACTS

**Every number carries an evidence card — no bare figures reach the operator**

Left: `docs/screenshots/02-suggestion-evidence-card.png` — verified to show all of the
following on one screen:
- Call: *"Basis weight is on track to breach the low limit in 35 s"*
- Constrained move: machine speed 680 → 644 m/min over 60 s (−36 m/min/min)
- Peak deviation **1.83% with the move vs 3.20% if left alone**; stabilisation gain 260 s
- Evidence card, **5 sources, weights sum to 1.00**: physics 0.395, model 0.241, causal 0.168, recipe 0.104, historical 0.092
- Retrieved episodes **k = 9**, split 4 held spec / 5 went off, listed by episode ID
- Accept / Reject controls with a reason-code selector

Right: `docs/screenshots/05-history-trust-economics.png` — verified to show:
- Suggestion history with per-card source mix and decisions (one Accepted, one Rejected — reason "Too aggressive")
- Broke avoided **1.12 t** on this transition, against 2.61 t realised with no advice — a 43% shallower excursion
- Trust calibration over decisions logged this session (**n = 2** — the mechanism, not yet evidence)

*Repository* — `github.com/Hardik884/grade-pilot`; seed 42 regenerates every episode
byte-identically.

---

## Slide 6 — RESEARCH AND REFERENCES

*Process physics*
- Basis-weight mass balance with first-pass retention
- Speed-dependent transport delay, `theta = 60 · L / speed`
- QCS traversing scanner: scan-average, zero-order hold, 20–45 s period
- Wet-end first-order mixing lag, 20–60 s

*Method*
- Gray-box modelling: known physics plus a learned residual
- Lagged association sweep, 0–120 s in 5 s steps, over 300 episodes
- Spearman rank correlation with reported p-values
- Baseline discipline: naive and physics-only reported alongside every result

*Project documentation*
- `README.md` — five-criteria map, headline result, architecture, quickstart
- `docs/RESULTS.md` — full tables, discovered lags, p-values, rejected tuning
- `docs/LIMITATIONS.md` — what this does and does not establish
- `docs/decisions.md` — dated structural decisions

*Reproduce every number in this deck*
```
python -m src.sim.generate --n 300 --seed 42
python -m tests.physics_check data/episodes
python -m src.analysis.impact
python -m src.analysis.predictor
```
