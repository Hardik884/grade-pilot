---
name: dashboard-language
description: Visual and interaction conventions for the GCI operator dashboard - layout, colour semantics, chart rules, and required panels. Use whenever building or editing anything under src/ui/, or when producing dashboard screenshots for the submission deck.
paths: src/ui/**
---

# Dashboard conventions

Audience is a control-room operator, not a data scientist. Optimise for glanceability
under time pressure.

## Required panels

1. **Transition timeline** — basis weight vs time, spec band shaded, predicted trajectory
   as a dashed forward extension with an uncertainty cone. Vertical marker at "now".
2. **Risk header** — single sentence status plus time-to-breach countdown when at risk.
3. **Impact ranking** — loops and parameters ordered by contribution to breach risk and
   to stabilisation time, with the discovered lag shown next to each.
4. **Suggestion panel** — proposed setpoint, the evidence card rendered as grouped source
   chips, Accept and Reject buttons, reason-code selector on reject.
5. **Suggestion history** — every card issued, its source mix, accepted or rejected, and
   the realised outcome.
6. **Trust calibration** — acceptance rate and realised-accuracy over time.
7. **Economic tile** — estimated broke tonnes avoided this transition and cumulative.

## Colour semantics

Fixed meanings. Never reuse these hues decoratively.

| Meaning | Use |
|---|---|
| In spec / accepted | Green |
| At risk / predicted breach | Amber |
| Off spec / rejected | Red |
| Prediction and uncertainty | Blue, prediction always dashed |
| Measured history | Neutral dark grey, solid |

Spec band is a light fill, never a pair of hard lines that compete with the data.

## Chart rules

- Time axis in seconds from transition start, zero always visible and labelled.
- Never truncate the y-axis on basis weight — the 2.5% band must be readable in context.
- Uncertainty as a filled cone, not error bars.
- Annotate the breach point directly on the chart; do not make the operator read a legend.
- One idea per chart. Split rather than dual-axis.

## Interaction rules

- Accept and Reject are always visible together, equally weighted. Never make Accept the
  visually dominant path — that biases the feedback data you are collecting.
- Reject requires a reason code before it commits.
- Every action writes to the feedback log immediately and shows a confirmation.
- Nothing auto-applies. The system is advisory; the operator is the actuator.

## Screenshot discipline

Deck screenshots must show a live at-risk state with an evidence card expanded, not an
idle steady state. Capture at a fixed 1600x1000 viewport for consistency.
