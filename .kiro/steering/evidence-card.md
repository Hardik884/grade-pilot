---
inclusion: auto
---

# Evidence card

The provenance format every prediction and recommendation must carry. Satisfies the
deliverable "tag every suggestion with possible source of inference". No number reaches
the operator without one.

Activate this before building or editing the advisor, the evidence layer, the narrator,
or any dashboard component that displays a suggestion or a rationale.

## Structure

```json
{
  "card_id": "EC-0417",
  "episode_id": "EP-G12-G07-0031",
  "issued_t_sec": 96.0,
  "kind": "recommendation",

  "claim": {
    "statement": "Basis weight is predicted to breach the low limit in 52 s.",
    "predicted_value": 61.4,
    "unit": "g/m2",
    "horizon_sec": 52.0,
    "confidence": 0.81,
    "interval": [59.8, 63.1]
  },

  "action": {
    "variable": "stock_flow",
    "from": 2180.0,
    "to": 2295.0,
    "unit": "L/min",
    "ramp_rate_per_min": 115.0,
    "expected_max_dev_pct": 1.2,
    "expected_stabilisation_gain_sec": 96.0
  },

  "sources": [
    {"type": "physics",    "detail": "Mass balance: speed ramped 8.4% with stock flow held; predicted bw drop 7.9%.", "weight": 0.41},
    {"type": "causal",     "detail": "speed -> bw, lag 34 s, strength 0.71 (PCMCI, 287 episodes).", "weight": 0.28},
    {"type": "historical", "detail": "9 nearest transitions in grade space; 7 breached without early stock correction.", "episode_ids": ["EP-G12-G07-0004", "EP-G11-G07-0018"], "weight": 0.22},
    {"type": "recipe",     "detail": "Stock flow upper limit 3600 L/min; proposal uses 64% of headroom.", "weight": 0.09}
  ],

  "constraints_checked": {
    "recipe_limits": "pass",
    "actuator_rate": "pass",
    "downstream_quality": "moisture predicted 6.1%, within 4-9% band"
  },

  "counterfactual": {
    "no_action_max_dev_pct": 3.9,
    "no_action_broke_tonnes": 2.4,
    "with_action_broke_tonnes": 0.3
  },

  "narration": "Machine speed has ramped ahead of stock flow, so the sheet is running light..."
}
```

## Source taxonomy

Exactly these five `type` values. Every source needs a concrete, checkable `detail`.

| type | Means | `detail` must contain |
|---|---|---|
| `physics` | First-principles mass balance or delay term | The relation and the computed quantity |
| `causal` | Edge from the discovered causal graph | Cause, effect, lag in seconds, strength, sample size |
| `historical` | Retrieved similar past episodes | k, outcome split, and the actual episode IDs |
| `recipe` | Recipe limit or actuator constraint | The limit value and headroom used |
| `model` | Learned residual contribution | Feature and its attribution, e.g. SHAP value |

`weight` values are the normalised contribution to the claim and must sum to 1.0 ± 0.01.

## Narration rules

The narrator is a rephraser, not an author.

1. It receives **only** the card. No episode data, no free context.
2. Every number in the narration must appear verbatim in the card.
3. No causal language beyond edges present in `sources[type=causal]`.
4. Two to four sentences: what is happening, why, what to do, what happens if you don't.
5. Operator register — "running light", "ramped ahead of", "the sheet". Not "the model
   predicts a negative residual".
6. If a required field is missing, it emits a gap notice. It never fills the gap.

A validator runs before display: extract all numerals from `narration` and assert each is
present in the card. Fail closed — an unvalidated narration is not shown.

## Rejection capture

When an operator rejects a suggestion, record a reason code against the card:
`unsafe`, `already_handling`, `wrong_variable`, `too_aggressive`, `too_late`,
`disagree_with_cause`, `other`. These feed the trust-calibration view and are the training
signal for suggestion quality. A rejection is data, not a failure to be hidden.
