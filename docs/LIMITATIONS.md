# Limitations

What this system does not establish. Stated plainly, because a judge or a process
engineer will find these anyway and it is better that they are already written down.

## The data is synthetic

No real mill data was provided for this work. Every number in
[RESULTS.md](RESULTS.md) comes from 300 episodes produced by our own physics simulator.

The simulator is built from the documented process relations — mass balance,
speed-dependent transport delay, first-order wet-end lag, scanner traverse averaging —
and every episode passes a six-rule plausibility gate before use. But a simulator
validated against the same physics the predictor assumes is a closed loop. **The strongest
honest claim is that the method recovers structure it was not told about, in data whose
generating process is independently constrained.** It is not evidence of accuracy on a
real machine, and the reported recall of 0.783 should not be expected to transfer
unchanged.

The most likely source of transfer loss is that real mills carry disturbances this
simulator does not model: broke ratio swings, refiner variation, wire and felt condition,
steam header pressure interactions, and headbox consistency dynamics beyond a
slow-varying drift.

## The stabilisation ranking is directional, not significant

No feature reaches p < 0.05. The strongest is `dev_at_90s` at rho −0.166, p = 0.112, on
93 episodes. It is reported because the direction is physically sensible, not because it
is established. It should not be presented as a finding, and the dashboard labels it as
directional. See the full table in [RESULTS.md](RESULTS.md).

The underlying problem is that stabilisation time is degenerate in this dataset: 80% of
episodes are already in-band when the setpoint ramp completes. Fixing this properly needs
either a harder-to-control mill configuration or considerably more off-spec episodes, not
a different statistic.

## The constraint filter has not been exercised by real data

The filter that removes recipe-limit and actuator-rate violations before scoring is the
system's principal safety property, and in this dataset **it almost never has anything to
reject**. The generated episodes operate comfortably inside their limits by construction,
because the grade catalogue is built to leave at least 15% headroom on every actuator.

It is verified only by an adversarial test that artificially tightens the limits until
candidates become infeasible, and confirms none survive to scoring. That test proves the
filter is wired in and ordered correctly. It does **not** demonstrate the filter behaving
under realistic pressure, where limits bind occasionally and the interesting question is
whether the remaining candidate set is still useful rather than merely safe.

## Advisory only

The system never writes a setpoint. There is no path from GCI to the controller — it
produces a recommendation, an operator accepts or rejects it, and any actual move is made
by a human through the existing MPC. Nothing here has been designed, tested or reviewed
for closed-loop use, and the failure modes of an automated version are not the same as
the failure modes analysed here.

## Further limitations worth knowing

**The 90-second decision window is a fixed choice, not an optimised one.** It was picked
because it is early enough to be actionable and long enough to measure ramp rates. No
sweep over window length was run, so it is not known whether 60 s or 120 s would be
better, nor how sharply performance degrades either side.

**The wet-end time constant is a fixed nominal.** The simulator draws it per episode from
20–60 s; the physics projection assumes 40 s because 90 s of data is not enough to
identify it. The residual model absorbs the difference, which is a legitimate use of a
learned term but means the physics path is knowingly approximate on any given episode.

**Test-set size limits the precision of every accuracy figure.** 60 held-out episodes
containing 23 real breaches means one episode moves recall by about 0.043. Differences
smaller than roughly 0.1 in these tables should not be treated as meaningful — a
hyperparameter sweep that appeared to improve all three metrics turned out to be split
noise when repeated across 15 splits.

**Results are reported on a single stratified 80/20 split.** Cross-validated figures are
not reported, though the 15-split repeats described in RESULTS.md give some indication of
variance: recall standard deviation across splits is 0.065.

**The physics projection is only trusted inside the measurement lag.** Beyond roughly
41 s ahead, the mill is closed-loop and the controller acts, so open-loop mass balance
stops being meaningful. An early version projected ten minutes ahead and predicted 26%
mean deviation against an actual 2.8%. The horizon is now bounded, which is correct, but
it means the system cannot forecast breaches that develop later than about a minute out.

**Grade coverage is uneven by design.** Transition pairs are sampled Zipf-distributed, so
74 of 132 possible pairs appear at all and some are represented by a single episode. This
is deliberate — it exercises grade-space retrieval on sparse pairs — but performance on
rarely-seen transitions is correspondingly less well characterised.

**Avoided-broke tonnage is an estimate, not a measured outcome.** Broke tonnes for a
transition that actually happened is computed exactly, from basis weight, speed and trim
over off-spec samples. But "tonnes avoided if you accept this suggestion" is derived
severity-proportionally from the evidence card's own counterfactual — the alternative
history was never run. It is a consistent way to size a recommendation, not a
measurement, and it inherits any error in the underlying breach prediction.
