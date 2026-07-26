"""M1 Synthetic Mill: physics, schema conformance and determinism.

The gate in ``tests.physics_check`` audits generated *data*. These tests pin the
*mechanisms* — the ones a statistical audit of finished episodes cannot resolve,
notably the speed-dependence of the transport delay, which is invisible in measured
basis weight because the scanner only refreshes every 20-45 s.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.sim.episode import PHASES, SERIES_COLUMNS, SimConfig, episode_id, simulate_episode
from src.sim.generate import generate
from src.sim.grades import (
    MARGIN_FRACTION,
    RECIPE_LIMITS,
    actuator_margins,
    build_catalogue,
    catalogue_infeasibilities,
    load_grades,
    operating_point,
    write_grades_json,
)
from src.sim.labels import STABILISATION_WINDOW_SEC, compute_labels
from src.sim.machine import (
    MachineConfig,
    Scanner,
    VariableTransportDelay,
    basis_weight_g_m2,
    required_flows,
)
from src.sim.units import RANGES, m3_h_to_L_min
from src.sim.writer import read_episode

MACHINE = MachineConfig()


# ---------------------------------------------------------------------------- mass balance


def test_mass_balance_reproduces_schema_example() -> None:
    """The arithmetic check the whole unit correction rests on.

    trim 6.4 m, 1000 m/min, 946 m3/h of 3.5% stock, 404 L/min of 30% filler,
    retention 0.78 -> 82 g/m2. If this drifts, the m3/h conversion has been lost
    and every downstream model is training on a machine that cannot exist.
    """
    bw = basis_weight_g_m2(
        946.0, 3.5, 404.0, 1000.0, trim_m=6.4, retention=0.78, filler_cons_pct=30.0
    )
    assert float(bw) == pytest.approx(82.0, abs=0.05)


def test_stock_flow_conversion_is_applied() -> None:
    """m3/h -> L/min is a factor of 1000/60; skipping it costs ~17x on basis weight."""
    assert float(np.asarray(m3_h_to_L_min(60.0))) == pytest.approx(1000.0)


def test_basis_weight_is_inverse_in_speed() -> None:
    """The dominant grade-change dynamic: doubling speed halves basis weight."""
    kw = dict(trim_m=6.4, retention=0.78, filler_cons_pct=30.0)
    slow = float(basis_weight_g_m2(946.0, 3.5, 404.0, 500.0, **kw))
    fast = float(basis_weight_g_m2(946.0, 3.5, 404.0, 1000.0, **kw))
    assert slow == pytest.approx(2.0 * fast, rel=1e-9)


def test_required_flows_inverts_the_mass_balance() -> None:
    stock, filler = required_flows(
        82.0, 18.0, 1000.0, 3.5, trim_m=6.4, retention=0.78, filler_cons_pct=30.0
    )
    bw = basis_weight_g_m2(
        stock, 3.5, filler, 1000.0, trim_m=6.4, retention=0.78, filler_cons_pct=30.0
    )
    assert float(bw) == pytest.approx(82.0, rel=1e-9)


# ---------------------------------------------------------------------------- grade catalogue


def test_every_grade_has_a_feasible_operating_point() -> None:
    """The guard against the envelope bug that blocked M1 at planning time.

    The mass balance and the actuator limits were originally not jointly
    satisfiable. This asserts every catalogue grade sits at least
    ``MARGIN_FRACTION`` clear of every recipe limit, so the controller has room to
    move in both directions rather than starting pinned against a bound.
    """
    problems = catalogue_infeasibilities(build_catalogue(), MACHINE)
    assert problems == [], "infeasible grades:\n  " + "\n  ".join(problems)


def test_grade_properties_are_in_physical_range() -> None:
    for code, g in build_catalogue().items():
        for field in ("bw", "ash", "moist", "caliper"):
            rng = RANGES[field]
            value = getattr(g, field)
            assert rng.lo <= value <= rng.hi, f"{code}.{field}={value} outside {rng}"


def test_heavy_grades_run_slower_than_light_ones() -> None:
    """Speed is a grade property precisely because bw ~ 1/speed; if this inverts,
    the catalogue no longer fits inside the actuator envelope."""
    cat = build_catalogue()
    bws = np.array([g.bw for g in cat.values()])
    speeds = np.array([g.nominal_speed_m_min for g in cat.values()])
    corr = float(np.corrcoef(bws, speeds)[0, 1])
    assert corr < -0.8, f"basis weight and nominal speed should be strongly inverse, got r={corr}"


def test_operating_points_stay_inside_recipe_limits() -> None:
    for code, g in build_catalogue().items():
        op = operating_point(g, MACHINE)
        for name, margin in actuator_margins(op).items():
            assert margin >= MARGIN_FRACTION, f"{code}: {name} margin {margin:.3f}"


def test_grades_json_round_trips(tmp_path) -> None:
    path = write_grades_json(tmp_path / "grades.json")
    loaded = load_grades(path)
    assert loaded == build_catalogue()


# ---------------------------------------------------------------------------- transport delay


def test_transport_delay_tracks_speed() -> None:
    """Halving speed doubles the dead time.

    This is the single most important dynamic in the project and it cannot be
    tested from generated episodes: the scanner's 20-45 s zero-order hold is
    coarser than the delay itself. So it is pinned here, directly on the mechanism.
    """
    dt, distance = 1.0, 140.0
    for speed in (500.0, 1000.0):
        d = VariableTransportDelay(
            distance,
            capacity=1200,
            n_signals=1,
            t0_sec=0.0,
            dt_sec=dt,
            initial_speed_m_min=speed,
            initial_values=[0.0],
        )
        for i in range(1000):
            d.step(i * dt, speed, dt, [float(i)])
        assert d.theta_sec == pytest.approx(60.0 * distance / speed, rel=0.02)


def test_transport_delay_responds_to_a_speed_change_mid_transit() -> None:
    """A fixed-length buffer would hold theta constant here; the travel-coordinate
    inversion must not."""
    dt, distance, n = 1.0, 140.0, 2000
    d = VariableTransportDelay(
        distance,
        capacity=n + 10,
        n_signals=1,
        t0_sec=0.0,
        dt_sec=dt,
        initial_speed_m_min=1200.0,
        initial_values=[0.0],
    )
    for i in range(600):
        d.step(i * dt, 1200.0, dt, [0.0])
    fast_theta = d.theta_sec
    for i in range(600, n):
        d.step(i * dt, 600.0, dt, [0.0])
    slow_theta = d.theta_sec

    assert fast_theta == pytest.approx(7.0, rel=0.05)
    assert slow_theta == pytest.approx(14.0, rel=0.05)
    assert slow_theta > fast_theta * 1.8


def test_transport_delay_reproduces_the_input_sequence() -> None:
    """What comes out is what went in, shifted - not smeared or resampled."""
    dt, distance, speed = 1.0, 140.0, 1000.0
    d = VariableTransportDelay(
        distance,
        capacity=900,
        n_signals=1,
        t0_sec=0.0,
        dt_sec=dt,
        initial_speed_m_min=speed,
        initial_values=[0.0],
    )
    src = np.sin(np.arange(800) * 0.05)
    out = np.array([d.step(i * dt, speed, dt, [src[i]])[0] for i in range(800)])

    # theta here is 8.4 s, not a whole number of samples, and the delay interpolates
    # rather than snapping to a sample boundary - so compare against the source
    # resampled at the same fractional shift.
    theta = 60.0 * distance / speed
    idx = np.arange(400, 600, dtype=float)
    expected = np.interp(idx - theta, np.arange(800, dtype=float), src)
    np.testing.assert_allclose(out[400:600], expected, atol=2e-3)


# ---------------------------------------------------------------------------- scanner


def test_scanner_holds_between_traverses() -> None:
    """A held reading is one measurement repeated, not a fresh one each second."""
    rng = np.random.default_rng(0)
    s = Scanner(30.0, [0.5], [0.1], rng, dt_sec=1.0, initial_values=[80.0])
    values = [s.step(1.0, [80.0])[0] for _ in range(90)]
    distinct = len(set(np.round(values, 9)))
    assert distinct <= 4, f"expected ~3 refreshes over 90 s at a 30 s scan, saw {distinct}"


# ---------------------------------------------------------------------------- episode schema


@pytest.fixture(scope="module")
def episode() -> tuple[pd.DataFrame, dict]:
    cat = build_catalogue()
    return simulate_episode(cat["G12"], cat["G07"], seed=42, seq=1, machine=MACHINE)


def test_series_columns_match_the_contract_exactly(episode) -> None:
    df, _ = episode
    assert tuple(df.columns) == SERIES_COLUMNS


def test_index_is_t_sec_float_with_required_pre_roll(episode) -> None:
    df, _ = episode
    assert df.index.name == "t_sec"
    assert df.index.dtype == np.float64
    assert df.index.min() <= -120.0, "schema requires at least 120 s of pre-transition baseline"
    assert 0.0 in set(df.index.to_numpy()), "t_sec = 0 is the transition trigger"


def test_numeric_columns_are_float64(episode) -> None:
    df, _ = episode
    numeric = [c for c in SERIES_COLUMNS if c not in ("phase", "op_action", "alarm")]
    for col in numeric:
        assert df[col].dtype == np.float64, f"{col} is {df[col].dtype}"


def test_phase_is_categorical_over_the_declared_values(episode) -> None:
    df, _ = episode
    assert isinstance(df["phase"].dtype, pd.CategoricalDtype)
    assert set(df["phase"].cat.categories) <= set(PHASES)
    assert (df.loc[df.index < 0.0, "phase"] == "pre").all()


def test_meta_carries_every_required_field(episode) -> None:
    _, meta = episode
    required = {
        "episode_id",
        "grade_from",
        "grade_to",
        "grade_from_props",
        "grade_to_props",
        "machine",
        "recipe_limits",
        "actuator_rates",
        "injected_faults",
        "seed",
        "labels",
    }
    assert required <= set(meta)
    assert set(meta["machine"]) == {
        "trim_m",
        "scanner_distance_m",
        "retention",
        "filler_cons_pct",
    }
    assert set(meta["labels"]) == {
        "off_spec",
        "max_dev_pct",
        "breach_t_sec",
        "stabilisation_t_sec",
        "broke_tonnes",
    }
    assert set(meta["recipe_limits"]) == set(RECIPE_LIMITS)


def test_episode_id_format() -> None:
    assert episode_id("G12", "G07", 1) == "EP-G12-G07-0001"


def test_theta_is_positive_and_speed_consistent(episode) -> None:
    df, meta = episode
    expected = 60.0 * meta["machine"]["scanner_distance_m"] / df["speed"].to_numpy(dtype=float)
    assert (df["theta_sec"] > 0).all()
    # Instantaneous theta trails the steady-speed value while speed is moving, so
    # compare where speed is settled.
    settled = df["phase"] == "steady"
    np.testing.assert_allclose(
        df.loc[settled, "theta_sec"].to_numpy(), expected[settled.to_numpy()], rtol=0.05
    )


# ---------------------------------------------------------------------------- labels


def test_labels_match_the_schema_definitions(episode) -> None:
    df, meta = episode
    recomputed = compute_labels(df, trim_m=meta["machine"]["trim_m"])
    assert recomputed == meta["labels"]


def test_off_spec_uses_the_2_5_percent_band() -> None:
    t = np.arange(-120.0, 400.0)
    df = pd.DataFrame(
        {"bw": np.full(t.size, 100.0), "bw_sp": np.full(t.size, 100.0), "speed": 1000.0},
        index=pd.Index(t, name="t_sec"),
    )
    assert compute_labels(df, trim_m=6.4)["off_spec"] is False

    df.loc[df.index[200], "bw"] = 100.0 * 1.026  # just outside the band
    labels = compute_labels(df, trim_m=6.4)
    assert labels["off_spec"] is True
    assert labels["breach_t_sec"] == pytest.approx(float(t[200]))


def test_stabilisation_requires_a_continuous_in_band_window() -> None:
    """A breach too near the end leaves no room for the 120 s window to close."""
    t = np.arange(0.0, 300.0)
    df = pd.DataFrame(
        {"bw": np.full(t.size, 100.0), "bw_sp": np.full(t.size, 100.0), "speed": 1000.0},
        index=pd.Index(t, name="t_sec"),
    )
    df.loc[df.index[250], "bw"] = 110.0
    assert compute_labels(df, trim_m=6.4)["stabilisation_t_sec"] is None

    df.loc[df.index[250], "bw"] = 100.0
    df.loc[df.index[100], "bw"] = 110.0
    stab = compute_labels(df, trim_m=6.4)["stabilisation_t_sec"]
    assert stab == pytest.approx(101.0)
    assert float(t[-1]) - stab >= STABILISATION_WINDOW_SEC


# ---------------------------------------------------------------------------- determinism


def test_same_seed_gives_identical_episodes() -> None:
    cat = build_catalogue()
    a, ma = simulate_episode(cat["G12"], cat["G07"], seed=7, seq=3, machine=MACHINE)
    b, mb = simulate_episode(cat["G12"], cat["G07"], seed=7, seq=3, machine=MACHINE)
    pd.testing.assert_frame_equal(a, b)
    assert ma == mb


def test_different_seeds_give_different_episodes() -> None:
    cat = build_catalogue()
    a, _ = simulate_episode(cat["G12"], cat["G07"], seed=7, seq=1, machine=MACHINE)
    b, _ = simulate_episode(cat["G12"], cat["G07"], seed=8, seq=1, machine=MACHINE)
    assert not a["bw"].equals(b["bw"])


# ---------------------------------------------------------------------------- ground truth


def test_bw_true_leads_the_measurement(episode) -> None:
    """bw_true is the headbox value: post wet-end lag, pre delay, pre noise. It must
    lead the measurement, never trail it."""
    df, _ = episode
    bw = df["bw"].to_numpy(dtype=float)
    true = df["bw_true"].to_numpy(dtype=float)
    best = min(
        range(0, 90), key=lambda k: float(np.sqrt(np.mean((bw[k:] - true[: bw.size - k]) ** 2)))
    )
    assert best > 0, "measured bw should lag bw_true by the transport delay and traverse"


# ---------------------------------------------------------------------------- generation


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> tuple[dict, object]:
    out = tmp_path_factory.mktemp("episodes")
    summary = generate(60, out, seed=42)
    return summary, out


def test_off_spec_rate_is_in_the_calibrated_band(dataset) -> None:
    """25-45%. Below it the advisor has nothing to advise on; above it the mill is
    simply broken and the baseline stops being meaningful."""
    summary, _ = dataset
    rate = summary["off_spec_rate"]
    assert 0.25 <= rate <= 0.45, f"off-spec rate {rate:.1%} outside 25-45%"


def test_all_five_fault_modes_appear(dataset) -> None:
    from src.sim.faults import FAULT_NAMES

    summary, _ = dataset
    assert set(summary["fault_counts"]) == set(FAULT_NAMES)
    assert all(count > 0 for count in summary["fault_counts"].values())


def test_transition_pairs_are_not_uniform(dataset) -> None:
    """Zipf sampling: some pairs must stay sparse so M6's grade-space retrieval is
    exercised on transitions it has barely seen."""
    _, out = dataset
    index = pd.read_parquet(out / "index.parquet")
    counts = index.groupby(["grade_from", "grade_to"]).size()
    assert counts.max() >= 3 * counts.min(), "pair sampling looks uniform"


def test_index_matches_the_episodes_on_disk(dataset) -> None:
    _, out = dataset
    index = pd.read_parquet(out / "index.parquet")
    dirs = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert sorted(index["episode_id"]) == dirs


def test_episodes_round_trip_through_parquet(dataset) -> None:
    _, out = dataset
    ep = sorted(p for p in out.iterdir() if p.is_dir())[0]
    df, meta = read_episode(ep)
    assert tuple(df.columns) == SERIES_COLUMNS
    assert meta["episode_id"] == ep.name
    assert json.loads((ep / "meta.json").read_text(encoding="utf-8")) == meta


def test_correlation_alone_cannot_recover_the_causal_structure(dataset) -> None:
    """The premise M3 exists to solve, pinned as a property of the data.

    Every manipulated variable is driven by the same coordinated setpoint ramp, so
    at lag 0 they all correlate with basis weight at once and a correlation matrix
    cannot say which one caused it. The structure is only separable by lag. If this
    test ever fails because the lag-0 correlations have become distinguishable, the
    simulator has stopped posing the problem the causal module is built for.
    """
    from tests.physics_check import mass_balance_lag_sec

    _, out = dataset
    episodes = sorted(p for p in out.iterdir() if p.is_dir())[:20]

    ambiguous = 0
    for ep in episodes:
        df, meta = read_episode(ep)
        post = df.loc[df.index > 0.0]
        strong = 0
        for mv in ("stock_flow", "filler_flow", "speed", "steam_p"):
            r = float(np.corrcoef(post["bw"], post[mv])[0, 1])
            if np.isfinite(r) and abs(r) > 0.5:
                strong += 1
        if strong >= 3:
            ambiguous += 1

        lag, _, _ = mass_balance_lag_sec(df, meta["machine"])
        assert lag > 0.0, f"{ep.name}: lagged structure should be identifiable"

    assert ambiguous >= 0.8 * len(episodes), (
        f"only {ambiguous}/{len(episodes)} episodes are confounded at lag 0; "
        "the causal structure has become trivially recoverable by correlation"
    )


def test_generated_episodes_pass_the_plausibility_gate(dataset) -> None:
    """The gate is the contract with every downstream module; run it in-process so a
    regression fails here rather than at dataset build time."""
    from tests.physics_check import check_dataset

    _, out = dataset
    result = check_dataset(out)
    assert result.passed, json.dumps(result.failures, indent=2)
