"""Grade catalogue and per-grade steady-state operating points.

Machine speed is a *grade* property, not a machine constant: basis weight is
inversely proportional to speed, so one speed cannot serve a 45-160 g/m2 catalogue
inside a fixed actuator envelope. Light grades run fast, heavy grades slow.

The operating-point solver is the guard against the envelope bug: every grade must
have a feasible steady state with headroom on every actuator, or the catalogue is
wrong and no amount of controller tuning will save it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.sim.machine import CaliperModel, DryingModel, MachineConfig, required_flows
from src.sim.units import RANGES

__all__ = [
    "GradeProps",
    "OperatingPoint",
    "RECIPE_LIMITS",
    "ACTUATOR_RATES",
    "NOMINAL_STOCK_CONS_PCT",
    "MARGIN_FRACTION",
    "build_catalogue",
    "operating_point",
    "actuator_margins",
    "catalogue_infeasibilities",
    "write_grades_json",
    "load_grades",
]

#: Recipe envelope, canonical units (stock_flow in m3/h). Written into every
#: ``meta.json`` so the advisor's constraint filter has explicit bounds.
RECIPE_LIMITS: dict[str, list[float]] = {
    "stock_flow": [550.0, 1900.0],
    "filler_flow": [0.0, 720.0],
    "steam_p": [1.2, 5.5],
    "speed": [420.0, 1700.0],
}

#: Maximum rate of change per minute, canonical units per minute.
ACTUATOR_RATES: dict[str, float] = {
    "stock_flow": 120.0,
    "speed": 85.0,
    "steam_p": 0.4,
}

NOMINAL_STOCK_CONS_PCT: float = 3.5

#: Required headroom to each recipe limit, as a fraction of the limit itself.
#: A point is feasible-with-margin when ``x >= (1+m)*lo`` and ``x <= (1-m)*hi``.
MARGIN_FRACTION: float = 0.15


@dataclass(frozen=True)
class GradeProps:
    """One grade. ``bw``/``ash``/``moist``/``caliper`` form the advisor's embedding;
    ``nominal_speed_m_min`` is an operating parameter and is deliberately excluded
    from it."""

    code: str
    bw: float
    ash: float
    moist: float
    caliper: float
    nominal_speed_m_min: float

    def props(self) -> dict[str, float]:
        """The four-dimensional property vector used by ``grade_*_props`` in meta."""
        return {"bw": self.bw, "ash": self.ash, "moist": self.moist, "caliper": self.caliper}


@dataclass(frozen=True)
class OperatingPoint:
    """Feasible steady state that realises a grade."""

    stock_flow_m3_h: float
    filler_flow_L_min: float
    steam_p_bar: float
    speed_m_min: float
    stock_cons_pct: float


# (code, bw g/m2, ash %, moist %, nominal speed m/min). Caliper is derived from the
# bulk model so the catalogue and the simulator cannot disagree. G07 and G12 are the
# two grades worked through in the episode schema and reproduce its numbers exactly.
_CATALOGUE_SPEC: tuple[tuple[str, float, float, float, float], ...] = (
    ("G01", 45.0, 6.5, 5.0, 1400.0),
    ("G02", 52.0, 14.0, 5.4, 1340.0),
    ("G03", 56.0, 22.0, 6.6, 1300.0),
    ("G04", 59.0, 26.0, 7.4, 1285.0),
    ("G05", 60.0, 6.8, 4.8, 1270.0),
    ("G06", 72.0, 25.0, 7.6, 1080.0),
    ("G07", 64.0, 12.0, 5.8, 1180.0),
    ("G08", 90.0, 20.0, 6.8, 910.0),
    ("G09", 105.0, 15.0, 6.0, 790.0),
    ("G10", 128.0, 9.0, 5.6, 630.0),
    ("G11", 160.0, 7.5, 8.0, 505.0),
    ("G12", 82.0, 18.0, 6.2, 1000.0),
)


def build_catalogue(caliper_model: CaliperModel | None = None) -> dict[str, GradeProps]:
    """Deterministic 12-grade catalogue."""
    cal = caliper_model or CaliperModel()
    out: dict[str, GradeProps] = {}
    for code, bw, ash, moist, speed in _CATALOGUE_SPEC:
        caliper = round(float(cal.caliper_um(bw, ash)), 1)
        out[code] = GradeProps(
            code=code,
            bw=float(bw),
            ash=float(ash),
            moist=float(moist),
            caliper=caliper,
            nominal_speed_m_min=float(speed),
        )
    return out


def operating_point(
    grade: GradeProps,
    machine: MachineConfig,
    *,
    stock_cons_pct: float = NOMINAL_STOCK_CONS_PCT,
    drying: DryingModel | None = None,
) -> OperatingPoint:
    """Solve the steady state that holds ``grade`` at its nominal speed."""
    dry = drying or DryingModel()
    stock, filler = required_flows(
        grade.bw,
        grade.ash,
        grade.nominal_speed_m_min,
        stock_cons_pct,
        trim_m=machine.trim_m,
        retention=machine.retention,
        filler_cons_pct=machine.filler_cons_pct,
    )
    steam = float(dry.steam_p_bar(grade.moist, grade.bw, grade.nominal_speed_m_min))
    return OperatingPoint(
        stock_flow_m3_h=stock,
        filler_flow_L_min=filler,
        steam_p_bar=steam,
        speed_m_min=grade.nominal_speed_m_min,
        stock_cons_pct=stock_cons_pct,
    )


def actuator_margins(
    op: OperatingPoint, limits: dict[str, list[float]] | None = None
) -> dict[str, float]:
    """Fractional headroom to the nearest recipe limit, per actuator.

    ``1.0`` means the actuator sits at the middle of nowhere near a limit; ``0.0``
    means it is exactly on one. A grade is acceptable when every entry is
    >= :data:`MARGIN_FRACTION`. Headroom is expressed relative to the limit itself
    (the usual process reading of "15% off the limit"), which also handles the
    ``filler_flow`` lower limit of exactly zero without special-casing.
    """
    lim = limits or RECIPE_LIMITS
    values = {
        "stock_flow": op.stock_flow_m3_h,
        "filler_flow": op.filler_flow_L_min,
        "steam_p": op.steam_p_bar,
        "speed": op.speed_m_min,
    }
    out: dict[str, float] = {}
    for name, x in values.items():
        lo, hi = lim[name]
        up = (hi - x) / hi if hi > 0 else float("inf")
        down = (x - lo) / lo if lo > 0 else float("inf")
        out[name] = float(min(up, down))
    return out


def catalogue_infeasibilities(
    catalogue: dict[str, GradeProps],
    machine: MachineConfig,
    *,
    limits: dict[str, list[float]] | None = None,
    margin: float = MARGIN_FRACTION,
    stock_cons_pct: float = NOMINAL_STOCK_CONS_PCT,
) -> list[str]:
    """Report every grade whose steady state lacks ``margin`` on any actuator, or
    whose properties leave their canonical range."""
    problems: list[str] = []
    for code, grade in catalogue.items():
        for field, key in (("bw", "bw"), ("ash", "ash"), ("moist", "moist"), ("caliper", "caliper")):
            value = getattr(grade, field)
            rng = RANGES[key]
            if not (rng.lo <= value <= rng.hi):
                problems.append(f"{code}: {field}={value} outside [{rng.lo}, {rng.hi}] {rng.unit}")
        op = operating_point(grade, machine, stock_cons_pct=stock_cons_pct)
        for name, m in actuator_margins(op, limits).items():
            if m < margin:
                problems.append(
                    f"{code}: {name} margin {m * 100:.1f}% < {margin * 100:.0f}% "
                    f"at the nominal operating point"
                )
    return problems


def write_grades_json(path: str | Path, catalogue: dict[str, GradeProps] | None = None) -> Path:
    """Write ``data/grades.json``. Deterministic, sorted by grade code."""
    cat = catalogue or build_catalogue()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {code: _grade_payload(cat[code]) for code in sorted(cat)}
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def _grade_payload(g: GradeProps) -> dict[str, float]:
    d = asdict(g)
    d.pop("code")
    return d


def load_grades(path: str | Path) -> dict[str, GradeProps]:
    """Load a grade catalogue written by :func:`write_grades_json`."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        code: GradeProps(
            code=code,
            bw=float(v["bw"]),
            ash=float(v["ash"]),
            moist=float(v["moist"]),
            caliper=float(v["caliper"]),
            nominal_speed_m_min=float(v["nominal_speed_m_min"]),
        )
        for code, v in sorted(raw.items())
    }
