"""Score the engine against simulated ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ccee.mkt import mkt, mkt_decimal
from ccee.monitor import Monitor, MonitorConfig, run_monitor
from ccee.sim.fleet import Shipment
from ccee.sim.logger import correct_clock

TRUTH_MIN_MINUTES = 10  # a true excursion = payload out of range for at least this long
MATCH_TOL_MIN = 15.0


@dataclass
class TruthEvent:
    start: pd.Timestamp
    end: pd.Timestamp
    direction: str
    cause: str = "natural"

    @property
    def minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0 + 1.0


def truth_events(ship: Shipment, min_minutes: int = TRUTH_MIN_MINUTES) -> list[TruthEvent]:
    p, x = ship.profile, ship.truth
    out: list[TruthEvent] = []
    for direction, mask in (("high", x > p.hi), ("low", x < p.lo)):
        d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=True):
            if b - a >= min_minutes:
                out.append(TruthEvent(ship.t0 + pd.Timedelta(minutes=int(a)), ship.t0 + pd.Timedelta(minutes=int(b - 1)), direction))
    out.sort(key=lambda e: e.start)
    for ev in out:
        ev.cause = _attribute(ship, ev)
    return out


def _attribute(ship: Shipment, ev: TruthEvent) -> str:
    onset_min = (ev.start - ship.t0).total_seconds() / 60.0
    best, best_start = "natural", -1
    for f in ship.faults:
        if f["start"] - 30 <= onset_min <= f["end"] + 720 or f["kind"] == "pcm_under":
            if f["start"] > best_start:
                best, best_start = f["kind"], f["start"]
    return best


def breach_starts(ship: Shipment) -> dict[str, list[pd.Timestamp]]:
    """Start of every true breach of any length (used to judge whether an early warning was justified)."""
    p, x = ship.profile, ship.truth
    res: dict[str, list[pd.Timestamp]] = {}
    for direction, mask in (("high", x > p.hi), ("low", x < p.lo)):
        d = np.diff(np.concatenate([[0], mask.astype(int)]))
        res[direction] = [ship.t0 + pd.Timedelta(minutes=int(i)) for i in np.where(d == 1)[0]]
    return res


@dataclass
class ShipmentResult:
    ship: Shipment
    monitor: Monitor
    truth: list[TruthEvent]
    matched_truth: list[bool]
    matched_det: list[bool]
    ew_lead_min: list[float | None]  # per truth event
    ew_false: int
    ew_total: int
    mkt_logger: float
    mkt_truth: float
    start_err_min: list[float] = field(default_factory=list)


def corrected_logger(ship: Shipment) -> pd.DataFrame:
    return correct_clock(ship.logger, ship.meta.sync_start_offset_s, ship.meta.sync_end_offset_s)


def analyze_shipment(ship: Shipment, config: MonitorConfig | None = None, correct: bool = True) -> ShipmentResult:
    cfg = config or MonitorConfig(tau_s=ship.container.tau_s)
    df = corrected_logger(ship) if correct else ship.logger
    mon = run_monitor(df, ship.profile, cfg)
    truth = truth_events(ship)
    tol = pd.Timedelta(minutes=MATCH_TOL_MIN)
    matched_truth = [False] * len(truth)
    matched_det = [False] * len(mon.events)
    start_err: list[float] = []
    for i, d in enumerate(mon.events):
        d_end = d.end or d.start
        for j, t in enumerate(truth):
            if d.direction == t.direction and d.start <= t.end + tol and d_end >= t.start - tol:
                matched_det[i] = True
                if not matched_truth[j]:
                    start_err.append((d.start - t.start).total_seconds() / 60.0)
                matched_truth[j] = True

    horizon = pd.Timedelta(minutes=cfg.ew_horizon_min + 30)
    br = breach_starts(ship)
    lead: list[float | None] = []
    for t in truth:
        cands = [w for w in mon.warnings if w.direction == t.direction and pd.Timedelta(0) <= t.start - w.ts <= horizon]
        lead.append(max((t.start - w.ts).total_seconds() / 60.0 for w in cands) if cands else None)
    false = 0
    for w in mon.warnings:
        if not any(pd.Timedelta(0) <= b - w.ts <= horizon for b in br[w.direction]):
            false += 1
    return ShipmentResult(
        ship, mon, truth, matched_truth, matched_det, lead, false, len(mon.warnings),
        mon.mkt_c, mkt(ship.truth, delta_h_kj_mol=ship.profile.delta_h_kj_mol), start_err,
    )


def summarize(
    fleet: list[Shipment], config_factory=None, correct: bool = True, numeric: bool = True
) -> tuple[dict, list[ShipmentResult]]:
    results = [
        analyze_shipment(s, config_factory(s) if config_factory else None, correct=correct) for s in fleet
    ]
    return aggregate(results, numeric), results


def aggregate(results: list[ShipmentResult], numeric: bool = True) -> dict:
    n_truth = sum(len(r.truth) for r in results)
    n_det = sum(len(r.monitor.events) for r in results)
    tp_truth = sum(sum(r.matched_truth) for r in results)
    tp_det = sum(sum(r.matched_det) for r in results)
    leads = [x for r in results for x in r.ew_lead_min if x is not None]
    n_warn = sum(r.ew_total for r in results)
    n_false = sum(r.ew_false for r in results)
    days = sum(r.ship.duration_h for r in results) / 24.0
    by_cause: dict[str, list[int]] = {}
    for r in results:
        for t, m in zip(r.truth, r.matched_truth, strict=True):
            c = by_cause.setdefault(t.cause, [0, 0])
            c[0] += 1
            c[1] += int(m)
    ship_truth = [bool(r.truth) for r in results]
    ship_det = [bool(r.monitor.events) for r in results]
    tp_s = sum(a and b for a, b in zip(ship_truth, ship_det, strict=True))
    errs = [e for r in results for e in r.start_err_min]
    mkt_err = [abs(r.mkt_logger - r.mkt_truth) for r in results if np.isfinite(r.mkt_logger)]
    num_err = (
        [
            abs(mkt(r.ship.logger["temp_c"].to_numpy()) - mkt_decimal(r.ship.logger["temp_c"].to_numpy().tolist()))
            for r in results
        ]
        if numeric
        else [float("nan")]
    )
    return {
        "n_shipments": len(results),
        "shipment_days": round(days, 1),
        "n_true_excursions": n_truth,
        "n_detected_events": n_det,
        "event_recall": _ratio(tp_truth, n_truth),
        "event_precision": _ratio(tp_det, n_det),
        "detections_per_true_excursion": round(tp_det / tp_truth, 2) if tp_truth else None,
        "shipments_with_true_excursion": int(sum(ship_truth)),
        "shipment_recall": _ratio(tp_s, sum(ship_truth)),
        "shipment_precision": _ratio(tp_s, sum(ship_det)),
        "recall_by_cause": {k: {"n": v[0], "recall": _ratio(v[1], v[0])} for k, v in sorted(by_cause.items())},
        "start_time_error_min_median_abs": float(np.median(np.abs(errs))) if errs else None,
        "early_warning": {
            "warned_fraction": _ratio(len(leads), n_truth),
            "lead_min_median": float(np.median(leads)) if leads else None,
            "lead_min_p25": float(np.percentile(leads, 25)) if leads else None,
            "lead_min_p75": float(np.percentile(leads, 75)) if leads else None,
            "n_warnings": n_warn,
            "warning_precision": _ratio(n_warn - n_false, n_warn),
            "false_alarms_per_100_shipment_days": round(100 * n_false / days, 2) if days else None,
        },
        "mkt": {
            "logger_vs_true_mae_c": float(np.mean(mkt_err)),
            "logger_vs_true_p95_c": float(np.percentile(mkt_err, 95)),
            "logger_vs_true_max_c": float(np.max(mkt_err)),
            "float_vs_40digit_max_abs_c": float(np.max(num_err)),
        },
    }


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None
