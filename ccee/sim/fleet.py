"""Generate simulated shipments with injected faults and ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ccee.profiles import ProductProfile, get_profile
from ccee.sim.logger import LoggerMeta, simulate_logger
from ccee.sim.route import ROUTES, Segment, ambient_series, build_itinerary, layout, position_series
from ccee.sim.thermal import CONTAINERS, Container, simulate_payload
from ccee.sim.weather import WeatherArchive

PHYSICAL_FAULTS = {
    "passive_2_8": ["tarmac_delay", "customs_hold", "door_open", "pcm_under", "cold_exposure"],
    "active_2_8": ["tarmac_delay", "customs_hold", "door_open", "power_loss", "cold_exposure"],
    "passive_15_25": ["tarmac_delay", "customs_hold", "door_open", "pcm_under", "cold_exposure"],
    "passive_frozen": ["tarmac_delay", "customs_hold", "door_open", "pcm_under"],
}
START_STATE = {  # (start temperature offset from pack temperature, melt fraction)
    "passive_2_8": 0.4,
    "active_2_8": 0.0,
    "passive_15_25": 0.5,
    "passive_frozen": 0.05,
}


@dataclass
class Shipment:
    shipment_id: str
    route: str
    window: str
    profile: ProductProfile
    container: Container
    t0: pd.Timestamp
    truth: np.ndarray  # per-minute true payload temperature
    ambient: np.ndarray
    logger: pd.DataFrame  # as recorded, logger clock
    meta: LoggerMeta
    faults: list[dict] = field(default_factory=list)  # physical faults with true windows (minutes from t0)
    ua_scale: float = 1.0

    @property
    def duration_h(self) -> float:
        return len(self.truth) / 60.0

    def truth_series(self) -> pd.Series:
        return pd.Series(self.truth, index=self.t0 + pd.to_timedelta(np.arange(len(self.truth)), unit="min"))


def _apply_fault(kind: str, segs: list[Segment], rng: np.random.Generator) -> dict:
    """Modify the itinerary for a fault and return a record (window resolved after layout)."""
    if kind == "tarmac_delay":
        cands = [s for s in segs if s.kind == "tarmac"]
        s = cands[int(rng.integers(0, len(cands)))]
        extra = int(rng.integers(240, 1200))
        s.minutes += extra
        return {"kind": kind, "seg": s, "offset": s.minutes - extra, "minutes": extra}
    if kind == "customs_hold":
        cands = [s for s in segs if s.kind == "dock"]
        s = cands[-1]
        extra = int(rng.integers(600, 2700))
        s.minutes += extra
        return {"kind": kind, "seg": s, "offset": s.minutes - extra, "minutes": extra}
    if kind == "cold_exposure":
        i = int(rng.integers(1, len(segs) - 2))
        dur = int(rng.integers(120, 480))
        seg = Segment("cold", segs[i].loc, dur, cold_c=float(rng.uniform(-25, -8)))
        segs.insert(i, seg)
        return {"kind": kind, "seg": seg, "offset": 0, "minutes": dur}
    # door_open, power_loss, pcm_under are resolved on the timeline
    return {"kind": kind, "seg": None, "offset": 0, "minutes": 0}


def make_shipment(
    idx: int,
    rng: np.random.Generator,
    archives: dict[str, WeatherArchive],
    fault_prob: float = 0.55,
    interval_min: int = 5,
) -> Shipment:
    window = "summer" if rng.random() < 0.5 else "winter"
    route = list(ROUTES)[int(rng.integers(0, len(ROUTES)))]
    pname = str(rng.choice(["2-8C", "15-25C", "frozen"], p=[0.55, 0.25, 0.20]))
    if pname == "2-8C":
        cname = "passive_2_8" if rng.random() < 0.65 else "active_2_8"
    else:
        cname = {"15-25C": "passive_15_25", "frozen": "passive_frozen"}[pname]
    container = CONTAINERS[cname]
    profile = get_profile(pname)
    archive = archives[window]

    segs = build_itinerary(ROUTES[route], rng)
    faults: list[dict] = []
    if rng.random() < fault_prob:
        n_f = 2 if rng.random() < 0.15 else 1
        for kind in rng.choice(PHYSICAL_FAULTS[cname], size=n_f, replace=False):
            faults.append(_apply_fault(str(kind), segs, rng))
    total = layout(segs)

    start_latest = archive.end - pd.Timedelta(hours=total / 60.0 + 6)
    span_s = max(3600.0, (start_latest - archive.start).total_seconds())
    t0 = (archive.start + pd.Timedelta(seconds=float(rng.uniform(0, span_s)))).floor("min")
    t0_epoch = t0.timestamp()

    hold_c = 15.0 + float(rng.uniform(-2, 3))
    room_c = 20.0 + float(rng.uniform(-1.5, 1.5))
    amb = ambient_series(segs, t0_epoch, archive, hold_c, room_c)
    mult = np.ones(total)
    power = np.ones(total, dtype=bool)
    melt0 = START_STATE[cname]

    for f in faults:
        kind = f["kind"]
        if f["seg"] is not None:
            f["start"] = f["seg"].start + f["offset"]
            f["end"] = f["start"] + f["minutes"]
        elif kind == "door_open":
            cands = [s for s in segs if s.kind in ("tarmac", "dock")]
            s = cands[int(rng.integers(0, len(cands)))]
            dur = int(rng.integers(15, 90))
            a = s.start + int(rng.integers(0, max(1, s.minutes - dur)))
            mult[a : a + dur] = float(rng.uniform(8, 25))
            f.update(start=a, end=a + dur, minutes=dur)
        elif kind == "power_loss":
            dur = int(rng.integers(240, 1200))
            a = int(rng.integers(int(0.1 * total), max(int(0.1 * total) + 1, int(0.8 * total))))
            power[a : a + dur] = False
            f.update(start=a, end=min(total, a + dur), minutes=dur)
        elif kind == "pcm_under":
            melt0 = float(rng.uniform(0.7, 1.0))
            f.update(start=0, end=total, minutes=total)
        f.pop("seg", None)
        f.pop("offset", None)

    ua_scale = float(np.exp(rng.normal(0, 0.12)))
    t_start = container.pcm_temp_c if container.pcm_temp_c is not None else container.setpoint_c
    truth, _ = simulate_payload(container, amb, mult, power, float(t_start), melt0, ua_scale=ua_scale)

    lat, lon = position_series(segs)
    in_flight = np.zeros(total, dtype=bool)
    for s in segs:
        if s.kind == "hold":
            in_flight[s.start : s.start + s.minutes] = True
    df, meta = simulate_logger(t0, truth, lat, lon, in_flight, rng, interval_min=interval_min)
    return Shipment(
        shipment_id=f"SIM-{idx:04d}",
        route=route,
        window=window,
        profile=profile,
        container=container,
        t0=t0,
        truth=truth,
        ambient=amb,
        logger=df,
        meta=meta,
        faults=faults,
        ua_scale=ua_scale,
    )


def make_fleet(
    n: int,
    seed: int = 7,
    archives: dict[str, WeatherArchive] | None = None,
    fault_prob: float = 0.55,
    interval_min: int = 5,
) -> list[Shipment]:
    if archives is None:
        archives = {w: WeatherArchive(w) for w in ("summer", "winter")}
    rng = np.random.default_rng(seed)
    return [make_shipment(i, rng, archives, fault_prob, interval_min) for i in range(n)]
