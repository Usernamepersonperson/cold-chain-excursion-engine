"""End-to-end analysis of one logger record: monitor + MKT + integrity flags + audit trail."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ccee import __version__
from ccee.audit import AuditLog, canonical
from ccee.integrity import Flag, check_integrity
from ccee.mkt import mkt
from ccee.monitor import Monitor, MonitorConfig, run_monitor
from ccee.profiles import ProductProfile
from ccee.sim.weather import LOCATIONS


@dataclass
class Analysis:
    df: pd.DataFrame
    profile: ProductProfile
    config: MonitorConfig
    monitor: Monitor
    flags: list[Flag]
    audit: AuditLog
    metadata: dict = field(default_factory=dict)
    mkt_c: float = float("nan")
    stats: dict = field(default_factory=dict)
    locations: list[str] = field(default_factory=list)  # one per excursion


def analyze_record(
    df: pd.DataFrame,
    profile: ProductProfile,
    config: MonitorConfig | None = None,
    metadata: dict | None = None,
    rejects: int = 0,
    sync_offsets_s: tuple[float, float] | None = None,
    actor: str = "ccee",
    audit: AuditLog | None = None,
) -> Analysis:
    cfg = config or MonitorConfig()
    meta = dict(metadata or {})
    audit = audit or AuditLog()
    df = df.sort_values("ts", kind="stable").reset_index(drop=True) if not df["ts"].is_monotonic_increasing else df
    audit.append(
        "record_loaded",
        actor,
        {"rows": int(len(df)), "rejected_rows": rejects, "source_sha256": meta.get("source_sha256"),
         "first": str(df["ts"].iloc[0]) if len(df) else None, "last": str(df["ts"].iloc[-1]) if len(df) else None},
    )
    if sync_offsets_s is not None:
        audit.append("clock_correction", actor, {"start_offset_s": sync_offsets_s[0], "end_offset_s": sync_offsets_s[1], "method": "linear"})
    audit.append(
        "analysis_run", actor,
        {"engine_version": __version__, "profile": profile.name, "range_c": [profile.lo, profile.hi],
         "config": canonical(cfg.__dict__), "delta_h_kj_mol": profile.delta_h_kj_mol},
    )
    mon = run_monitor(df, profile, cfg)
    flags = check_integrity(
        df, max_gap_min=cfg.max_gap_min, rejects=rejects, metadata=meta, sync_offsets_s=sync_offsets_s
    )
    for fl in flags:
        audit.append("integrity_flag", actor, {"code": fl.code, "severity": fl.severity, "alcoa": fl.alcoa, "message": fl.message})
    for ev in mon.events:
        audit.append(
            "excursion_detected", actor,
            {"direction": ev.direction, "start": str(ev.start), "end": str(ev.end), "peak_c": round(ev.peak_c, 2),
             "minutes": round(ev.minutes, 1)},
        )
    for a in mon.alarms:
        audit.append("alarm", actor, {"ts": str(a.ts), "rule": a.rule, "severity": a.severity, "message": a.message})
    m = mkt(df["temp_c"].to_numpy(), df["ts"].to_numpy(), profile.delta_h_kj_mol, cfg.max_gap_min * 60.0) if len(df) else float("nan")
    t = df["temp_c"].to_numpy(float)
    stats = {
        "n": int(len(df)),
        "min_c": float(np.nanmin(t)) if len(t) else math.nan,
        "max_c": float(np.nanmax(t)) if len(t) else math.nan,
        "mean_c": float(np.nanmean(t)) if len(t) else math.nan,
        "duration_h": float((df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds() / 3600.0) if len(df) > 1 else 0.0,
        "observed_h": mon.minutes_observed / 60.0,
        "in_range_pct": 100.0 * mon.minutes_in_range / mon.minutes_observed if mon.minutes_observed else math.nan,
        "gap_h": sum(g.minutes for g in mon.gaps) / 60.0,
    }
    audit.append("mkt_computed", actor, {"mkt_c": round(m, 3), "delta_h_kj_mol": profile.delta_h_kj_mol})
    return Analysis(df, profile, cfg, mon, flags, audit, meta, m, stats, [locate(df, ev.start) for ev in mon.events])


def locate(df: pd.DataFrame, when: pd.Timestamp, max_km: float = 200.0) -> str:
    """Nearest GPS fix in time, labelled with the nearest known site if close enough."""
    if "lat" not in df or df["lat"].notna().sum() == 0:
        return "no GPS in record"
    g = df.dropna(subset=["lat", "lon"])
    i = int((g["ts"] - when).abs().argmin())
    lat, lon = float(g["lat"].iloc[i]), float(g["lon"].iloc[i])
    best, best_d = None, 1e9
    for loc in LOCATIONS.values():
        d = _hav(lat, lon, loc.lat, loc.lon)
        if d < best_d:
            best, best_d = loc, d
    if best is not None and best_d <= max_km:
        return f"near {best.name} ({best.code}), fix {abs((g['ts'].iloc[i] - when).total_seconds()) / 60:.0f} min away"
    return f"in transit ({lat:.2f}, {lon:.2f}), fix {abs((g['ts'].iloc[i] - when).total_seconds()) / 60:.0f} min away"


def _hav(la1: float, lo1: float, la2: float, lo2: float) -> float:
    p1, p2 = math.radians(la1), math.radians(la2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))
