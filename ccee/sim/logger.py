"""Data-logger behaviour: sampling on a drifting clock, sensor bias/noise/resolution, GPS fixes, data faults."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LoggerMeta:
    interval_min: int
    drift_ppm: float
    offset_s: float
    bias_c: float
    noise_sd_c: float
    sync_start_offset_s: float
    sync_end_offset_s: float
    data_faults: list[dict] = field(default_factory=list)


def simulate_logger(
    t0: pd.Timestamp,
    truth_c: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    in_flight: np.ndarray,
    rng: np.random.Generator,
    interval_min: int = 5,
    gps_every: int = 3,
) -> tuple[pd.DataFrame, LoggerMeta]:
    n = len(truth_c)
    minute_s = np.arange(n) * 60.0
    drift = rng.normal(0, 1000e-6) if rng.random() < 0.1 else rng.normal(0, 100e-6)
    offset = rng.normal(0, 20.0)
    bias = rng.normal(0, 0.12)
    noise = float(rng.uniform(0.05, 0.22))
    step = interval_min * 60.0

    k = np.arange(0, int((n * 60 * 1.01) / step) + 2)
    s = k * step
    tau = (s - offset) / (1.0 + drift)
    keep = (tau >= 0) & (tau <= minute_s[-1])
    s, tau = s[keep], tau[keep]
    temp = np.interp(tau, minute_s, truth_c) + bias + rng.normal(0, noise, len(tau))
    temp = np.round(temp, 1)

    la = np.interp(tau, minute_s, lat) + rng.normal(0, 0.0003, len(tau))
    lo = np.interp(tau, minute_s, lon) + rng.normal(0, 0.0003, len(tau))
    fl = np.interp(tau, minute_s, in_flight.astype(float)) > 0.5
    fix = np.zeros(len(tau), dtype=bool)
    fix[::gps_every] = True
    fix &= ~(fl & (rng.random(len(tau)) < 0.8))
    la[~fix] = np.nan
    lo[~fix] = np.nan

    faults: list[dict] = []
    m = len(tau)
    alive = np.ones(m, dtype=bool)
    if rng.random() < 0.35 and m > 100:
        dur = int(rng.integers(60, 361) / interval_min)
        a = int(rng.integers(int(0.1 * m), int(0.9 * m)))
        alive[a : a + dur] = False
        faults.append({"kind": "gap", "start_s": float(tau[a]), "minutes": dur * interval_min})
    if rng.random() < 0.30 and m > 100:
        for _ in range(int(rng.integers(1, 4))):
            i = int(rng.integers(5, m - 5))
            temp[i] = round(temp[i] + float(rng.choice([-1, 1]) * rng.uniform(6, 15)), 1)
            faults.append({"kind": "spike", "start_s": float(tau[i]), "minutes": interval_min})
    if rng.random() < 0.10 and m > 100:
        dur = int(rng.integers(120, 301) / interval_min)
        a = int(rng.integers(int(0.1 * m), int(0.8 * m)))
        temp[a : a + dur] = temp[a]
        faults.append({"kind": "stuck", "start_s": float(tau[a]), "minutes": dur * interval_min})
    if rng.random() < 0.05 and m > 100:
        cut = int(m * rng.uniform(0.85, 0.95))
        alive[cut:] = False
        faults.append({"kind": "truncated", "start_s": float(tau[cut]), "minutes": (m - cut) * interval_min})

    df = pd.DataFrame(
        {
            "ts": t0 + pd.to_timedelta(s, unit="s"),
            "temp_c": temp,
            "lat": la,
            "lon": lo,
        }
    )[alive].reset_index(drop=True)
    total_true = minute_s[-1]
    meta = LoggerMeta(
        interval_min=interval_min,
        drift_ppm=float(drift * 1e6),
        offset_s=float(offset),
        bias_c=float(bias),
        noise_sd_c=noise,
        sync_start_offset_s=float(offset + rng.normal(0, 1.0)),
        sync_end_offset_s=float(offset + drift * total_true + rng.normal(0, 1.0)),
        data_faults=faults,
    )
    return df, meta


def correct_clock(df: pd.DataFrame, start_offset_s: float, end_offset_s: float) -> pd.DataFrame:
    """Remove a linear clock error given the logger-minus-reference offset read at start and at download.

    Offsets are positive when the logger clock runs ahead of the reference.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    t = out["ts"]
    span = (t.iloc[-1] - t.iloc[0]).total_seconds()
    frac = (t - t.iloc[0]).dt.total_seconds() / span if span > 0 else 0.0
    err = start_offset_s + (end_offset_s - start_offset_s) * frac
    out["ts"] = t - pd.to_timedelta(err, unit="s")
    return out
