"""Data-integrity checks on a logger record.

Each flag carries an ALCOA+ attribute it relates to (Attributable, Legible, Contemporaneous, Original,
Accurate, Complete, Consistent, Enduring, Available). The mapping is descriptive, to help a reviewer; passing
these checks is not evidence of regulatory compliance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Flag:
    code: str
    severity: str  # "info" | "warning" | "critical"
    alcoa: str
    message: str


def check_integrity(
    df: pd.DataFrame,
    interval_min: float | None = None,
    max_gap_min: float = 30.0,
    rejects: int = 0,
    metadata: dict | None = None,
    sync_offsets_s: tuple[float, float] | None = None,
    drift_tolerance_s: float = 60.0,
    physical_range: tuple[float, float] = (-80.0, 70.0),
) -> list[Flag]:
    flags: list[Flag] = []
    meta = metadata or {}
    if df.empty:
        return [Flag("empty", "critical", "Complete", "no readings in record")]
    ts = df["ts"]
    temp = df["temp_c"].to_numpy(float)

    if rejects:
        flags.append(Flag("rejected_rows", "warning", "Legible", f"{rejects} input row(s) could not be parsed and were excluded"))
    if not ts.is_monotonic_increasing:
        flags.append(Flag("non_monotonic_time", "critical", "Consistent", "timestamps go backwards; record may be edited or merged"))
    dup = int(ts.duplicated().sum())
    if dup:
        flags.append(Flag("duplicate_timestamps", "warning", "Original", f"{dup} duplicate timestamp(s)"))
    if np.isnan(temp).any():
        flags.append(Flag("missing_values", "warning", "Complete", f"{int(np.isnan(temp).sum())} reading(s) without a value"))

    diffs = ts.diff().dt.total_seconds().dropna().to_numpy() / 60.0
    pos = diffs[diffs > 0]
    if interval_min is None and pos.size:
        interval_min = float(np.median(pos))
    gaps = pos[pos > max_gap_min]
    if gaps.size:
        flags.append(
            Flag("data_gaps", "warning", "Complete",
                 f"{gaps.size} gap(s) longer than {max_gap_min:.0f} min, total {gaps.sum():.0f} min, longest {gaps.max():.0f} min")
        )
    if interval_min and pos.size:
        irregular = int(((pos > 0) & (np.abs(pos - interval_min) > 0.25 * interval_min) & (pos <= max_gap_min)).sum())
        if irregular > 0.05 * pos.size:
            flags.append(Flag("irregular_sampling", "info", "Consistent", f"{irregular} interval(s) deviate >25 % from {interval_min:.1f} min"))

    lo, hi = physical_range
    bad = (temp < lo) | (temp > hi)
    if bad.any():
        flags.append(Flag("implausible_values", "critical", "Accurate", f"{int(bad.sum())} reading(s) outside {lo:.0f}..{hi:.0f} C"))

    step = np.abs(np.diff(temp))
    if step.size > 2:
        isolated = 0
        for i in range(1, len(temp) - 1):
            if abs(temp[i] - temp[i - 1]) > 5 and abs(temp[i] - temp[i + 1]) > 5 and abs(temp[i + 1] - temp[i - 1]) < 2:
                isolated += 1
        if isolated:
            flags.append(Flag("isolated_spikes", "warning", "Accurate", f"{isolated} isolated reading(s) differing >5 C from both neighbours"))

    run = _longest_flat_run(temp)
    if run >= 12 and interval_min:
        flags.append(Flag("flatline", "warning", "Accurate", f"{run} identical consecutive readings (~{run * interval_min:.0f} min): stuck sensor or copy-forward?"))

    now = pd.Timestamp.now(tz="UTC")
    if (ts > now).any():
        flags.append(Flag("future_timestamps", "critical", "Contemporaneous", "timestamps lie in the future"))

    if sync_offsets_s is None:
        flags.append(Flag("no_clock_sync", "info", "Contemporaneous", "no clock offsets recorded at start/download; drift cannot be corrected"))
    else:
        a, b = sync_offsets_s
        if abs(b - a) > drift_tolerance_s:
            flags.append(Flag("clock_drift", "warning", "Contemporaneous", f"logger clock drifted {b - a:+.0f} s over the record (corrected linearly)"))
    for key in ("device_id", "calibration_date"):
        if not meta.get(key):
            flags.append(Flag(f"missing_{key}", "info", "Attributable", f"{key.replace('_', ' ')} not supplied with the record"))
    if meta.get("source_sha256") is None:
        flags.append(Flag("no_source_hash", "info", "Original", "hash of the original file not recorded"))
    return flags


def _longest_flat_run(x: np.ndarray) -> int:
    if x.size == 0:
        return 0
    best = cur = 1
    for i in range(1, len(x)):
        cur = cur + 1 if x[i] == x[i - 1] else 1
        best = max(best, cur)
    return best
