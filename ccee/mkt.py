"""Mean kinetic temperature (Arrhenius form).

MKT = (dH/R) / -ln( mean( exp(-dH/(R*T_i)) ) ),  T in kelvin, default dH = 83.144 kJ/mol so dH/R = 10000 K.
"""

from __future__ import annotations

import numpy as np

R_KJ_MOL_K = 0.0083144
KELVIN = 273.15
DEFAULT_DELTA_H = 83.144


def mkt(
    temps_c,
    times=None,
    delta_h_kj_mol: float = DEFAULT_DELTA_H,
    max_gap_s: float | None = None,
) -> float:
    """MKT in degrees C.

    Without `times` this is the pharmacopoeia form (simple mean over equally spaced readings).
    With `times` (datetime64 or seconds) terms are trapezoid time-weighted; intervals longer than
    `max_gap_s` (data gaps) are excluded rather than interpolated.
    """
    t = np.asarray(temps_c, dtype=float)
    ok = np.isfinite(t)
    if times is None:
        t = t[ok]
        if t.size == 0:
            return float("nan")
        terms = _term(t, delta_h_kj_mol)
        return _inverse(float(terms.mean()), delta_h_kj_mol)
    sec = _to_seconds(times)
    if sec.shape != t.shape:
        raise ValueError("temps and times must have the same length")
    sec, t = sec[ok], t[ok]
    if t.size == 0:
        return float("nan")
    if t.size == 1:
        return float(t[0])
    order = np.argsort(sec, kind="stable")
    sec, t = sec[order], t[order]
    dt = np.diff(sec)
    terms = _term(t, delta_h_kj_mol)
    w = dt.copy()
    if max_gap_s is not None:
        w[dt > max_gap_s] = 0.0
    if w.sum() <= 0:
        return float(np.mean(t))
    avg = float(np.sum(w * 0.5 * (terms[:-1] + terms[1:])) / w.sum())
    return _inverse(avg, delta_h_kj_mol)


def _term(t_c: np.ndarray, dh: float) -> np.ndarray:
    return np.exp(-(dh / R_KJ_MOL_K) / (t_c + KELVIN))


def _inverse(mean_term: float, dh: float) -> float:
    return (dh / R_KJ_MOL_K) / (-np.log(mean_term)) - KELVIN


def _to_seconds(times) -> np.ndarray:
    arr = np.asarray(times)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype("int64") / 1e9
    if arr.dtype == object:  # tz-aware pandas timestamps
        import pandas as pd

        return pd.to_datetime(pd.Series(arr), utc=True).astype("int64").to_numpy() / 1e9
    return arr.astype(float)


class RunningMKT:
    """Incremental time-weighted MKT for streaming use."""

    def __init__(self, delta_h_kj_mol: float = DEFAULT_DELTA_H, max_gap_s: float = 1800.0):
        self.dh = delta_h_kj_mol
        self.max_gap_s = max_gap_s
        self._last: tuple[float, float] | None = None  # (seconds, term)
        self._acc = 0.0
        self._w = 0.0
        self._last_temp: float | None = None

    def update(self, t_seconds: float, temp_c: float) -> float:
        term = float(_term(np.asarray(temp_c), self.dh))
        if self._last is not None:
            dt = t_seconds - self._last[0]
            if 0 < dt <= self.max_gap_s:
                self._acc += dt * 0.5 * (self._last[1] + term)
                self._w += dt
        self._last = (t_seconds, term)
        self._last_temp = temp_c
        return self.value

    @property
    def value(self) -> float:
        if self._w <= 0:
            return float("nan") if self._last_temp is None else float(self._last_temp)
        return _inverse(self._acc / self._w, self.dh)


def mkt_decimal(temps_c, delta_h_kj_mol: float = DEFAULT_DELTA_H, digits: int = 40) -> float:
    """Independent 40-digit reference implementation of the simple-mean MKT, for validating the float version."""
    from decimal import Decimal, localcontext

    with localcontext() as ctx:
        ctx.prec = digits
        return _mkt_decimal(temps_c, delta_h_kj_mol, Decimal)


def _mkt_decimal(temps_c, delta_h_kj_mol, Decimal) -> float:
    dh_over_r = Decimal(str(delta_h_kj_mol)) / Decimal(str(R_KJ_MOL_K))
    k = Decimal("273.15")
    vals = [t for t in temps_c if t == t]
    if not vals:
        return float("nan")
    acc = Decimal(0)
    for t in vals:
        acc += (-(dh_over_r) / (Decimal(str(float(t))) + k)).exp()
    mean = acc / Decimal(len(vals))
    return float(dh_over_r / (-(mean.ln())) - k)
