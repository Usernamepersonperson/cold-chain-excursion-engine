"""Causal shipment monitor: excursion state machine, cumulative budgets, alarms, early warning, MKT.

One implementation serves batch analysis (feed every row) and live use (feed rows as they arrive).
Time-out-of-range convention: an excursion runs from the first out-of-range reading to the first
reading back in range (conservative: up to one sampling interval is over-counted per excursion).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from ccee.mkt import RunningMKT
from ccee.profiles import ProductProfile


@dataclass
class MonitorConfig:
    debounce_samples: int = 2  # consecutive out-of-range readings that open an excursion
    max_gap_min: float = 30.0  # longer silence is a data gap
    long_excursion_min: float = 60.0
    rate_alarm_c_per_h: float = 8.0
    budget_warn_fraction: float = 0.8
    gap_alarm_min: float = 60.0
    # early warning
    ew_enabled: bool = True
    ew_method: str = "thermal"  # "thermal" (first-order relaxation) or "linear" (trend extrapolation)
    ew_horizon_min: float = 45.0
    ew_window_min: float = 30.0
    ew_min_points: int = 4
    ew_consecutive: int = 2
    ew_min_rate_c_per_h: float = 0.5
    ew_rearm_min: float = 30.0
    tau_s: float = 20 * 3600.0  # nominal container time constant used by the thermal predictor


@dataclass
class Excursion:
    direction: str  # "high" | "low"
    start: pd.Timestamp
    end: pd.Timestamp | None
    peak_c: float
    degree_minutes: float = 0.0
    open_ended: bool = False  # still out of range when data ended
    onset_unobserved: bool = False  # first out-of-range reading came right after a data gap
    spans_gap: bool = False

    @property
    def minutes(self) -> float:
        return 0.0 if self.end is None else (self.end - self.start).total_seconds() / 60.0


@dataclass
class Alarm:
    ts: pd.Timestamp
    rule: str
    severity: str  # "info" | "warning" | "critical"
    message: str


@dataclass
class EarlyWarning:
    ts: pd.Timestamp
    direction: str
    predicted_c: float
    horizon_min: float
    current_c: float
    rate_c_per_h: float


@dataclass
class Gap:
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0


@dataclass
class Monitor:
    profile: ProductProfile
    config: MonitorConfig = field(default_factory=MonitorConfig)

    def __post_init__(self) -> None:
        self.events: list[Excursion] = []
        self.alarms: list[Alarm] = []
        self.warnings: list[EarlyWarning] = []
        self.gaps: list[Gap] = []
        self.band_minutes = [0.0] * len(self.profile.bands)
        self.unbudgeted_minutes = 0.0
        self.minutes_in_range = 0.0
        self.minutes_observed = 0.0
        self.n_samples = 0
        self._mkt = RunningMKT(self.profile.delta_h_kj_mol, self.config.max_gap_min * 60.0)
        self._prev: tuple[pd.Timestamp, float] | None = None
        self._run: list[tuple[pd.Timestamp, float]] = []
        self._run_dir: str | None = None
        self._open: Excursion | None = None
        self._after_gap = False
        self._run_after_gap = False
        self._rates: deque[float] = deque(maxlen=2)
        self._hist: deque[tuple[float, float]] = deque()
        self._ew_hits = 0
        self._ew_armed = True
        self._ew_clear_since: float | None = None
        self._fired: set[str] = set()
        self._long_fired = False
        self._first_ts: pd.Timestamp | None = None

    # ---------------------------------------------------------------- public
    def update(self, ts: pd.Timestamp, temp_c: float) -> list[Alarm]:
        new: list[Alarm] = []
        if not math.isfinite(temp_c):
            return new
        cfg = self.config
        gap_min = 0.0
        if self._prev is not None:
            gap_min = (ts - self._prev[0]).total_seconds() / 60.0
            if gap_min <= 0:
                return new  # non-monotonic or duplicate stamp; integrity checks report these
        self.n_samples += 1
        if self._first_ts is None:
            self._first_ts = ts
        is_gap = gap_min > cfg.max_gap_min
        if is_gap:
            self._on_gap(ts, new)
        elif self._prev is not None:
            self._account(ts, temp_c, gap_min)
        self._mkt.update(ts.timestamp(), temp_c)
        self._excursion_step(ts, temp_c, is_gap, new)
        if cfg.ew_enabled:
            self._early_warning(ts, temp_c, is_gap)
        self._rate_check(ts, temp_c, new)
        self._budget_check(ts, new)
        self._prev = (ts, temp_c)
        self.alarms.extend(new)
        return new

    def finish(self) -> None:
        if self._open is not None and self._open.end is None:
            last = self._prev[0] if self._prev else self._open.start
            self._open.end = last
            self._open.open_ended = True
            self._open = None

    @property
    def mkt_c(self) -> float:
        return self._mkt.value

    def budget_fractions(self) -> list[float]:
        return [m / b.max_minutes if b.max_minutes > 0 else (math.inf if m > 0 else 0.0)
                for m, b in zip(self.band_minutes, self.profile.bands, strict=True)]

    def budget_consumed(self) -> float:
        """Additive stability budget used (1.0 = allowance exhausted). Unbudgeted out-of-range time counts as inf."""
        if self.unbudgeted_minutes > 0:
            return math.inf
        return float(sum(self.budget_fractions()))

    def total_minutes_out(self) -> float:
        return float(sum(self.band_minutes) + self.unbudgeted_minutes)

    # --------------------------------------------------------------- internals
    def _account(self, ts: pd.Timestamp, temp: float, gap_min: float) -> None:
        prev_temp = self._prev[1]
        worst = temp if abs(temp - self._mid()) > abs(prev_temp - self._mid()) else prev_temp
        self.minutes_observed += gap_min
        if self.profile.in_range(worst):
            self.minutes_in_range += gap_min
            return
        i = self.profile.band_index(worst)
        if i is None:
            self.unbudgeted_minutes += gap_min
        else:
            self.band_minutes[i] += gap_min

    def _mid(self) -> float:
        return 0.5 * (self.profile.lo + self.profile.hi)

    def _on_gap(self, ts: pd.Timestamp, new: list[Alarm]) -> None:
        start = self._prev[0]
        self.gaps.append(Gap(start, ts))
        gm = (ts - start).total_seconds() / 60.0
        if gm >= self.config.gap_alarm_min:
            new.append(Alarm(ts, "data_gap", "warning", f"no data for {gm:.0f} min (ends {ts:%Y-%m-%d %H:%M})"))
        self._after_gap = True
        self._hist.clear()
        self._ew_hits = 0

    def _excursion_step(self, ts: pd.Timestamp, temp: float, is_gap: bool, new: list[Alarm]) -> None:
        direction = self.profile.direction(temp)
        if is_gap and self._open is not None:
            self._open.spans_gap = True
        if direction is None:
            if self._open is not None:
                self._close(ts)
            self._run, self._run_dir = [], None
        else:
            if self._open is not None and self._open.direction != direction:
                self._close(ts)
            if self._run_dir != direction:
                self._run, self._run_dir = [], direction
                self._run_after_gap = self._after_gap
            self._run.append((ts, temp))
            if self._open is not None:
                self._open.peak_c = max(self._open.peak_c, temp) if direction == "high" else min(self._open.peak_c, temp)
                self._extend_degree_minutes(ts, temp)
                self._check_long(ts, new)
            elif len(self._run) >= self.config.debounce_samples:
                start = self._run[0][0]
                peak = max(t for _, t in self._run) if direction == "high" else min(t for _, t in self._run)
                self._open = Excursion(direction, start, None, peak, onset_unobserved=self._run_after_gap)
                self.events.append(self._open)
                self._long_fired = False
                self._ew_arm_reset()
                lim = self.profile.hi if direction == "high" else self.profile.lo
                new.append(Alarm(ts, "excursion", "critical",
                                 f"{direction} excursion: {temp:.1f} C vs limit {lim:.1f} C (since {start:%H:%M})"))
        self._after_gap = False

    def _extend_degree_minutes(self, ts: pd.Timestamp, temp: float) -> None:
        if self._prev is None or self._open is None:
            return
        dt = (ts - self._prev[0]).total_seconds() / 60.0
        lim = self.profile.hi if self._open.direction == "high" else self.profile.lo
        over = abs(0.5 * (temp + self._prev[1]) - lim)
        self._open.degree_minutes += over * dt

    def _close(self, ts: pd.Timestamp) -> None:
        if self._open is not None:
            self._open.end = ts
            self._open = None

    def _check_long(self, ts: pd.Timestamp, new: list[Alarm]) -> None:
        ev = self._open
        if ev is None or self._long_fired:
            return
        mins = (ts - ev.start).total_seconds() / 60.0
        if mins >= self.config.long_excursion_min:
            self._long_fired = True
            new.append(Alarm(ts, "excursion_long", "critical", f"excursion lasting {mins:.0f} min and still open"))

    def _rate_check(self, ts: pd.Timestamp, temp: float, new: list[Alarm]) -> None:
        if self._prev is None:
            return
        dt_h = (ts - self._prev[0]).total_seconds() / 3600.0
        if not 0 < dt_h <= self.config.max_gap_min / 60.0:
            self._rates.clear()
            return
        self._rates.append(abs(temp - self._prev[1]) / dt_h)
        # one large step is far more often a sensor glitch than physics; require two consecutive
        if len(self._rates) == 2 and min(self._rates) > self.config.rate_alarm_c_per_h and "rate" not in self._fired:
            self._fired.add("rate")
            new.append(Alarm(ts, "rate_of_change", "warning", f"temperature changing {self._rates[-1]:.1f} C/h"))

    def _budget_check(self, ts: pd.Timestamp, new: list[Alarm]) -> None:
        used = self.budget_consumed()
        if used >= 1.0 and "budget_100" not in self._fired:
            self._fired.add("budget_100")
            new.append(Alarm(ts, "budget", "critical", "stability allowance exhausted (>= 100 %)"))
        elif used >= self.config.budget_warn_fraction and "budget_80" not in self._fired:
            self._fired.add("budget_80")
            new.append(Alarm(ts, "budget", "warning", f"stability allowance {used * 100:.0f} % used"))

    # ----------------------------------------------------------- early warning
    def _ew_arm_reset(self) -> None:
        self._ew_armed = False
        self._ew_clear_since = None
        self._ew_hits = 0

    def _early_warning(self, ts: pd.Timestamp, temp: float, is_gap: bool) -> None:
        cfg = self.config
        now = ts.timestamp()
        self._hist.append((now, temp))
        while self._hist and now - self._hist[0][0] > cfg.ew_window_min * 60.0:
            self._hist.popleft()
        if len(self._hist) < cfg.ew_min_points:
            return
        slope = _slope(self._hist)  # C per second
        direction, pred = self._predict(temp, slope)
        in_range_now = self.profile.in_range(temp)
        if self._open is not None or not in_range_now:
            self._ew_hits = 0
            return
        breach = direction is not None
        if breach:
            self._ew_clear_since = None
            self._ew_hits += 1
            if self._ew_armed and self._ew_hits >= cfg.ew_consecutive:
                self._ew_armed = False
                self.warnings.append(EarlyWarning(ts, direction, pred, cfg.ew_horizon_min, temp, slope * 3600.0))
                self.alarms.append(Alarm(ts, "early_warning", "warning",
                                         f"predicted {direction} breach within {cfg.ew_horizon_min:.0f} min "
                                         f"({pred:.1f} C, trend {slope * 3600:+.1f} C/h)"))
        else:
            self._ew_hits = 0
            if not self._ew_armed:
                if self._ew_clear_since is None:
                    self._ew_clear_since = now
                elif now - self._ew_clear_since >= cfg.ew_rearm_min * 60.0:
                    self._ew_armed = True

    def _predict(self, temp: float, slope: float) -> tuple[str | None, float]:
        cfg = self.config
        h = cfg.ew_horizon_min * 60.0
        if cfg.ew_method == "linear":
            pred = temp + slope * h
        else:
            ta = temp + cfg.tau_s * slope  # ambient implied by the heat balance dT/dt = (Ta - T) / tau
            pred = ta + (temp - ta) * math.exp(-h / cfg.tau_s)
        min_rate = cfg.ew_min_rate_c_per_h / 3600.0
        if slope >= min_rate and pred > self.profile.hi:
            return "high", pred
        if slope <= -min_rate and pred < self.profile.lo:
            return "low", pred
        return None, pred


def _slope(hist: deque[tuple[float, float]]) -> float:
    """Theil-Sen slope (median of pairwise slopes, C per second): robust to isolated sensor spikes."""
    pts = list(hist)
    slopes = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dt = pts[j][0] - pts[i][0]
            if dt > 0:
                slopes.append((pts[j][1] - pts[i][1]) / dt)
    if not slopes:
        return 0.0
    slopes.sort()
    m = len(slopes)
    return slopes[m // 2] if m % 2 else 0.5 * (slopes[m // 2 - 1] + slopes[m // 2])


def run_monitor(df: pd.DataFrame, profile: ProductProfile, config: MonitorConfig | None = None) -> Monitor:
    """Feed a whole logger frame (columns ts, temp_c) through a Monitor."""
    mon = Monitor(profile, config or MonitorConfig())
    for ts, temp in zip(df["ts"], df["temp_c"], strict=True):
        mon.update(pd.Timestamp(ts), float(temp))
    mon.finish()
    return mon
