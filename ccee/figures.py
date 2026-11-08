"""Figures: README result figure, LinkedIn hook image, and the chart embedded in shipment reports."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED, GREY, VIOLET = "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#64748B", "#A78BFA"


def _family() -> list[str]:
    names = {f.name for f in font_manager.fontManager.ttflist}
    return ["Segoe UI" if "Segoe UI" in names else "DejaVu Sans", "DejaVu Sans"]


def dark_style() -> None:
    plt.rcParams.update(
        {
            "font.family": _family(),
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "savefig.facecolor": BG,
        }
    )


def _curve(ax, df: pd.DataFrame, profile, gap_min: float = 30.0, color: str = ACCENT, lw: float = 2.2) -> None:
    ts = df["ts"].reset_index(drop=True)
    y = df["temp_c"].to_numpy(float).copy()
    d = ts.diff().dt.total_seconds().to_numpy() / 60.0
    seg_start = [0] + [i for i in range(1, len(ts)) if d[i] > gap_min]
    seg_start.append(len(ts))
    for a, b in zip(seg_start[:-1], seg_start[1:], strict=True):
        ax.plot(ts[a:b], y[a:b], color=color, lw=lw, solid_capstyle="round")
    ax.axhspan(profile.lo, profile.hi, color=GREEN, alpha=0.14, lw=0)
    ax.axhline(profile.hi, color=GREEN, lw=1, ls="--", alpha=0.7)
    ax.axhline(profile.lo, color=GREEN, lw=1, ls="--", alpha=0.7)


def pick_example(results) -> object | None:
    """A warned, clearly visible excursion with a typical lead time and no logger gap around it."""
    best, best_score = None, 1e9
    lead_all = [x for r in results for x in r.ew_lead_min if x is not None]
    target = float(np.median(lead_all)) if lead_all else 30.0
    for r in results:
        for t, lead in zip(r.truth, r.ew_lead_min, strict=True):
            if lead is None or t.direction != "high" or t.minutes < 90:
                continue
            win_a = t.start - pd.Timedelta(hours=10)
            near_gap = any(g.start < t.end and g.end > win_a for g in r.monitor.gaps)
            spikes = any(f["kind"] == "spike" for f in r.ship.meta.data_faults)
            if near_gap or spikes:
                continue
            score = abs(lead - target) - min(t.minutes, 400) / 100.0
            if score < best_score:
                best, best_score = (r, t, lead), score
    return best


def _annotate_event(ax, r, t, lead: float, small: bool = False) -> None:
    mon = r.monitor
    det = next((e for e in mon.events if e.direction == t.direction and e.start <= t.end and (e.end or e.start) >= t.start), None)
    warn = next((w for w in mon.warnings if w.direction == t.direction and pd.Timedelta(0) <= t.start - w.ts <= pd.Timedelta(minutes=120)), None)
    p = r.ship.profile
    fs = 12 if small else 11
    if det is not None:
        end = det.end or det.start
        ax.axvspan(det.start, end, color=RED, alpha=0.22, lw=0)
    if warn is not None:
        ax.axvline(warn.ts, color=AMBER, lw=2, ls=":")
        ax.plot([warn.ts], [warn.current_c], marker="v", color=AMBER, ms=13, zorder=5, mec=BG, mew=1.5)
        ax.annotate(
            f"early warning\n{lead:.0f} min before breach",
            xy=(warn.ts, warn.current_c), xytext=(-14, 36), textcoords="offset points", ha="right",
            color=AMBER, fontsize=fs, fontweight="bold", arrowprops={"arrowstyle": "-", "color": AMBER, "lw": 1.2},
        )
    if det is not None:
        ax.plot([det.start], [p.hi], marker="o", color=RED, ms=11, zorder=5, mec=BG, mew=1.5)
        ax.annotate(
            f"excursion detected\npeak {det.peak_c:.1f} °C, {det.minutes / 60:.1f} h",
            xy=(det.start, p.hi), xytext=(14, -52), textcoords="offset points", ha="left",
            color=RED, fontsize=fs, fontweight="bold", arrowprops={"arrowstyle": "-", "color": RED, "lw": 1.2},
        )


def _setup_time_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))


def hook_figure(path: str | Path, example, title: str, subtitle: str) -> None:
    """960 x 700 LinkedIn image: temperature, band, excursion, early-warning marker."""
    dark_style()
    r, t, lead = example
    ship = r.ship
    from ccee.evaluate import corrected_logger

    log = corrected_logger(ship)
    a = t.start - pd.Timedelta(hours=9)
    b = (t.end if t.end > t.start else t.start) + pd.Timedelta(hours=2)
    w = log[(log["ts"] >= a) & (log["ts"] <= b)]
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1], left=0.1, right=0.97, top=0.80, bottom=0.12, hspace=0.12)
    ax = fig.add_subplot(gs[0])
    _curve(ax, w, ship.profile, lw=3.0)
    _annotate_event(ax, r, t, lead)
    ax.set_xlim(a, b)
    lo, hi = float(w["temp_c"].min()), float(w["temp_c"].max())
    ax.set_ylim(min(lo, ship.profile.lo) - 1.2, max(hi, ship.profile.hi) + 2.0)
    ax.set_ylabel("payload temperature (°C)", fontsize=12)
    ax.tick_params(labelbottom=False, labelsize=12)
    ax.text(a + (b - a) * 0.01, ship.profile.lo + 0.15, f"allowed range {ship.profile.lo:g} to {ship.profile.hi:g} °C",
            color=GREEN, fontsize=11.5, va="bottom")
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    amb_t = ship.t0 + pd.to_timedelta(np.arange(len(ship.ambient)), unit="min")
    m = (amb_t >= a) & (amb_t <= b)
    ax2.plot(amb_t[m], ship.ambient[m], color=GREY, lw=2.2)
    ax2.set_ylabel("ambient (°C)", fontsize=12)
    ax2.tick_params(labelsize=12)
    _setup_time_axis(ax2)
    ax2.set_xlim(a, b)
    fig.text(0.04, 0.945, title, fontsize=14.5, fontweight="bold", color=TEXT, va="top")
    fig.text(0.04, 0.895, subtitle, fontsize=11.5, color=MUTED, va="top")
    fig.text(0.04, 0.025, "ccee, simulated shipment | ambient: Open-Meteo / ERA5 (CC BY 4.0)", fontsize=10.5, color=MUTED)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main_figure(path: str | Path, example, results, agg: dict) -> None:
    """1600 x 900 README figure: example shipment plus headline distributions."""
    dark_style()
    from ccee.evaluate import corrected_logger

    r, t, lead = example
    ship = r.ship
    log = corrected_logger(ship)
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1.6, 0.55, 1.15], left=0.06, right=0.98, top=0.86, bottom=0.08, hspace=0.42, wspace=0.28
    )
    ax = fig.add_subplot(gs[0, :])
    a = log["ts"].iloc[0]
    b = log["ts"].iloc[-1]
    _curve(ax, log, ship.profile, lw=2.4)
    _annotate_event(ax, r, t, lead)
    ax.set_xlim(a, b)
    ax.set_ylim(ship.profile.lo - 4.5, float(log["temp_c"].max()) + 1.5)
    ax.set_ylabel("payload temperature (°C)", fontsize=11)
    ax.tick_params(labelbottom=False, labelsize=10)
    axa = fig.add_subplot(gs[1, :], sharex=ax)
    amb_t = ship.t0 + pd.to_timedelta(np.arange(len(ship.ambient)), unit="min")
    axa.plot(amb_t, ship.ambient, color=GREY, lw=1.6)
    axa.set_ylabel("ambient (°C)", fontsize=10.5)
    _setup_time_axis(axa)
    axa.tick_params(labelsize=10)
    axa.set_xlim(a, b)

    ax1 = fig.add_subplot(gs[2, 0])
    leads = [x for rr in results for x in rr.ew_lead_min if x is not None]
    ax1.hist(leads, bins=np.arange(0, 80, 5), color=AMBER, edgecolor=BG)
    med = agg["early_warning"]["lead_min_median"]
    if med is not None:
        ax1.axvline(med, color=TEXT, lw=1.5, ls="--")
        ax1.text(med - 1, ax1.get_ylim()[1] * 0.88, f"median {med:.0f} min", color=TEXT, fontsize=11, ha="right")
    ax1.set_xlabel("early-warning lead time before true breach (min)", fontsize=10.5)
    ax1.set_ylabel("excursions", fontsize=10.5)
    ax1.set_title("Early warning lead time", loc="left", fontsize=12, color=TEXT)

    ax2 = fig.add_subplot(gs[2, 1])
    causes = agg["recall_by_cause"]
    names = list(causes)
    vals = [causes[k]["recall"] * 100 for k in names]
    ax2.barh(names, vals, color=GREEN, edgecolor=BG)
    for i, k in enumerate(names):
        ax2.text(vals[i] + 2, i, f"n={causes[k]['n']}", va="center", color=TEXT, fontsize=10)
    ax2.set_xlim(0, 125)
    ax2.set_xlabel("detected (%) by injected cause", fontsize=10.5)
    ax2.set_title("Excursion recall by cause", loc="left", fontsize=12, color=TEXT)
    ax2.invert_yaxis()

    ax3 = fig.add_subplot(gs[2, 2])
    tr = np.array([rr.mkt_truth for rr in results])
    lg = np.array([rr.mkt_logger for rr in results])
    ok = np.isfinite(lg)
    ax3.scatter(tr[ok], lg[ok], s=14, color=VIOLET, alpha=0.8, edgecolors="none")
    lim = [min(tr[ok].min(), lg[ok].min()) - 1, max(tr[ok].max(), lg[ok].max()) + 1]
    ax3.plot(lim, lim, color=GREY, lw=1.2)
    ax3.set_xlabel("MKT from true payload series (°C)", fontsize=10.5)
    ax3.set_ylabel("MKT from logger readings (°C)", fontsize=10.5)
    ax3.set_title(f"MKT, logger vs truth (MAE {agg['mkt']['logger_vs_true_mae_c']:.2f} C)", loc="left", fontsize=12, color=TEXT)
    for a_ in (ax1, ax2, ax3):
        a_.tick_params(labelsize=10)

    fig.text(0.06, 0.965, "Pharma shipment excursion engine: detect, budget, warn", fontsize=20, fontweight="bold", color=TEXT, va="top")
    fig.text(
        0.06, 0.918,
        f"Simulated shipments ({agg['n_shipments']}, shown: {ship.shipment_id} {ship.route} {ship.profile.name}); "
        "ambient: Open-Meteo / ERA5, July 2025 and January 2025 (CC BY 4.0)",
        fontsize=12, color=MUTED, va="top",
    )
    fig.savefig(path, dpi=100)
    plt.close(fig)


def report_chart_png(df: pd.DataFrame, profile, monitor) -> bytes:
    """Light-background chart for the printable shipment report."""
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=110)
    ts = df["ts"]
    ax.axhspan(profile.lo, profile.hi, color="#34D399", alpha=0.18, lw=0)
    ax.plot(ts, df["temp_c"], color="#0369A1", lw=1.4)
    for e in monitor.events:
        ax.axvspan(e.start, e.end or e.start, color="#F87171", alpha=0.35, lw=0)
    for w in monitor.warnings:
        ax.axvline(w.ts, color="#D97706", lw=1, ls=":")
    for g in monitor.gaps:
        ax.axvspan(g.start, g.end, color="#94A3B8", alpha=0.25, lw=0)
    ax.set_ylabel("temperature (°C)")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate()
    ax.set_title("green: allowed range, red: excursion, amber: early warning, grey: data gap", fontsize=8, loc="left")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
