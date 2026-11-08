"""Command line: python -m ccee <command>."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ccee import __version__
from ccee.analysis import analyze_record
from ccee.audit import AuditLog
from ccee.ingest import MqttCollector, read_logger_csv
from ccee.mkt import mkt
from ccee.monitor import MonitorConfig
from ccee.profiles import PROFILES, get_profile
from ccee.report import write_report

DEFAULT_SEED = 2026


def _demo(args: argparse.Namespace) -> int:
    from ccee.evaluate import summarize
    from ccee.figures import hook_figure, main_figure, pick_example
    from ccee.sim.fleet import make_fleet
    from ccee.sim.weather import WINDOWS, WeatherArchive

    root = Path(args.out)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    archives = {w: WeatherArchive(w) for w in WINDOWS}
    fleet = make_fleet(args.n, seed=args.seed, archives=archives)
    print(f"simulated {len(fleet)} shipments ({time.time() - t0:.0f} s)")

    agg, results = summarize(fleet, lambda s: MonitorConfig(tau_s=s.container.tau_s, ew_method="thermal"))
    lin, _ = summarize(fleet, lambda s: MonitorConfig(tau_s=s.container.tau_s, ew_method="linear"), numeric=False)
    unc, _ = summarize(fleet, lambda s: MonitorConfig(tau_s=s.container.tau_s), correct=False, numeric=False)
    horizons = {}
    for h in (15, 30, 60):
        a, _ = summarize(fleet, lambda s, h=h: MonitorConfig(tau_s=s.container.tau_s, ew_horizon_min=h), numeric=False)
        horizons[str(h)] = a["early_warning"]
    sweep = {}
    for iv in (15, 30):
        sub = make_fleet(min(args.n, 100), seed=args.seed + iv, archives=archives, interval_min=iv)
        a, _ = summarize(
            sub, lambda s, iv=iv: MonitorConfig(tau_s=s.container.tau_s, ew_window_min=6.0 * iv), numeric=False
        )
        sweep[f"{iv}min"] = {k: a[k] for k in ("n_shipments", "n_true_excursions", "event_recall", "event_precision", "detections_per_true_excursion")}
        sweep[f"{iv}min"]["early_warning"] = a["early_warning"]
    metrics = {
        "engine_version": __version__,
        "seed": args.seed,
        "truth_definition": "payload out of range for >= 10 consecutive minutes (noise-free simulated payload temperature)",
        "match_tolerance_min": 15,
        "main": agg,
        "early_warning_linear_trend": lin["early_warning"],
        "start_time_error_min_median_abs_without_clock_correction": unc["start_time_error_min_median_abs"],
        "early_warning_by_horizon_min": horizons,
        "logger_interval_sweep": sweep,
        "weather": {
            "source": "Open-Meteo historical weather API (ERA5 reanalysis), CC BY 4.0",
            "windows": {k: list(v) for k, v in WINDOWS.items()},
        },
    }
    (root / "results" / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")

    ex = pick_example(results)
    if ex is not None:
        r, t, lead = ex
        main_figure(root / "docs" / "figure.png", ex, results, agg)
        hook_figure(root / "docs" / "post-figure.png", ex, _hook_title(r, lead), _hook_subtitle(r))
        _write_sample(root, r)
        print(f"example {r.ship.shipment_id}, lead {lead:.0f} min; figures and sample report written")
    print(json.dumps({k: agg[k] for k in ("event_recall", "event_precision", "early_warning", "mkt")}, indent=1))
    print(f"done in {time.time() - t0:.0f} s; metrics in {root / 'results' / 'metrics.json'}")
    return 0


def _write_sample(root: Path, r) -> None:
    from ccee.evaluate import corrected_logger

    df = corrected_logger(r.ship)
    an = analyze_record(
        df, r.ship.profile, MonitorConfig(tau_s=r.ship.container.tau_s),
        {"device_id": f"SIMLOG-{r.ship.shipment_id[-4:]}", "calibration_date": "n/a (simulated)"},
        sync_offsets_s=(r.ship.meta.sync_start_offset_s, r.ship.meta.sync_end_offset_s),
    )
    write_report(an, root / "docs" / "sample-report", r.ship.shipment_id, _sim_header(r.ship), "SIMULATED DATA, not a real shipment.")


def _figures(args: argparse.Namespace) -> int:
    from ccee.evaluate import summarize
    from ccee.figures import hook_figure, main_figure, pick_example
    from ccee.sim.fleet import make_fleet

    root = Path(args.out)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    agg, results = summarize(make_fleet(args.n, seed=args.seed), numeric=False)
    ex = pick_example(results)
    if ex is None:
        print("no warned excursion to show")
        return 1
    r, _t, lead = ex
    main_figure(root / "docs" / "figure.png", ex, results, agg)
    hook_figure(root / "docs" / "post-figure.png", ex, _hook_title(r, lead), _hook_subtitle(r))
    _write_sample(root, r)
    print("figures and sample report written")
    return 0


def _hook_title(r, lead: float) -> str:
    return f"Breach of the {r.ship.profile.lo:g} to {r.ship.profile.hi:g} °C range predicted {lead:.0f} min ahead"


def _hook_subtitle(r) -> str:
    box = r.ship.container.name.replace("_", " ", 1).replace("_", "-")
    return f"Simulated shipment {r.ship.route}, {box}; real hourly weather (Open-Meteo / ERA5)"


def _sim_header(ship) -> list[tuple[str, str]]:
    return [
        ("Shipment ID", ship.shipment_id + " (simulated)"),
        ("Route", ship.route),
        ("Product profile", f"{ship.profile.name} ({ship.profile.lo:g} to {ship.profile.hi:g} C)"),
        ("Container model", ship.container.name),
        ("Logger", f"every {ship.meta.interval_min} min"),
        ("Start (UTC)", f"{ship.t0:%Y-%m-%d %H:%M}"),
    ]


def _simulate(args: argparse.Namespace) -> int:
    from ccee.evaluate import truth_events
    from ccee.sim.fleet import make_fleet

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fleet = make_fleet(args.n, seed=args.seed)
    for s in fleet:
        s.logger.to_csv(out / f"{s.shipment_id}.csv", index=False)
        truth = [{"start": str(t.start), "end": str(t.end), "direction": t.direction, "cause": t.cause} for t in truth_events(s)]
        meta = {"profile": s.profile.name, "route": s.route, "container": s.container.name, "faults": s.faults,
                "sync_start_offset_s": s.meta.sync_start_offset_s, "sync_end_offset_s": s.meta.sync_end_offset_s,
                "truth_excursions": truth}
        (out / f"{s.shipment_id}.truth.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"wrote {len(fleet)} shipments to {out}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    res = read_logger_csv(args.file, tz=args.tz, dayfirst=not args.monthfirst)
    profile = get_profile(args.profile)
    meta = dict(res.metadata)
    if args.device:
        meta["device_id"] = args.device
    sync = (args.sync_start, args.sync_end) if args.sync_start is not None and args.sync_end is not None else None
    df = res.df
    if sync is not None:
        from ccee.sim.logger import correct_clock

        df = correct_clock(df, *sync)
    an = analyze_record(df, profile, MonitorConfig(), meta, rejects=len(res.rejects), sync_offsets_s=sync)
    sid = args.id or Path(args.file).stem
    header = [("Shipment ID", sid), ("Source file", meta.get("source_file", "")), ("Source SHA-256", meta.get("source_sha256", "")),
              ("Device", meta.get("device_id", "not supplied")), ("Product profile", f"{profile.name}")]
    paths = write_report(an, args.out, sid, header, "Imported logger record.", args.reviewer, pdf=args.pdf)
    for k, v in paths.items():
        print(f"{k}: {v}")
    print(f"excursions: {len(an.monitor.events)}  MKT: {an.mkt_c:.2f} C  rejects: {len(res.rejects)}")
    return 1 if an.monitor.events else 0


def _mkt(args: argparse.Namespace) -> int:
    if args.temps:
        print(f"{mkt([float(x) for x in args.temps], delta_h_kj_mol=args.delta_h):.4f}")
        return 0
    res = read_logger_csv(args.file)
    print(f"{mkt(res.df['temp_c'].to_numpy(), res.df['ts'].to_numpy(), args.delta_h, 1800.0):.4f}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    ok, msg = AuditLog.load(args.file).verify()
    print(("OK: " if ok else "FAILED: ") + msg)
    return 0 if ok else 2


def _fetch(args: argparse.Namespace) -> int:
    from ccee.sim import weather as w

    for win in w.WINDOWS:
        for code in w.LOCATIONS:
            df = w.fetch(code, win)
            print(code, win, len(df), "rows")
    return 0


def _mqtt(args: argparse.Namespace) -> int:
    def show(device: str, alarm) -> None:
        print(f"[{alarm.severity}] {device} {alarm.ts:%H:%M} {alarm.rule}: {alarm.message}", flush=True)

    col = MqttCollector(get_profile(args.profile), MonitorConfig(), on_alarm=show)
    col.run(args.broker, args.port, args.topic, args.seconds)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ccee", description="Cold-chain excursion engine")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="simulate, evaluate, and write metrics, figures and a sample report")
    d.add_argument("--n", type=int, default=200)
    d.add_argument("--seed", type=int, default=DEFAULT_SEED)
    d.add_argument("--out", default=".")
    d.set_defaults(fn=_demo)

    g = sub.add_parser("figures", help="regenerate docs/figure.png and docs/post-figure.png only")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g.add_argument("--out", default=".")
    g.set_defaults(fn=_figures)

    s = sub.add_parser("simulate", help="write simulated logger CSVs with ground truth")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--out", default="out/sim")
    s.set_defaults(fn=_simulate)

    a = sub.add_parser("analyze", help="analyze a logger CSV and write a shipment report")
    a.add_argument("file")
    a.add_argument("--profile", default="2-8C", choices=sorted(PROFILES))
    a.add_argument("--out", default="out/report")
    a.add_argument("--id")
    a.add_argument("--device")
    a.add_argument("--tz", default="UTC")
    a.add_argument("--monthfirst", action="store_true", help="interpret ambiguous dates as month-first")
    a.add_argument("--sync-start", type=float, help="logger-minus-reference clock offset (s) read at start")
    a.add_argument("--sync-end", type=float, help="logger-minus-reference clock offset (s) read at download")
    a.add_argument("--reviewer")
    a.add_argument("--pdf", action="store_true")
    a.set_defaults(fn=_analyze)

    m = sub.add_parser("mkt", help="mean kinetic temperature of a CSV or of listed temperatures")
    m.add_argument("file", nargs="?")
    m.add_argument("--temps", nargs="+")
    m.add_argument("--delta-h", type=float, default=83.144)
    m.set_defaults(fn=_mkt)

    v = sub.add_parser("verify-audit", help="verify a hash-chained audit file")
    v.add_argument("file")
    v.set_defaults(fn=_verify)

    f = sub.add_parser("fetch-weather", help="refresh the cached open weather archive files")
    f.set_defaults(fn=_fetch)

    q = sub.add_parser("mqtt", help="subscribe to MQTT telemetry and raise live alarms (needs the mqtt extra)")
    q.add_argument("--broker", default="localhost")
    q.add_argument("--port", type=int, default=1883)
    q.add_argument("--topic", default="ccee/+/telemetry")
    q.add_argument("--profile", default="2-8C", choices=sorted(PROFILES))
    q.add_argument("--seconds", type=float)
    q.set_defaults(fn=_mqtt)

    args = p.parse_args(argv)
    if args.cmd == "mkt" and not (args.file or args.temps):
        p.error("mkt needs a file or --temps")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
