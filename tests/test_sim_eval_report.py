import json

import numpy as np
import pandas as pd
import pytest

from ccee.analysis import analyze_record
from ccee.cli import main
from ccee.evaluate import analyze_shipment, corrected_logger, summarize, truth_events
from ccee.ingest import to_csv_text
from ccee.monitor import MonitorConfig
from ccee.report import render, write_report
from ccee.sim.fleet import make_fleet
from ccee.sim.logger import correct_clock
from ccee.sim.weather import WeatherArchive


@pytest.fixture(scope="module")
def fleet():
    arch = {w: WeatherArchive(w, download=False) for w in ("summer", "winter")}
    return make_fleet(24, seed=11, archives=arch)


def test_weather_cache_is_real_and_plausible():
    w = WeatherArchive("winter", download=False)
    t = pd.Timestamp("2025-01-15T12:00:00Z").timestamp()
    assert -15 < float(w.air("BSL", np.array([t]))[0]) < 15
    assert float(w.air("SIN", np.array([t]))[0]) > 20
    assert float(w.solar("SIN", np.array([t]))[0]) >= 0


def test_fleet_is_deterministic_per_seed():
    arch = {w: WeatherArchive(w, download=False) for w in ("summer", "winter")}
    a = make_fleet(3, seed=5, archives=arch)
    b = make_fleet(3, seed=5, archives=arch)
    assert all(np.array_equal(x.truth, y.truth) for x, y in zip(a, b, strict=True))
    assert all(x.logger.equals(y.logger) for x, y in zip(a, b, strict=True))


def test_shipments_have_required_features(fleet):
    assert any(s.meta.data_faults for s in fleet)
    assert any(s.logger["lat"].notna().any() for s in fleet)
    assert all(np.isfinite(s.truth).all() for s in fleet)
    assert all(s.duration_h > 20 for s in fleet)


def test_logger_clock_correction_restores_true_time(fleet):
    s = max(fleet, key=lambda x: abs(x.meta.drift_ppm))
    raw = s.logger
    fixed = correct_clock(raw, s.meta.sync_start_offset_s, s.meta.sync_end_offset_s)
    shift_raw = abs((raw["ts"].iloc[-1] - s.t0).total_seconds() - (len(s.truth) - 1) * 60)
    shift_fix = abs((fixed["ts"].iloc[-1] - s.t0).total_seconds() - (len(s.truth) - 1) * 60)
    assert shift_fix <= shift_raw + 5
    assert shift_fix < 600


def test_correct_clock_empty_frame():
    empty = pd.DataFrame({"ts": pd.to_datetime([], utc=True), "temp_c": []})
    assert correct_clock(empty, 1, 2).empty


def test_truth_events_follow_definition(fleet):
    for s in fleet:
        for t in truth_events(s):
            assert t.minutes >= 10
            seg = s.truth_series()[t.start : t.end]
            assert ((seg > s.profile.hi) | (seg < s.profile.lo)).all()


def test_engine_finds_most_truth_events_in_small_fleet(fleet):
    agg, results = summarize(fleet)
    assert agg["n_shipments"] == 24
    if agg["n_true_excursions"]:
        assert agg["event_recall"] >= 0.8
    assert agg["mkt"]["float_vs_40digit_max_abs_c"] < 1e-9
    assert agg["mkt"]["logger_vs_true_mae_c"] < 1.0
    assert all(len(r.ew_lead_min) == len(r.truth) for r in results)


def test_analyze_shipment_reports_nonnegative_leads(fleet):
    for s in fleet:
        r = analyze_shipment(s)
        assert all(x is None or x >= 0 for x in r.ew_lead_min)


def test_analysis_audit_and_integrity(fleet):
    s = fleet[0]
    an = analyze_record(corrected_logger(s), s.profile, MonitorConfig(tau_s=s.container.tau_s), {"device_id": "X"}, sync_offsets_s=(0.0, 1.0))
    ok, _ = an.audit.verify()
    assert ok and an.audit.entries[0]["action"] == "record_loaded"
    assert {"analysis_run", "mkt_computed"} <= {e["action"] for e in an.audit.entries}


def make_analysis(fleet):
    s = next((x for x in fleet if truth_events(x)), fleet[0])
    an = analyze_record(corrected_logger(s), s.profile, MonitorConfig(tau_s=s.container.tau_s), {"device_id": "<b>X</b>"})
    return s, an


def test_report_html_and_markdown(fleet):
    s, an = make_analysis(fleet)
    html, md = render(an, s.shipment_id, [("Route", s.route), ("Note", "<script>x</script>")], "SIMULATED")
    assert "no claim of regulatory compliance" in html and "no claim of regulatory compliance" in md
    assert "&lt;script&gt;" in html and "<script>x" not in html  # autoescape on HTML
    assert "&amp;" not in md
    assert "Chain verification: **" in md and "ALCOA" in md
    assert "data:image/png;base64," in html


def test_write_report_files_and_audit_verifies(fleet, tmp_path, capsys):
    s, an = make_analysis(fleet)
    paths = write_report(an, tmp_path, s.shipment_id, [("Route", s.route)], "SIMULATED", reviewer="Q. Reviewer")
    assert paths["html"].exists() and paths["md"].exists()
    assert main(["verify-audit", str(paths["audit"])]) == 0
    assert "OK" in capsys.readouterr().out
    lines = paths["audit"].read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"actor":"ccee"', '"actor":"someone"')
    paths["audit"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert main(["verify-audit", str(paths["audit"])]) == 2


def test_cli_analyze_end_to_end(fleet, tmp_path, capsys):
    s = next(x for x in fleet if truth_events(x))
    csv = tmp_path / "logger.csv"
    csv.write_text(to_csv_text(corrected_logger(s)), encoding="utf-8")
    code = main(["analyze", str(csv), "--profile", s.profile.name, "--out", str(tmp_path / "rep"), "--id", "T1"])
    out = capsys.readouterr().out
    assert code == 1 and "excursions:" in out
    assert (tmp_path / "rep" / "T1.html").exists()


def test_cli_mkt_values(capsys):
    assert main(["mkt", "--temps", "20", "20", "20"]) == 0
    assert float(capsys.readouterr().out) == pytest.approx(20.0, abs=1e-3)


def test_cli_simulate_writes_truth(tmp_path):
    arch_ok = WeatherArchive("summer", download=False)
    assert arch_ok is not None
    assert main(["simulate", "--n", "2", "--seed", "3", "--out", str(tmp_path)]) == 0
    meta = json.loads((tmp_path / "SIM-0000.truth.json").read_text(encoding="utf-8"))
    assert "truth_excursions" in meta and (tmp_path / "SIM-0000.csv").exists()
