import numpy as np
import pandas as pd
import pytest
from conftest import frame

from ccee.monitor import Monitor, MonitorConfig, run_monitor


def run(temps, profile, step=5, **cfg):
    return run_monitor(frame(temps, step_min=step), profile, MonitorConfig(**cfg))


def ramp(rate_per_step=0.1, steps=80):
    return [5.0] * 20 + list(5.0 + rate_per_step * np.arange(1, steps))


def test_in_range_record_has_no_events_or_alarms(p28):
    m = run([5.0] * 100, p28)
    assert not m.events and not m.alarms
    assert m.minutes_in_range == pytest.approx(99 * 5)


def test_single_out_of_range_reading_is_debounced(p28):
    assert not run([5, 5, 9.5, 5, 5, 5], p28).events


def test_two_consecutive_readings_open_and_close_excursion(p28):
    m = run([5, 5, 9, 9.5, 9.2, 5, 5], p28)
    assert len(m.events) == 1
    e = m.events[0]
    assert e.direction == "high" and e.peak_c == 9.5
    assert e.start == pd.Timestamp("2025-07-01T00:10:00Z")
    assert e.end == pd.Timestamp("2025-07-01T00:25:00Z")  # first reading back in range
    assert e.minutes == pytest.approx(15)
    assert not e.open_ended


def test_boundaries_are_in_range(p28):
    assert not run([2.0, 8.0, 2.0, 8.0, 2.0], p28).events


def test_low_excursion_and_direction_switch(p28):
    m = run([5, 1, 1, 9, 9, 5], p28)
    assert [e.direction for e in m.events] == ["low", "high"]
    assert m.events[0].end == pd.Timestamp("2025-07-01T00:15:00Z")


def test_open_ended_excursion_at_end_of_record(p28):
    assert run([5, 5, 9, 9, 9], p28).events[0].open_ended


def test_excursion_alarm_raised_when_debounce_met(p28):
    mon = Monitor(p28)
    frm = frame([5, 9, 9])
    out = [mon.update(t, v) for t, v in zip(frm.ts, frm.temp_c, strict=True)]
    assert out[1] == [] and out[2][0].rule == "excursion" and out[2][0].severity == "critical"


def test_long_excursion_alarm(p28):
    m = run([5, 9, 9] + [9] * 20, p28, long_excursion_min=60)
    assert any(a.rule == "excursion_long" for a in m.alarms)


def test_cumulative_budget_by_band(p28):
    m = run([5] + [10] * 15, p28)  # 15 intervals of 5 min at 10 C; band 8..15 allows 720 min
    i = p28.band_index(10)
    assert m.band_minutes[i] == pytest.approx(75)
    assert m.budget_fractions()[i] == pytest.approx(75 / 720)
    assert m.total_minutes_out() == pytest.approx(75)


def test_unbudgeted_temperature_exhausts_budget(p28):
    m = run([5, 70, 70, 5], p28)
    assert m.unbudgeted_minutes > 0
    assert m.budget_consumed() == float("inf")
    assert any(a.rule == "budget" and a.severity == "critical" for a in m.alarms)


def test_budget_warning_before_exhaustion(p28):
    m = run([5] + [28] * 6, p28)  # band 25..60 allows 30 min; 25 min used is 83 %
    assert any(a.rule == "budget" and a.severity == "warning" for a in m.alarms)


def test_data_gap_flagged_and_alarmed(p28):
    frm = frame([5, 5, 5, 5, 5])
    frm.loc[3:, "ts"] += pd.Timedelta(hours=3)
    m = run_monitor(frm, p28)
    assert len(m.gaps) == 1 and m.gaps[0].minutes > 170
    assert any(a.rule == "data_gap" for a in m.alarms)


def test_gap_is_not_counted_as_observed_time(p28):
    frm = frame([5, 5, 5, 5])
    frm.loc[2:, "ts"] += pd.Timedelta(hours=5)
    assert run_monitor(frm, p28).minutes_observed == pytest.approx(10)


def test_excursion_closes_conservatively_at_return_after_gap(p28):
    frm = frame([5, 9, 9, 5, 5])
    frm.loc[3:, "ts"] += pd.Timedelta(hours=4)
    m = run_monitor(frm, p28)
    assert m.events[0].end == frm.ts.iloc[3]
    assert m.events[0].spans_gap


def test_onset_after_gap_marked_unobserved(p28):
    frm = frame([5, 5, 9, 9, 9])
    frm.loc[2:, "ts"] += pd.Timedelta(hours=4)
    assert run_monitor(frm, p28).events[0].onset_unobserved


def test_non_monotonic_and_nan_rows_are_ignored(p28):
    mon = Monitor(p28)
    t0 = pd.Timestamp("2025-07-01T00:00:00Z")
    mon.update(t0, 5.0)
    mon.update(t0 - pd.Timedelta(minutes=5), 50.0)
    mon.update(t0, 50.0)
    mon.update(t0 + pd.Timedelta(minutes=5), float("nan"))
    mon.finish()
    assert not mon.events and mon.n_samples == 1


def test_early_warning_fires_on_ramp_before_breach(p28):
    m = run(ramp(), p28, ew_method="linear")  # 1.2 C/h
    assert m.warnings and m.events
    lead = (m.events[0].start - m.warnings[0].ts).total_seconds() / 60
    assert 10 < lead <= 60


def test_early_warning_thermal_method_also_fires(p28):
    assert run(ramp(), p28, ew_method="thermal", tau_s=20 * 3600).warnings


def test_no_early_warning_on_noisy_in_range_record(p28):
    rng = np.random.default_rng(1)
    assert not run(5 + rng.normal(0, 0.15, 400), p28).warnings


def test_isolated_spike_does_not_trigger_early_warning(p28):
    temps = [5.0] * 30
    temps[15] = 14.0
    assert not run(temps, p28).warnings


def test_early_warning_disabled(p28):
    assert not run(ramp(), p28, ew_enabled=False).warnings


def test_warning_direction_low(p28):
    m = run([5.0] * 20 + list(5.0 - 0.1 * np.arange(1, 60)), p28, ew_method="linear")
    assert m.warnings and m.warnings[0].direction == "low"


def test_rate_of_change_alarm_needs_two_consecutive_fast_steps(p28):
    fast = run([5, 5, 5, 5.9, 6.8, 7.7, 7.7], p28, rate_alarm_c_per_h=8.0)
    assert any(a.rule == "rate_of_change" for a in fast.alarms)
    single = run([5, 5, 5, 6.5, 6.5, 6.5], p28, rate_alarm_c_per_h=8.0)
    assert not any(a.rule == "rate_of_change" for a in single.alarms)


def test_running_mkt_available(p28):
    assert run([5.0] * 30, p28).mkt_c == pytest.approx(5.0, abs=1e-6)
