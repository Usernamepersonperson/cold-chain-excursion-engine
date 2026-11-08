import math

import numpy as np
import pandas as pd
import pytest

from ccee.mkt import R_KJ_MOL_K, RunningMKT, mkt, mkt_decimal

DH_R = 83.144 / R_KJ_MOL_K


def test_default_delta_h_over_r_is_10000_kelvin():
    assert DH_R == pytest.approx(10000.0, rel=1e-9)


@pytest.mark.parametrize("t", [-20.0, 2.0, 5.0, 25.0, 40.0])
def test_constant_temperature_returns_itself(t):
    assert mkt([t] * 50) == pytest.approx(t, abs=1e-9)


def test_two_level_analytic_value():
    # half the readings at 20 C, half at 30 C; hand formula
    a = math.exp(-10000 / 293.15)
    b = math.exp(-10000 / 303.15)
    expected = 10000 / -math.log((a + b) / 2) - 273.15
    assert mkt([20, 30] * 10) == pytest.approx(expected, abs=1e-9)
    assert 25.0 < expected < 27.0  # above the arithmetic mean, as MKT must be


def test_mkt_lies_between_mean_and_max():
    rng = np.random.default_rng(1)
    t = rng.uniform(2, 40, 500)
    assert t.mean() <= mkt(t) <= t.max()


def test_hot_spike_raises_mkt_more_than_mean():
    base = [5.0] * 100
    spiked = base[:-1] + [40.0]
    assert mkt(spiked) - 5.0 > np.mean(spiked) - 5.0


def test_matches_40_digit_reference():
    rng = np.random.default_rng(3)
    t = rng.uniform(-25, 45, 2000)
    assert mkt(t) == pytest.approx(mkt_decimal(t.tolist()), abs=1e-10)


def test_other_delta_h_changes_result():
    t = [4.0] * 10 + [30.0] * 2
    assert mkt(t, delta_h_kj_mol=60.0) < mkt(t, delta_h_kj_mol=100.0)


def test_nan_values_ignored_and_empty_is_nan():
    assert mkt([5.0, np.nan, 5.0]) == pytest.approx(5.0)
    assert math.isnan(mkt([]))
    assert math.isnan(mkt([np.nan]))


def test_time_weighting_irregular_constant():
    ts = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:07", "2025-01-01 01:30"], utc=True).to_numpy()
    assert mkt([6.0, 6.0, 6.0], ts) == pytest.approx(6.0, abs=1e-9)


def test_time_weighting_long_interval_dominates():
    ts = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:10", "2025-01-01 10:00"], utc=True).to_numpy()
    weighted = mkt([30.0, 5.0, 5.0], ts)
    unweighted = mkt([30.0, 5.0, 5.0])
    assert weighted < unweighted


def test_gap_excluded_when_max_gap_given():
    ts = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:05", "2025-01-01 20:00", "2025-01-01 20:05"], utc=True).to_numpy()
    with_gap_excluded = mkt([5.0, 5.0, 30.0, 30.0], ts, max_gap_s=1800)
    assert with_gap_excluded == pytest.approx(mkt([5.0, 5.0]) * 0 + _two_level(5.0, 30.0), abs=1e-6)


def _two_level(a, b):
    # intervals: (5,5) 5 min and (30,30) 5 min, equal weight
    ta, tb = math.exp(-10000 / (a + 273.15)), math.exp(-10000 / (b + 273.15))
    return 10000 / -math.log((ta + tb) / 2) - 273.15


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        mkt([1.0, 2.0], [0.0])


def test_running_mkt_matches_batch():
    rng = np.random.default_rng(5)
    t = rng.uniform(3, 12, 120)
    secs = np.arange(120) * 300.0
    r = RunningMKT(max_gap_s=1800.0)
    for s, v in zip(secs, t, strict=True):
        r.update(float(s), float(v))
    assert r.value == pytest.approx(mkt(t, secs, max_gap_s=1800.0), abs=1e-9)
