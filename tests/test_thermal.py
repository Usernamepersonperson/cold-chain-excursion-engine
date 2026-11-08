import math

import numpy as np
import pytest

from ccee.sim.thermal import Container, simulate_payload


def run(c, hours, amb, t0, mult=1.0, melt0=0.0, power=True):
    n = int(hours * 60)
    return simulate_payload(c, np.full(n, amb), np.full(n, mult), np.full(n, power), t0, melt0)


def test_no_pcm_follows_exponential_relaxation():
    c = Container("t", 20_000, 0.5)
    T, _ = run(c, 10, 30.0, 5.0)
    tau = c.tau_s
    expected = 30.0 + (5.0 - 30.0) * math.exp(-(10 * 3600 - 60) / tau)
    assert T[-1] == pytest.approx(expected, abs=0.05)


def test_pcm_holds_temperature_while_melting():
    c = Container("t", 20_000, 0.3, pcm_temp_c=5.0, pcm_latent_j=1_000_000)
    T, f = run(c, 20, 25.0, 5.0)
    assert np.allclose(T, 5.0, atol=1e-6)
    assert f[-1] > f[0]


def test_pcm_energy_balance():
    c = Container("t", 20_000, 0.3, pcm_temp_c=5.0, pcm_latent_j=1_000_000)
    hours = 10
    _, f = run(c, hours, 25.0, 5.0)
    expected = 0.3 * 20.0 * (hours * 3600 - 60) / 1_000_000
    assert f[-1] == pytest.approx(expected, rel=1e-3)


def test_pcm_exhaustion_then_warming():
    c = Container("t", 20_000, 0.5, pcm_temp_c=5.0, pcm_latent_j=200_000)
    T, f = run(c, 30, 30.0, 5.0)
    assert f[-1] == pytest.approx(1.0)
    assert T[-1] > 20.0


def test_pcm_clamps_cold_side_too():
    c = Container("t", 20_000, 0.3, pcm_temp_c=5.0, pcm_latent_j=1_000_000)
    T, f = run(c, 5, -10.0, 5.0, melt0=0.5)
    assert np.allclose(T, 5.0, atol=1e-6)
    assert f[-1] < 0.5


def test_active_holds_setpoint_when_powered():
    c = Container("a", 60_000, 0.8, setpoint_c=5.0, p_cool_w=45, p_heat_w=25, gain_w_per_k=40)
    T, _ = run(c, 20, 30.0, 5.0)
    assert abs(T[-1] - 5.0) < 1.0


def test_active_power_loss_drifts_towards_ambient():
    c = Container("a", 60_000, 0.8, setpoint_c=5.0, p_cool_w=45, p_heat_w=25, gain_w_per_k=40)
    T, _ = run(c, 20, 30.0, 5.0, power=False)
    assert T[-1] > 15.0


def test_door_opening_multiplier_speeds_warming():
    c = Container("t", 20_000, 0.3)
    slow, _ = run(c, 1, 30.0, 5.0)
    fast, _ = run(c, 1, 30.0, 5.0, mult=10.0)
    assert fast[-1] > slow[-1] + 3.0


def test_tau_property():
    assert Container("t", 36_000, 0.5).tau_s == 72_000
