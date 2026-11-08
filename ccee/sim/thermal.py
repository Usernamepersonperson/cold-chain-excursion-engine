"""Lumped thermal model of a shipping container or parcel.

Payload node (heat capacity C) exchanges heat with ambient through insulation conductance UA. An optional
phase-change pack absorbs heat at its melting temperature: while the payload sits at that temperature the
heat goes into melting or freezing the pack (latent capacity L) instead of changing temperature. Active
units add a proportional heating/cooling power around a setpoint and can lose power.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Container:
    name: str
    c_j_per_k: float
    ua_w_per_k: float
    pcm_temp_c: float | None = None
    pcm_latent_j: float = 0.0
    setpoint_c: float | None = None
    p_cool_w: float = 0.0
    p_heat_w: float = 0.0
    gain_w_per_k: float = 0.0

    @property
    def active(self) -> bool:
        return self.setpoint_c is not None

    @property
    def tau_s(self) -> float:
        return self.c_j_per_k / self.ua_w_per_k


CONTAINERS: dict[str, Container] = {
    "passive_2_8": Container("passive_2_8", 25_000, 0.35, pcm_temp_c=5.0, pcm_latent_j=1_200_000),
    "active_2_8": Container(
        "active_2_8", 60_000, 0.8, setpoint_c=5.0, p_cool_w=45.0, p_heat_w=25.0, gain_w_per_k=40.0
    ),
    "passive_15_25": Container("passive_15_25", 25_000, 0.45, pcm_temp_c=20.0, pcm_latent_j=500_000),
    "passive_frozen": Container("passive_frozen", 40_000, 0.40, pcm_temp_c=-21.0, pcm_latent_j=5_000_000),
}


def simulate_payload(
    c: Container,
    t_amb: np.ndarray,
    ua_mult: np.ndarray,
    power_on: np.ndarray,
    t0_c: float,
    melt0: float = 0.0,
    dt: float = 60.0,
    ua_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate payload temperature on a fixed step. Returns (temperature, melt_fraction) per step."""
    n = len(t_amb)
    out = [0.0] * n
    melt_out = [0.0] * n
    amb = t_amb.tolist()
    mult = ua_mult.tolist()
    pw = power_on.tolist()
    cap = c.c_j_per_k
    ua = c.ua_w_per_k * ua_scale
    tp = c.pcm_temp_c
    lat = c.pcm_latent_j
    has_pcm = tp is not None and lat > 0
    temp = float(t0_c)
    f = float(melt0)
    for i in range(n):
        out[i] = temp
        melt_out[i] = f
        q = ua * mult[i] * (amb[i] - temp)
        if c.active and pw[i]:
            pc = c.gain_w_per_k * (temp - c.setpoint_c)
            pc = max(-c.p_heat_w, min(c.p_cool_w, pc))
            q -= pc
        e = q * dt
        if not has_pcm:
            temp += e / cap
            continue
        if abs(temp - tp) < 1e-12:
            if e > 0 and f < 1.0:
                f += e / lat
                if f > 1.0:
                    temp = tp + (f - 1.0) * lat / cap
                    f = 1.0
            elif e < 0 and f > 0.0:
                f += e / lat
                if f < 0.0:
                    temp = tp + f * lat / cap
                    f = 0.0
            else:
                temp = tp + e / cap
        else:
            new = temp + e / cap
            if temp < tp <= new and f < 1.0:
                f += (new - tp) * cap / lat
                temp = tp
                if f > 1.0:
                    temp = tp + (f - 1.0) * lat / cap
                    f = 1.0
            elif temp > tp >= new and f > 0.0:
                f -= (tp - new) * cap / lat
                temp = tp
                if f < 0.0:
                    temp = tp + f * lat / cap
                    f = 0.0
            else:
                temp = new
    return np.asarray(out), np.asarray(melt_out)
