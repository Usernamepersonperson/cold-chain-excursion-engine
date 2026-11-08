"""Route itineraries: ground handling, tarmac, flight, customs dwell; ambient per segment kind.

Ambient assumptions (simulated, documented in the README):
  warehouse  controlled room, about 20 C
  truck      un-conditioned insulated box truck: 60 % outdoor air (+ solar) and 40 % cabin 20 C
  tarmac     outdoor air + 0.015 C per W/m2 of shortwave radiation (sun-heated apron)
  hold       aircraft cargo hold, constant per shipment around 15 C
  dock       customs / transfer dock: 70 % outdoor air, 30 % 21 C
  cold       unheated or refrigerated area at a fixed temperature (injected fault)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ccee.sim.weather import LOCATIONS, WeatherArchive

ROUTES = {
    "BSL-FRA-SIN": ["BSL", "FRA", "SIN"],
    "BSL-AMS-JFK": ["BSL", "AMS", "JFK"],
    "BSL-ZRH-DXB": ["BSL", "ZRH", "DXB"],
    "BSL-FRA-NRT": ["BSL", "FRA", "NRT"],
    "BSL-FRA-GRU": ["BSL", "FRA", "GRU"],
}

SOLAR_GAIN_C_PER_WM2 = 0.015


@dataclass
class Segment:
    kind: str
    loc: str
    minutes: int
    loc_to: str | None = None
    cold_c: float | None = None
    start: int = 0


def haversine_km(a: str, b: str) -> float:
    la, lb = LOCATIONS[a], LOCATIONS[b]
    p1, p2 = math.radians(la.lat), math.radians(lb.lat)
    dphi = p2 - p1
    dl = math.radians(lb.lon - la.lon)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def flight_minutes(a: str, b: str) -> int:
    return int(haversine_km(a, b) / 800.0 * 60 + 45)


def build_itinerary(stops: list[str], rng: np.random.Generator) -> list[Segment]:
    a, b, c = stops
    segs = [
        Segment("warehouse", a, int(rng.integers(120, 240))),
        Segment("truck", a, int(rng.integers(60, 90))),
        Segment("tarmac", a, int(rng.integers(45, 120))),
        Segment("hold", a, flight_minutes(a, b), loc_to=b),
        Segment("tarmac", b, int(rng.integers(60, 180))),
    ]
    if rng.random() < 0.4:
        segs.append(Segment("dock", b, int(rng.integers(120, 600))))
    segs += [
        Segment("hold", b, flight_minutes(b, c), loc_to=c),
        Segment("tarmac", c, int(rng.integers(30, 90))),
        Segment("truck", c, int(rng.integers(60, 120))),
        Segment("dock", c, int(rng.integers(120, 480))),
        Segment("warehouse", c, 120),
    ]
    layout(segs)
    return segs


def layout(segs: list[Segment]) -> int:
    t = 0
    for s in segs:
        s.start = t
        t += s.minutes
    return t


def ambient_series(segs: list[Segment], t0_epoch: float, archive: WeatherArchive, hold_c: float, room_c: float) -> np.ndarray:
    """Per-minute effective ambient temperature around the container."""
    total = sum(s.minutes for s in segs)
    out = np.empty(total)
    for s in segs:
        epoch = t0_epoch + (s.start + np.arange(s.minutes)) * 60.0
        if s.kind == "warehouse":
            out[s.start : s.start + s.minutes] = room_c
            continue
        if s.kind == "hold":
            out[s.start : s.start + s.minutes] = hold_c
            continue
        if s.kind == "cold":
            out[s.start : s.start + s.minutes] = s.cold_c
            continue
        air = archive.air(s.loc, epoch)
        sw = archive.solar(s.loc, epoch)
        if s.kind == "tarmac":
            val = air + SOLAR_GAIN_C_PER_WM2 * sw
        elif s.kind == "truck":
            val = 0.6 * (air + 0.5 * SOLAR_GAIN_C_PER_WM2 * sw) + 0.4 * 20.0
        elif s.kind == "dock":
            val = 0.7 * air + 0.3 * 21.0
        else:
            raise ValueError(f"unknown segment kind {s.kind}")
        out[s.start : s.start + s.minutes] = val
    return out


def position_series(segs: list[Segment]) -> tuple[np.ndarray, np.ndarray]:
    """Per-minute latitude/longitude (straight-line interpolation over flights)."""
    total = sum(s.minutes for s in segs)
    lat = np.empty(total)
    lon = np.empty(total)
    for s in segs:
        a = LOCATIONS[s.loc]
        if s.kind == "hold" and s.loc_to:
            b = LOCATIONS[s.loc_to]
            f = np.linspace(0, 1, s.minutes)
            lat[s.start : s.start + s.minutes] = a.lat + f * (b.lat - a.lat)
            lon[s.start : s.start + s.minutes] = a.lon + f * (b.lon - a.lon)
        else:
            lat[s.start : s.start + s.minutes] = a.lat
            lon[s.start : s.start + s.minutes] = a.lon
    return lat, lon
