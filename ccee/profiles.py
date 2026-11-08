"""Product temperature profiles with illustrative stability budgets.

The numbers in the built-in profiles are ILLUSTRATIVE defaults for simulation and demos. Real
allowances come from a product's own stability data and approved shipping specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Band:
    """Temperature band outside the labelled range with a cumulative time allowance. lo inclusive, hi exclusive."""

    lo: float
    hi: float
    max_minutes: float

    def contains(self, temp: float) -> bool:
        return self.lo <= temp < self.hi


@dataclass(frozen=True)
class ProductProfile:
    name: str
    lo: float
    hi: float
    bands: tuple[Band, ...] = field(default_factory=tuple)
    mkt_limit_c: float | None = None
    delta_h_kj_mol: float = 83.144

    def __post_init__(self) -> None:
        if self.lo >= self.hi:
            raise ValueError("profile lo must be below hi")
        ordered = sorted(self.bands, key=lambda b: b.lo)
        for a, b in zip(ordered, ordered[1:], strict=False):
            if a.hi > b.lo:
                raise ValueError(f"overlapping budget bands in profile {self.name}")
        for b in ordered:
            if b.lo < self.hi and b.hi > self.lo:
                raise ValueError(f"budget band overlaps the labelled range in profile {self.name}")

    def in_range(self, temp: float) -> bool:
        return self.lo <= temp <= self.hi

    def direction(self, temp: float) -> str | None:
        if temp > self.hi:
            return "high"
        if temp < self.lo:
            return "low"
        return None

    def band_index(self, temp: float) -> int | None:
        for i, b in enumerate(self.bands):
            if b.contains(temp):
                return i
        return None


PROFILES: dict[str, ProductProfile] = {
    "2-8C": ProductProfile(
        "2-8C", 2.0, 8.0,
        bands=(Band(-60, 0, 15), Band(0, 2, 120), Band(8, 15, 720), Band(15, 25, 240), Band(25, 60, 30)),
        mkt_limit_c=8.0,
    ),
    "15-25C": ProductProfile(
        "15-25C", 15.0, 25.0,
        bands=(Band(-60, 0, 30), Band(0, 10, 120), Band(10, 15, 720), Band(25, 30, 1440), Band(30, 60, 240)),
        mkt_limit_c=25.0,
    ),
    "frozen": ProductProfile(
        "frozen", -25.0, -15.0,
        bands=(Band(-15, -10, 240), Band(-10, 0, 60), Band(0, 60, 10)),
        mkt_limit_c=-15.0,
    ),
}


def get_profile(name: str) -> ProductProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(PROFILES)}") from exc
