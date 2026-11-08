"""Hourly ambient air temperature and shortwave radiation from the Open-Meteo historical archive.

Open-Meteo historical weather API (https://open-meteo.com/en/docs/historical-weather-api), data licence
CC BY 4.0; underlying reanalysis ERA5 (Copernicus Climate Change Service / ECMWF). Files are cached under
data/weather so the demo runs offline.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://archive-api.open-meteo.com/v1/archive"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "weather"

WINDOWS = {
    "summer": ("2025-07-01", "2025-07-31"),
    "winter": ("2025-01-01", "2025-01-31"),
}


@dataclass(frozen=True)
class Location:
    code: str
    name: str
    lat: float
    lon: float


LOCATIONS = {
    loc.code: loc
    for loc in [
        Location("BSL", "Basel", 47.59, 7.53),
        Location("FRA", "Frankfurt", 50.03, 8.57),
        Location("SIN", "Singapore", 1.36, 103.99),
        Location("AMS", "Amsterdam", 52.31, 4.76),
        Location("JFK", "New York", 40.64, -73.78),
        Location("ZRH", "Zurich", 47.46, 8.55),
        Location("DXB", "Dubai", 25.25, 55.36),
        Location("NRT", "Tokyo", 35.77, 140.39),
        Location("GRU", "Sao Paulo", -23.43, -46.47),
    ]
}


def cache_name(code: str, window: str) -> str:
    return f"{code}_{WINDOWS[window][0]}_{WINDOWS[window][1]}.csv"


def fetch(code: str, window: str, data_dir: Path | None = None, timeout: float = 60.0) -> pd.DataFrame:
    """Download one location/window from the archive and cache it as CSV."""
    loc = LOCATIONS[code]
    start, end = WINDOWS[window]
    q = urllib.parse.urlencode(
        {
            "latitude": loc.lat,
            "longitude": loc.lon,
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m,shortwave_radiation",
            "timezone": "GMT",
        }
    )
    with urllib.request.urlopen(f"{API}?{q}", timeout=timeout) as resp:  # fixed https endpoint
        payload = json.loads(resp.read().decode("utf-8"))
    h = payload["hourly"]
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(h["time"], utc=True),
            "temp_c": h["temperature_2m"],
            "sw_wm2": h["shortwave_radiation"],
        }
    )
    path = (data_dir or DATA_DIR) / cache_name(code, window)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load(code: str, window: str, data_dir: Path | None = None, download: bool = True) -> pd.DataFrame:
    path = (data_dir or DATA_DIR) / cache_name(code, window)
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        return fetch(code, window, data_dir)
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


class WeatherArchive:
    """Interpolating lookup of hourly series for a given window."""

    def __init__(self, window: str, codes=None, data_dir: Path | None = None, download: bool = True):
        self.window = window
        self.start = pd.Timestamp(WINDOWS[window][0], tz="UTC")
        self.end = pd.Timestamp(WINDOWS[window][1], tz="UTC") + pd.Timedelta(days=1)
        self._t: dict[str, np.ndarray] = {}
        self._air: dict[str, np.ndarray] = {}
        self._sw: dict[str, np.ndarray] = {}
        for code in codes or LOCATIONS:
            df = load(code, window, data_dir, download)
            self._t[code] = df["time"].astype("datetime64[ns, UTC]").astype("int64").to_numpy() / 1e9
            self._air[code] = df["temp_c"].to_numpy(float)
            self._sw[code] = df["sw_wm2"].to_numpy(float)

    def air(self, code: str, epoch_s: np.ndarray) -> np.ndarray:
        return np.interp(epoch_s, self._t[code], self._air[code])

    def solar(self, code: str, epoch_s: np.ndarray) -> np.ndarray:
        return np.interp(epoch_s, self._t[code], self._sw[code])
