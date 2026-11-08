"""Ingest: CSV logger exports and MQTT/JSON telemetry.

Nothing is dropped silently: rows that cannot be parsed are returned in `rejects` with line number and reason.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ccee.audit import sha256_file
from ccee.monitor import Alarm, Monitor, MonitorConfig
from ccee.profiles import ProductProfile

TIME_NAMES = ("timestamp", "datetime", "date/time", "date time", "time", "ts", "zeit", "datum")
TEMP_NAMES = ("temp_c", "temperature_c", "temperature", "temp", "value", "temperatur", "t_c")
LAT_NAMES = ("lat", "latitude")
LON_NAMES = ("lon", "lng", "longitude")


@dataclass
class IngestResult:
    df: pd.DataFrame
    rejects: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _norm(h: str) -> str:
    return re.sub(r"\s*[\(\[].*?[\)\]]", "", h).strip().lower()


def _find(headers: list[str], names: tuple[str, ...]) -> int | None:
    norm = [_norm(h) for h in headers]
    for name in names:
        if name in norm:
            return norm.index(name)
    for i, h in enumerate(norm):
        if any(n in h for n in names if len(n) > 3):
            return i
    return None


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("cannot decode file")  # pragma: no cover (latin-1 accepts every byte)


def read_logger_csv(
    path: str | Path,
    tz: str = "UTC",
    dayfirst: bool = True,
    unit: str | None = None,
) -> IngestResult:
    """Read a data-logger CSV export.

    Handles metadata preamble lines, comma/semicolon/tab delimiters, decimal commas, ISO or day-first dates,
    and Fahrenheit (auto-detected from a header such as "Temperature (F)" or forced with unit="F").
    """
    path = Path(path)
    text = _read_text(path)
    lines = text.splitlines()
    meta: dict = {"source_file": path.name, "source_sha256": sha256_file(path)}

    delim = _sniff_delimiter(lines)
    header_idx = None
    for i, ln in enumerate(lines[:60]):
        cells = next(csv.reader([ln], delimiter=delim), [])
        if len(cells) >= 2 and _find(cells, TIME_NAMES) is not None and _find(cells, TEMP_NAMES) is not None:
            header_idx = i
            break
        if ln.strip():
            kv = re.match(r"^\s*([^,;\t:=]+?)\s*[,;\t:=]\s*(.+?)\s*$", ln)
            if kv:
                meta[_meta_key(kv.group(1))] = kv.group(2)
    if header_idx is None:
        raise ValueError("no header row with a time column and a temperature column found")

    headers = next(csv.reader([lines[header_idx]], delimiter=delim))
    ti, vi = _find(headers, TIME_NAMES), _find(headers, TEMP_NAMES)
    li, oi = _find(headers, LAT_NAMES), _find(headers, LON_NAMES)
    if unit is None:
        unit = "F" if re.search(r"(\bf\b|°f|fahrenheit|_f$)", headers[vi].lower()) else "C"
    meta["unit_in_file"] = unit

    ok_ts, ok_t, ok_la, ok_lo, rejects = [], [], [], [], []
    for n, row in enumerate(csv.reader(lines[header_idx + 1 :], delimiter=delim), start=header_idx + 2):
        if not row or not any(c.strip() for c in row):
            continue
        try:
            ts = _parse_time(row[ti], tz, dayfirst)
            val = _parse_float(row[vi])
            if unit == "F":
                val = (val - 32.0) * 5.0 / 9.0
            la = _parse_float(row[li]) if li is not None and li < len(row) and row[li].strip() else np.nan
            lo = _parse_float(row[oi]) if oi is not None and oi < len(row) and row[oi].strip() else np.nan
        except (ValueError, IndexError) as exc:
            rejects.append({"line": n, "reason": str(exc), "raw": delim.join(row)[:120]})
            continue
        ok_ts.append(ts)
        ok_t.append(val)
        ok_la.append(la)
        ok_lo.append(lo)
    df = pd.DataFrame({"ts": pd.to_datetime(ok_ts, utc=True), "temp_c": ok_t, "lat": ok_la, "lon": ok_lo})
    return IngestResult(df, rejects, meta)


def _meta_key(k: str) -> str:
    k = re.sub(r"[^a-z0-9]+", "_", k.strip().lower()).strip("_")
    return {"serial_number": "device_id", "serial": "device_id", "device": "device_id", "logger_id": "device_id",
            "calibration": "calibration_date", "calibrated": "calibration_date"}.get(k, k)


def _sniff_delimiter(lines: list[str]) -> str:
    sample = [ln for ln in lines[:40] if ln.strip()]
    best, best_score = ",", -1
    for d in (",", ";", "\t"):
        counts = [ln.count(d) for ln in sample]
        score = sum(1 for c in counts if c >= 1 and c == max(counts))
        if max(counts, default=0) >= 1 and score > best_score:
            best, best_score = d, score
    return best


def _parse_float(s: str) -> float:
    s = s.strip().replace(" ", "")
    if not s:
        raise ValueError("empty value")
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError as exc:
        raise ValueError(f"not a number: {s!r}") from exc
    if not np.isfinite(v):
        raise ValueError(f"not finite: {s!r}")
    return v


def _parse_time(s: str, tz: str, dayfirst: bool) -> pd.Timestamp:
    s = s.strip()
    if not s:
        raise ValueError("empty timestamp")
    try:
        ts = pd.Timestamp(s) if re.match(r"^\d{4}-\d{2}-\d{2}", s) else pd.to_datetime(s, dayfirst=dayfirst)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"unparseable timestamp: {s!r}") from exc
    if pd.isna(ts):
        raise ValueError(f"unparseable timestamp: {s!r}")
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert("UTC")


# ---------------------------------------------------------------- MQTT / JSON
def _epoch_or_iso(v) -> pd.Timestamp:
    if isinstance(v, bool):
        raise ValueError("bad timestamp")
    if isinstance(v, (int, float)):
        unit = "ms" if v > 1e11 else "s"
        return pd.Timestamp(v, unit=unit, tz="UTC")
    ts = pd.Timestamp(str(v))
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def parse_mqtt_payload(payload: bytes | str, topic: str = "") -> list[dict]:
    """Parse one MQTT message into readings: dicts with device, ts, temp_c, lat, lon.

    Accepts an object or an array of objects. Keys: ts|time|timestamp, temp_c|temperature|t, lat, lon,
    device|device_id|id. When no device key is present the second-to-last topic level is used
    (e.g. ccee/<device>/telemetry). Raises ValueError for malformed input.
    """
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    items = obj if isinstance(obj, list) else [obj]
    parts = [p for p in topic.split("/") if p]
    topic_dev = parts[-2] if len(parts) >= 2 else (parts[0] if parts else "unknown")
    out = []
    for it in items:
        if not isinstance(it, dict):
            raise ValueError("reading must be a JSON object")
        raw_t = next((it[k] for k in ("ts", "time", "timestamp") if k in it), None)
        raw_v = next((it[k] for k in ("temp_c", "temperature", "t") if k in it), None)
        if raw_t is None or raw_v is None:
            raise ValueError("reading needs a timestamp and a temperature")
        try:
            val = float(raw_v)
            ts = _epoch_or_iso(raw_t)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bad reading: {exc}") from exc
        if not np.isfinite(val):
            raise ValueError("non-finite temperature")
        out.append(
            {
                "device": str(next((it[k] for k in ("device", "device_id", "id") if k in it), topic_dev)),
                "ts": ts,
                "temp_c": val,
                "lat": float(it.get("lat", np.nan)),
                "lon": float(it.get("lon", np.nan)),
            }
        )
    return out


class MqttCollector:
    """Routes parsed readings to one Monitor per device and keeps the raw frames."""

    def __init__(self, profile: ProductProfile, config: MonitorConfig | None = None, on_alarm=None):
        self.profile = profile
        self.config = config or MonitorConfig()
        self.monitors: dict[str, Monitor] = {}
        self.rows: dict[str, list[dict]] = {}
        self.errors: list[str] = []
        self.on_alarm = on_alarm

    def handle(self, topic: str, payload: bytes | str) -> list[Alarm]:
        try:
            readings = parse_mqtt_payload(payload, topic)
        except ValueError as exc:
            self.errors.append(f"{topic}: {exc}")
            return []
        alarms: list[Alarm] = []
        for r in readings:
            mon = self.monitors.setdefault(r["device"], Monitor(self.profile, self.config))
            self.rows.setdefault(r["device"], []).append(r)
            new = mon.update(r["ts"], r["temp_c"])
            for a in new:
                if self.on_alarm:
                    self.on_alarm(r["device"], a)
            alarms.extend(new)
        return alarms

    def frame(self, device: str) -> pd.DataFrame:
        return pd.DataFrame(self.rows.get(device, []))

    def run(self, broker: str, port: int = 1883, topic: str = "ccee/+/telemetry", seconds: float | None = None) -> None:
        """Subscribe with paho-mqtt (optional dependency: pip install '.[mqtt]') and process until stopped."""
        import time

        import paho.mqtt.client as mqtt

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_message = lambda _c, _u, msg: self.handle(msg.topic, msg.payload)
        client.on_connect = lambda c, *_a: c.subscribe(topic)
        client.connect(broker, port)
        client.loop_start()
        try:
            time.sleep(seconds) if seconds else time.sleep(10**9)
        except KeyboardInterrupt:
            pass
        finally:
            client.loop_stop()
            client.disconnect()


def to_csv_text(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    out = df.copy()
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(buf, index=False)
    return buf.getvalue()
