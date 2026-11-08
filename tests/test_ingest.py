import json

import numpy as np
import pandas as pd
import pytest

from ccee.ingest import MqttCollector, parse_mqtt_payload, read_logger_csv
from ccee.profiles import get_profile


def write(tmp_path, text, name="log.csv", enc="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(enc))
    return p


def test_plain_iso_csv(tmp_path):
    p = write(tmp_path, "timestamp,temperature\n2025-07-01T00:00:00Z,5.1\n2025-07-01T00:05:00Z,5.2\n")
    r = read_logger_csv(p)
    assert list(r.df["temp_c"]) == [5.1, 5.2] and not r.rejects
    assert r.df["ts"].dt.tz is not None
    assert len(r.metadata["source_sha256"]) == 64


def test_preamble_semicolon_decimal_comma_and_dayfirst(tmp_path):
    text = (
        "Logger report\nSerial number;AB-123\nCalibrated;2025-03-01\n\n"
        "Date/Time;Temperature (C)\n01.07.2025 00:00:00;5,1\n02.07.2025 00:05:00;5,2\n"
    )
    r = read_logger_csv(write(tmp_path, text, enc="cp1252"))
    assert r.metadata["device_id"] == "AB-123" and r.metadata["calibration_date"] == "2025-03-01"
    assert r.df["ts"].iloc[1] == pd.Timestamp("2025-07-02T00:05:00Z")
    assert r.df["temp_c"].iloc[0] == pytest.approx(5.1)


def test_tab_delimited(tmp_path):
    r = read_logger_csv(write(tmp_path, "time\ttemp_c\n2025-07-01 00:00\t4.0\n2025-07-01 00:05\t4.5\n"))
    assert len(r.df) == 2


def test_fahrenheit_auto_detected(tmp_path):
    r = read_logger_csv(write(tmp_path, "timestamp,Temperature (F)\n2025-07-01T00:00:00Z,41.0\n"))
    assert r.df["temp_c"].iloc[0] == pytest.approx(5.0)


def test_timezone_localisation(tmp_path):
    r = read_logger_csv(write(tmp_path, "timestamp,temp_c\n2025-07-01 12:00,5\n"), tz="Europe/Zurich")
    assert r.df["ts"].iloc[0] == pd.Timestamp("2025-07-01T10:00:00Z")


def test_malformed_rows_reported_not_dropped_silently(tmp_path):
    text = "timestamp,temp_c\n2025-07-01T00:00:00Z,5\nnot-a-date,5\n2025-07-01T00:10:00Z,abc\n2025-07-01T00:15:00Z,\n2025-07-01T00:20:00Z,6\n"
    r = read_logger_csv(write(tmp_path, text))
    assert len(r.df) == 2 and len(r.rejects) == 3
    assert {x["line"] for x in r.rejects} == {3, 4, 5}


def test_lat_lon_optional_columns(tmp_path):
    r = read_logger_csv(write(tmp_path, "timestamp,temp_c,lat,lon\n2025-07-01T00:00:00Z,5,47.5,7.5\n2025-07-01T00:05:00Z,5,,\n"))
    assert r.df["lat"].iloc[0] == 47.5 and np.isnan(r.df["lat"].iloc[1])


def test_missing_header_raises(tmp_path):
    with pytest.raises(ValueError):
        read_logger_csv(write(tmp_path, "a,b\n1,2\n"))


def test_mqtt_single_object_and_topic_device():
    r = parse_mqtt_payload('{"ts": "2025-07-01T00:00:00Z", "temp_c": 5.2, "lat": 47.5, "lon": 7.5}', "ccee/dev42/telemetry")
    assert r[0]["device"] == "dev42" and r[0]["temp_c"] == 5.2 and r[0]["lat"] == 47.5


def test_mqtt_batch_epoch_seconds_and_millis():
    r = parse_mqtt_payload(json.dumps([{"time": 1751328000, "temperature": 5}, {"time": 1751328300000, "t": 5.5, "device": "x"}]))
    assert r[0]["ts"] == pd.Timestamp("2025-07-01T00:00:00Z") and r[1]["ts"] == pd.Timestamp("2025-07-01T00:05:00Z")
    assert r[1]["device"] == "x"


@pytest.mark.parametrize(
    "payload",
    ["not json", "[1, 2]", '{"ts": "2025-07-01T00:00:00Z"}', '{"ts": "x", "temp_c": 5}', '{"ts": 1751328000, "temp_c": "hot"}', '{"ts": 1751328000, "temp_c": NaN}'],
)
def test_mqtt_malformed_payloads_raise(payload):
    with pytest.raises(ValueError):
        parse_mqtt_payload(payload, "ccee/d/telemetry")


def test_collector_raises_alarm_and_keeps_going_after_bad_message():
    seen = []
    col = MqttCollector(get_profile("2-8C"), on_alarm=lambda d, a: seen.append((d, a.rule)))
    base = pd.Timestamp("2025-07-01T00:00:00Z")
    col.handle("ccee/d1/telemetry", "garbage")
    for i, v in enumerate([5, 5, 9.5, 9.6, 9.7]):
        col.handle("ccee/d1/telemetry", json.dumps({"ts": (base + pd.Timedelta(minutes=5 * i)).isoformat(), "temp_c": v}))
    assert col.errors and ("d1", "excursion") in seen
    assert len(col.frame("d1")) == 5


def test_collector_separates_devices():
    col = MqttCollector(get_profile("2-8C"))
    col.handle("ccee/a/telemetry", '{"ts": 1751328000, "temp_c": 5}')
    col.handle("ccee/b/telemetry", '{"ts": 1751328000, "temp_c": 9}')
    assert set(col.monitors) == {"a", "b"}
