import numpy as np
import pandas as pd
from conftest import frame

from ccee.audit import GENESIS, AuditLog, entry_hash, sha256_file
from ccee.integrity import check_integrity


def make():
    log = AuditLog()
    log.append("record_loaded", "tester", {"rows": 10})
    log.append("analysis_run", "tester", {"profile": "2-8C"})
    log.append("alarm", "ccee", {"rule": "excursion"})
    return log


def test_chain_links_and_verifies():
    log = make()
    assert log.entries[0]["prev_hash"] == GENESIS
    assert log.entries[1]["prev_hash"] == log.entries[0]["hash"]
    assert log.verify()[0]


def test_empty_log_verifies():
    assert AuditLog().verify()[0]


def test_edit_detected():
    log = make()
    log.entries[1]["details"]["profile"] = "15-25C"
    ok, msg = log.verify()
    assert not ok and "entry 1" in msg


def test_deletion_detected():
    log = make()
    del log.entries[1]
    assert not log.verify()[0]


def test_reorder_detected():
    log = make()
    log.entries[1], log.entries[2] = log.entries[2], log.entries[1]
    assert not log.verify()[0]


def test_rehash_of_edited_entry_still_breaks_chain():
    log = make()
    log.entries[0]["details"]["rows"] = 99
    log.entries[0]["hash"] = entry_hash(log.entries[0])
    assert not log.verify()[0]  # entry 1 still points at the old hash


def test_save_load_roundtrip(tmp_path):
    log = make()
    path = tmp_path / "a.jsonl"
    log.save(path)
    again = AuditLog.load(path)
    assert again.entries == log.entries and again.verify()[0]


def test_file_hash(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def codes(df, **kw):
    return {f.code for f in check_integrity(df, **kw)}


def test_clean_record_has_no_flags():
    meta = {"device_id": "A", "calibration_date": "x", "source_sha256": "h"}
    assert codes(frame([5.0 + 0.01 * i for i in range(100)]), sync_offsets_s=(0, 1), metadata=meta) == set()


def test_gap_flag():
    f = frame([5.0] * 10)
    f.loc[5:, "ts"] += pd.Timedelta(hours=2)
    assert "data_gaps" in codes(f)


def test_non_monotonic():
    f = frame([5.0] * 10)
    f.loc[4, "ts"] = f.loc[1, "ts"]
    assert "non_monotonic_time" in codes(f)


def test_duplicate_timestamps():
    f = frame([5.0] * 10)
    f.loc[4, "ts"] = f.loc[3, "ts"]
    assert "duplicate_timestamps" in codes(f)


def test_implausible_and_missing_values():
    assert {"implausible_values", "missing_values"} <= codes(frame([5.0, 5.0, 400.0, np.nan, 5.0]))


def test_isolated_spike():
    t = [5.0] * 20
    t[10] = 16.0
    assert "isolated_spikes" in codes(frame(t))


def test_flatline_detected():
    t = list(5 + np.sin(np.arange(40) / 3)) + [5.5] * 20
    assert "flatline" in codes(frame(t))


def test_future_timestamps():
    assert "future_timestamps" in codes(frame([5.0] * 5, start="2099-01-01T00:00:00Z"))


def test_clock_drift_flag_and_missing_sync():
    f = frame([5.0] * 10)
    assert "no_clock_sync" in codes(f)
    assert "clock_drift" in codes(f, sync_offsets_s=(0, 300))


def test_empty_frame():
    assert codes(pd.DataFrame({"ts": pd.to_datetime([], utc=True), "temp_c": []})) == {"empty"}


def test_alcoa_tags_present():
    assert all(f.alcoa for f in check_integrity(frame([5.0] * 10)))
