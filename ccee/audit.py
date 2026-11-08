"""Hash-chained audit trail.

Every entry stores the SHA-256 of its own canonical content plus the previous entry's hash, so editing,
deleting or re-ordering an earlier entry breaks verification of all later ones. This makes tampering
detectable; it does not by itself prevent it (no signatures, no access control, no trusted time source).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

GENESIS = "0" * 64


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def entry_hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class AuditLog:
    entries: list[dict] = field(default_factory=list)

    def append(self, action: str, actor: str = "ccee", details: dict | None = None, ts: datetime | None = None) -> dict:
        prev = self.entries[-1]["hash"] if self.entries else GENESIS
        entry = {
            "seq": len(self.entries),
            "ts": (ts or datetime.now(UTC)).astimezone(UTC).isoformat(),
            "actor": actor,
            "action": action,
            "details": details or {},
            "prev_hash": prev,
        }
        entry["hash"] = entry_hash(entry)
        self.entries.append(entry)
        return entry

    def verify(self) -> tuple[bool, str]:
        prev = GENESIS
        for i, e in enumerate(self.entries):
            if e.get("seq") != i:
                return False, f"entry {i}: sequence number {e.get('seq')} out of order"
            if e.get("prev_hash") != prev:
                return False, f"entry {i}: chain broken (previous hash mismatch)"
            if entry_hash(e) != e.get("hash"):
                return False, f"entry {i}: content does not match its hash"
            prev = e["hash"]
        return True, f"{len(self.entries)} entries verified"

    def save(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(canonical(e) for e in self.entries) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AuditLog:
        lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        return cls([json.loads(ln) for ln in lines])
