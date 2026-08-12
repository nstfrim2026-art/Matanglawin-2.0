"""
Tests for import_ledger.py - content-hash de-duplication for /api/import.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from import_ledger import ImportLedger, hash_bytes  # noqa: E402


def test_hash_is_stable_and_content_based():
    assert hash_bytes(b"abc") == hash_bytes(b"abc")
    assert hash_bytes(b"abc") != hash_bytes(b"abd")


def test_seen_add_roundtrip(tmp_path):
    led = ImportLedger(str(tmp_path / "state.json"))
    d = hash_bytes(b"photo-bytes")
    assert led.seen(d) is False
    led.add(d)
    assert led.seen(d) is True


def test_dedup_persists_across_restart(tmp_path):
    p = str(tmp_path / "state.json")
    d = hash_bytes(b"same-photo")
    led = ImportLedger(p)
    led.add(d)
    # New instance simulates an app restart - it must remember the hash.
    led2 = ImportLedger(p)
    assert led2.seen(d) is True
    assert len(led2) == 1
