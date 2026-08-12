"""
Tests for photo_import.py - watch-folder importer: file-stability
(partial-copy safety), content-hash dedup (persisted across restarts),
corrupt-file tolerance, and that it feeds the SAME central service.
"""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_core  # noqa: E402
from inspection_db import InspectionDB  # noqa: E402
from inspection_service import InspectionService  # noqa: E402
from photo_import import PhotoImportWatcher  # noqa: E402


@pytest.fixture(autouse=True)
def restore():
    orig = inference_core.predict_masks
    # No-crack stub keeps analysis fast and model-free.
    inference_core.predict_masks = lambda *a, **k: type(
        "R", (), {"masks": None, "boxes": None, "orig_img": (a[0] if a else None)}
    )()
    yield
    inference_core.predict_masks = orig


@pytest.fixture()
def env(tmp_path):
    db = InspectionDB(str(tmp_path / "insp.db"))
    svc = InspectionService(db, str(tmp_path / "inspections"), weights="best.pt")
    watch = tmp_path / "watch"
    watch.mkdir()
    yield db, svc, watch, tmp_path
    db.close()


def _write_photo(path, value=100, size=50):
    cv2.imwrite(str(path), np.full((size, size, 3), value, dtype=np.uint8))


def test_stable_file_is_analyzed_after_two_polls(env):
    db, svc, watch, _ = env
    w = PhotoImportWatcher(str(watch), svc, stable_polls=2)
    _write_photo(watch / "p1.jpg")
    assert w.poll_once() == []          # first sight: not stable yet
    res = w.poll_once()                  # stable now
    assert any(r["action"] == "analyzed" for r in res)
    assert db.count() == 1


def test_same_file_not_reprocessed(env):
    db, svc, watch, _ = env
    w = PhotoImportWatcher(str(watch), svc, stable_polls=1)
    _write_photo(watch / "p1.jpg")
    w.poll_once()
    assert db.count() == 1
    w.poll_once()  # again
    assert db.count() == 1


def test_duplicate_content_is_deduplicated(env):
    db, svc, watch, _ = env
    w = PhotoImportWatcher(str(watch), svc, stable_polls=1)
    _write_photo(watch / "p1.jpg")
    w.poll_once()
    shutil.copy(watch / "p1.jpg", watch / "p1_copy.jpg")  # identical bytes
    res = w.poll_once()
    assert any(r["action"] == "skipped_duplicate" for r in res)
    assert db.count() == 1


def test_distinct_new_photo_is_processed(env):
    db, svc, watch, _ = env
    w = PhotoImportWatcher(str(watch), svc, stable_polls=1)
    _write_photo(watch / "p1.jpg", value=100)
    w.poll_once()
    _write_photo(watch / "p2.jpg", value=30)  # different content
    w.poll_once()
    assert db.count() == 2


def test_corrupt_file_is_marked_invalid_not_crashing(env):
    db, svc, watch, _ = env
    w = PhotoImportWatcher(str(watch), svc, stable_polls=1)
    (watch / "bad.png").write_bytes(b"totally not an image")
    res = w.poll_once()
    assert any(r["action"] == "invalid" for r in res)
    assert db.count() == 0


def test_partial_copy_is_not_processed_until_stable(env):
    db, svc, watch, _ = env
    w = PhotoImportWatcher(str(watch), svc, stable_polls=2)
    target = watch / "p1.jpg"

    # Simulate a growing (still-copying) file: size changes each poll.
    target.write_bytes(b"\x00" * 1000)
    assert w.poll_once() == []                 # sighted, size=1000
    target.write_bytes(b"\x00" * 5000)         # grew -> not stable
    assert all(r["action"] != "analyzed" for r in w.poll_once())
    assert db.count() == 0

    # Now write a real, complete image and let it settle.
    _write_photo(target)
    w.poll_once()   # size changed again -> reset stability
    w.poll_once()   # stable now
    assert db.count() == 1


def test_processed_hashes_persist_across_restart(env):
    db, svc, watch, tmp = env
    _write_photo(watch / "p1.jpg")
    w1 = PhotoImportWatcher(str(watch), svc, stable_polls=1)
    w1.poll_once()
    assert db.count() == 1

    # A brand new watcher instance (simulating a restart) must not
    # re-analyze the already-processed photo.
    w2 = PhotoImportWatcher(str(watch), svc, stable_polls=1)
    res = w2.poll_once()
    assert all(r["action"] != "analyzed" for r in res)
    assert db.count() == 1


def test_import_source_is_recorded(env):
    db, svc, watch, _ = env
    w = PhotoImportWatcher(str(watch), svc, stable_polls=1)
    _write_photo(watch / "p1.jpg")
    w.poll_once()
    rec = db.get_latest()
    assert rec.source == "import"
