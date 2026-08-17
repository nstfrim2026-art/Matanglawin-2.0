"""
Tests for dji_photo_bridge.PhotoUploadBridge - the companion uploader that
watches the DJI album folder and pushes originals to MatanglaWIN.

No network is used: an in-memory uploader is injected. Verifies size
stability (waits for the download to finish), content de-dup (never sends
the same photo twice), retry on PC-unavailable, byte-for-byte preservation
of the original, and non-retryable server rejections.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dji_photo_bridge import PhotoUploadBridge  # noqa: E402


class RecordingUploader:
    """Injected uploader. Records every call; return value is configurable."""
    def __init__(self, result=(True, False, "201 ok")):
        self.result = result
        self.calls = []  # list of (data, filename)

    def __call__(self, server, endpoint, data, filename):
        self.calls.append((data, filename))
        return self.result


def _bridge(tmp_path, uploader, stable_polls=1, retries=3):
    watch = tmp_path / "album"
    watch.mkdir(parents=True, exist_ok=True)
    return PhotoUploadBridge(
        watch_dir=str(watch),
        server="http://pc.example:5000",  # example host - not a real hardcoded LAN IP
        stable_polls=stable_polls,
        retries=retries,
        uploader=uploader,
        sleep=lambda _s: None,  # no real waiting in tests
    ), watch


def test_uploads_new_photo_and_preserves_bytes(tmp_path):
    up = RecordingUploader()
    bridge, watch = _bridge(tmp_path, up, stable_polls=1)
    data = bytes(range(256)) * 4
    (watch / "DJI_0001.jpg").write_bytes(data)

    res = bridge.scan_once()
    assert any(r["action"] == "uploaded" for r in res)
    assert len(up.calls) == 1
    # The uploaded bytes are exactly the original file bytes.
    assert up.calls[0][0] == data
    assert up.calls[0][1] == "DJI_0001.jpg"


def test_waits_until_file_is_stable(tmp_path):
    up = RecordingUploader()
    bridge, watch = _bridge(tmp_path, up, stable_polls=2)
    (watch / "DJI_0002.jpg").write_bytes(b"partial-then-complete")

    first = bridge.scan_once()   # first sighting -> deferred, not uploaded
    assert first and first[0]["action"] == "deferred"
    assert up.calls == []

    second = bridge.scan_once()  # size unchanged -> stable -> uploaded
    assert any(r["action"] == "uploaded" for r in second)
    assert len(up.calls) == 1


def test_never_uploads_same_photo_twice(tmp_path):
    up = RecordingUploader()
    bridge, watch = _bridge(tmp_path, up, stable_polls=1)
    data = b"identical-photo-content"
    (watch / "a.jpg").write_bytes(data)
    bridge.scan_once()
    assert len(up.calls) == 1

    # Same file re-seen, and an identical-content copy under a new name:
    (watch / "a_copy.jpg").write_bytes(data)
    res = bridge.scan_once()
    assert len(up.calls) == 1  # still only one upload
    assert any(r["action"] == "duplicate" for r in res)


def test_dedup_survives_restart(tmp_path):
    up = RecordingUploader()
    bridge, watch = _bridge(tmp_path, up, stable_polls=1)
    (watch / "b.jpg").write_bytes(b"photo-b")
    bridge.scan_once()
    assert len(up.calls) == 1

    # New bridge instance (restart) reads the persisted uploaded-hash state.
    up2 = RecordingUploader()
    bridge2 = PhotoUploadBridge(
        watch_dir=str(watch), server="http://pc.example:5000",
        stable_polls=1, uploader=up2, sleep=lambda _s: None,
    )
    res = bridge2.scan_once()
    assert up2.calls == []
    assert any(r["action"] == "duplicate" for r in res)


def test_retries_when_pc_unavailable(tmp_path):
    # Uploader always reports a retryable failure (PC down).
    up = RecordingUploader(result=(False, True, "unreachable"))
    bridge, watch = _bridge(tmp_path, up, stable_polls=1, retries=2)
    (watch / "c.jpg").write_bytes(b"photo-c")

    res = bridge.scan_once()
    assert any(r["action"] == "failed" for r in res)
    assert len(up.calls) == 2  # tried `retries` times

    # Not marked done -> a later scan (PC back up) retries and succeeds.
    up.result = (True, False, "201 ok")
    res2 = bridge.scan_once()
    assert any(r["action"] == "uploaded" for r in res2)


def test_server_rejection_is_not_retried_forever(tmp_path):
    # Non-retryable rejection (e.g. HTTP 400 not an image): handled once.
    up = RecordingUploader(result=(False, False, "HTTP 400"))
    bridge, watch = _bridge(tmp_path, up, stable_polls=1, retries=3)
    (watch / "bad.jpg").write_bytes(b"not really an image")

    res = bridge.scan_once()
    assert any(r["action"] == "uploaded" for r in res)  # treated as handled
    assert len(up.calls) == 1  # only ONE attempt, no retry loop

    res2 = bridge.scan_once()
    # Already handled -> treated as duplicate, no further attempts.
    assert len(up.calls) == 1


def test_ignores_non_image_files(tmp_path):
    up = RecordingUploader()
    bridge, watch = _bridge(tmp_path, up, stable_polls=1)
    (watch / "notes.txt").write_text("hello")
    (watch / "video.mp4").write_bytes(b"\x00\x00")
    assert bridge.scan_once() == []
    assert up.calls == []
