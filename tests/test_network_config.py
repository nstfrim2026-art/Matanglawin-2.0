"""
Tests for network_config.py - dynamic LAN IP / RTMP / RTSP / WebRTC
configuration. Covers:

- Manual MATANGLAWIN_HOST_IP override
- Invalid manual override (must not crash, must not silently guess a
  different IP)
- No-network fallback (host_ip -> None when auto-detection can't find
  a usable interface)
- Dynamic RTMP URL generation from whatever host_ip is active
- MATANGLAWIN_STREAM_KEY override
- RTSP/WebRTC URLs always resolve to localhost regardless of host_ip
- MediaMTX status probing against a real closed port (no crash)
"""

import os
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import network_config  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure no leftover MATANGLAWIN_* env vars leak between tests."""
    for key in [
        "MATANGLAWIN_HOST_IP",
        "MATANGLAWIN_STREAM_KEY",
        "MATANGLAWIN_RTMP_PORT",
        "MATANGLAWIN_RTSP_PORT",
        "MATANGLAWIN_WEBRTC_PORT",
    ]:
        monkeypatch.delenv(key, raising=False)
    yield


def test_manual_host_ip_override_is_used(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "192.168.1.20")
    info = network_config.get_network_info()
    assert info.host_ip == "192.168.1.20"
    assert info.host_ip_source == "env"


def test_manual_override_changes_rtmp_address(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "10.0.254.31")
    info = network_config.get_network_info()
    assert info.rtmp_address == "rtmp://10.0.254.31:1935"
    assert info.rtmp_url == "rtmp://10.0.254.31:1935/matanglawin"


def test_different_ip_values_produce_matching_rtmp(monkeypatch):
    for ip in ["172.20.10.3", "172.20.10.4", "192.168.0.15", "10.0.254.42"]:
        monkeypatch.setenv("MATANGLAWIN_HOST_IP", ip)
        info = network_config.get_network_info()
        assert info.host_ip == ip
        assert info.rtmp_address == f"rtmp://{ip}:1935"
        assert info.rtmp_url == f"rtmp://{ip}:1935/matanglawin"


def test_invalid_manual_override_does_not_crash_and_reports_unavailable(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "999.999.999.999")
    info = network_config.get_network_info()
    assert info.host_ip is None
    assert info.host_ip_source == "invalid_env"
    # Must not silently fall back to a *different* auto-detected IP -
    # an explicit-but-broken override should surface as unavailable,
    # not quietly substitute something else.
    assert info.rtmp_address is None
    assert info.rtmp_url is None


def test_invalid_override_garbage_string_does_not_crash(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "not-an-ip-address")
    info = network_config.get_network_info()
    assert info.host_ip is None
    assert info.host_ip_source == "invalid_env"


def test_loopback_and_link_local_are_rejected_as_lan_ips():
    assert network_config._is_usable_lan_ip("127.0.0.1") is False
    assert network_config._is_usable_lan_ip("0.0.0.0") is False
    assert network_config._is_usable_lan_ip("169.254.1.5") is False
    assert network_config._is_usable_lan_ip("") is False
    assert network_config._is_usable_lan_ip("not-an-ip") is False


def test_usable_lan_ip_accepts_real_private_addresses():
    for ip in ["172.20.10.3", "192.168.1.20", "10.0.254.31", "192.168.0.15"]:
        assert network_config._is_usable_lan_ip(ip) is True


def test_no_network_fallback_when_detection_fails(monkeypatch):
    # Force detect_lan_ip() to simulate "no usable network adapter found".
    monkeypatch.setattr(network_config, "detect_lan_ip", lambda: None)
    info = network_config.get_network_info()
    assert info.host_ip is None
    assert info.host_ip_source == "unavailable"
    assert info.rtmp_address is None
    assert info.rtmp_url is None
    # RTSP/WebRTC must still be valid localhost URLs even with no LAN IP.
    assert info.rtsp_url == "rtsp://localhost:8554/matanglawin"
    assert info.webrtc_url == "http://localhost:8889/matanglawin"


def test_no_network_never_raises():
    """get_network_info() must never raise, even with a totally broken environment."""
    try:
        network_config.get_network_info()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"get_network_info() raised unexpectedly: {exc}")


def test_stream_key_default_is_matanglawin(monkeypatch):
    info = network_config.get_network_info()
    assert info.stream_key == "matanglawin"


def test_stream_key_env_override(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_STREAM_KEY", "custom_key_123")
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "172.20.10.3")
    info = network_config.get_network_info()
    assert info.stream_key == "custom_key_123"
    assert info.rtmp_url == "rtmp://172.20.10.3:1935/custom_key_123"
    assert info.rtsp_url == "rtsp://localhost:8554/custom_key_123"
    assert info.webrtc_url == "http://localhost:8889/custom_key_123"


def test_rtsp_and_webrtc_always_use_localhost_regardless_of_host_ip(monkeypatch):
    """
    Critical requirement: MediaMTX + this app run on the same machine, so
    the AI pipeline (RTSP) and website (WebRTC) must ALWAYS target
    localhost, no matter what LAN IP DJI Fly is told to use.
    """
    for ip in ["172.20.10.3", "192.168.1.20", "10.0.254.31"]:
        monkeypatch.setenv("MATANGLAWIN_HOST_IP", ip)
        info = network_config.get_network_info()
        assert info.rtsp_url.startswith("rtsp://localhost:")
        assert info.webrtc_url.startswith("http://localhost:")
        assert ip not in info.rtsp_url
        assert ip not in info.webrtc_url


def test_custom_ports_via_env(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "172.20.10.3")
    monkeypatch.setenv("MATANGLAWIN_RTMP_PORT", "19350")
    monkeypatch.setenv("MATANGLAWIN_RTSP_PORT", "8555")
    monkeypatch.setenv("MATANGLAWIN_WEBRTC_PORT", "8890")
    info = network_config.get_network_info()
    assert info.rtmp_address == "rtmp://172.20.10.3:19350"
    assert info.rtsp_url == "rtsp://localhost:8555/matanglawin"
    assert info.webrtc_url == "http://localhost:8890/matanglawin"


def test_mediamtx_status_against_closed_port_does_not_crash():
    """
    No MediaMTX is running in the test environment - get_mediamtx_status()
    must report unreachable, not raise, and not hang (bounded timeout).
    """
    status = network_config.get_mediamtx_status(timeout=0.2)
    assert status["reachable"] is False
    assert status["rtmp_reachable"] is False
    assert status["rtsp_reachable"] is False
    assert status["webrtc_reachable"] is False


def test_mediamtx_status_reports_true_for_an_actually_listening_port():
    """Sanity check that check_tcp_port can detect a real open port."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        assert network_config.check_tcp_port("127.0.0.1", port, timeout=0.5) is True
    finally:
        server.close()


def test_check_tcp_port_false_for_unused_port():
    # Bind to find a free port, then close it immediately so nothing listens.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert network_config.check_tcp_port("127.0.0.1", port, timeout=0.2) is False


def test_detect_lan_ip_never_raises():
    """detect_lan_ip() must not raise even in a network-restricted sandbox."""
    try:
        network_config.detect_lan_ip()
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"detect_lan_ip() raised unexpectedly: {exc}")


def test_to_dict_contains_all_expected_keys(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "172.20.10.3")
    info = network_config.get_network_info()
    d = info.to_dict()
    for key in [
        "host_ip", "host_ip_source", "stream_key", "rtmp_port", "rtsp_port",
        "webrtc_port", "rtmp_address", "rtmp_url", "rtsp_url", "webrtc_url",
    ]:
        assert key in d
