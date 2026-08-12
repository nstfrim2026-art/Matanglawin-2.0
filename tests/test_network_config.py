"""
Tests for network_config.py - dynamic, non-hardcoded network configuration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import network_config as nc  # noqa: E402


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "172.20.10.3")  # example test value
    info = nc.get_network_info()
    assert info.host_ip == "172.20.10.3"
    assert info.host_ip_source == "env"
    assert info.rtmp_address == "rtmp://172.20.10.3:1935"
    assert info.rtmp_url == "rtmp://172.20.10.3:1935/matanglawin"


def test_invalid_env_override_is_reported_not_silently_replaced(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "not-an-ip")
    info = nc.get_network_info()
    assert info.host_ip is None
    assert info.host_ip_source == "invalid_env"
    assert info.rtmp_address is None  # nothing fabricated


def test_localhost_urls_never_use_lan_ip(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "192.168.1.50")  # example test value
    info = nc.get_network_info()
    # The consumer-side URLs are always localhost, never the LAN IP.
    assert info.rtsp_url == "rtsp://localhost:8554/matanglawin"
    assert info.webrtc_url == "http://localhost:8889/matanglawin"


def test_custom_ports_and_stream_key(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "10.0.0.5")  # example test value
    monkeypatch.setenv("MATANGLAWIN_STREAM_KEY", "drone1")
    monkeypatch.setenv("MATANGLAWIN_RTMP_PORT", "1940")
    monkeypatch.setenv("MATANGLAWIN_WEBRTC_PORT", "9000")
    info = nc.get_network_info()
    assert info.stream_key == "drone1"
    assert info.rtmp_address == "rtmp://10.0.0.5:1940"
    assert info.webrtc_url == "http://localhost:9000/drone1"


def test_detect_lan_ip_returns_usable_or_none():
    ip = nc.detect_lan_ip()
    assert ip is None or nc._is_usable_lan_ip(ip)


def test_is_usable_lan_ip_rejects_loopback_and_linklocal():
    assert not nc._is_usable_lan_ip("127.0.0.1")
    assert not nc._is_usable_lan_ip("169.254.1.1")
    assert not nc._is_usable_lan_ip("")
    assert nc._is_usable_lan_ip("192.168.1.10")


def test_mediamtx_yaml_generated_from_current_config(monkeypatch):
    monkeypatch.setenv("MATANGLAWIN_HOST_IP", "172.20.10.3")  # example test value
    monkeypatch.setenv("MATANGLAWIN_STREAM_KEY", "matanglawin")
    yaml = nc.render_mediamtx_yaml()
    # It reflects the configured ports/key/paths...
    assert "rtmpAddress: :1935" in yaml
    assert "webrtcAddress: :8889" in yaml
    assert "matanglawin:" in yaml
    assert "source: publisher" in yaml
    # ...and mentions the current RTMP address in a comment for the operator.
    assert "rtmp://172.20.10.3:1935" in yaml


def test_stream_status_is_display_only_shape(monkeypatch):
    # Force MediaMTX API to be unreachable -> UNKNOWN, never crash.
    monkeypatch.setattr(nc, "get_network_info", lambda: nc.NetworkInfo(
        host_ip=None, host_ip_source="unavailable", stream_key="matanglawin",
        rtmp_port=1935, rtsp_port=8554, webrtc_port=8889, api_port=1,  # port 1: nothing listening
        rtmp_address=None, rtmp_url=None,
        rtsp_url="rtsp://localhost:8554/matanglawin",
        webrtc_url="http://localhost:8889/matanglawin",
    ))
    status = nc.get_stream_status(timeout=0.1)
    assert status["pov_state"] in ("LIVE", "OFFLINE", "UNKNOWN")
