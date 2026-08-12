"""
network_config.py - Dynamic LAN IP detection, RTMP address generation,
and MediaMTX reachability checks for Matanglawin.

This module removes the need for hardcoded IP addresses. It auto-detects
the active LAN/Wi-Fi IPv4 address at runtime and constructs the RTMP
ingest URL that the DJI Fly app should target.

Environment variables:
    MATANGLAWIN_HOST_IP    Override auto-detected LAN IP (optional).
    MATANGLAWIN_STREAM_KEY Stream key used by MediaMTX (default: matanglawin).
    MEDIAMTX_RTSP_URL      Full RTSP URL override (optional).
    MEDIAMTX_URL           MediaMTX WebRTC player base URL (default: http://localhost:8889).
"""

from __future__ import annotations

import os
import socket
from typing import Optional


def detect_lan_ip() -> Optional[str]:
    """
    Auto-detect the active LAN/Wi-Fi IPv4 address of this machine.

    Uses the UDP connect trick (no actual traffic is sent) to determine
    which local interface the OS would use to reach an external address.
    Works cross-platform (Windows, Linux, macOS).

    Returns None if detection fails or the result is a loopback/link-local address.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Connect to a public DNS address - no data is actually sent
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()

        # Filter out loopback and link-local addresses
        if ip.startswith("127.") or ip.startswith("169.254."):
            return None
        return ip
    except (OSError, socket.error):
        return None


def get_host_ip() -> str:
    """
    Return the host IP to use for RTMP ingest.

    Priority:
        1. MATANGLAWIN_HOST_IP env var (manual override)
        2. Auto-detected LAN IP
        3. Fallback to "127.0.0.1" if detection fails
    """
    override = os.environ.get("MATANGLAWIN_HOST_IP", "").strip()
    if override:
        return override
    detected = detect_lan_ip()
    return detected if detected else "127.0.0.1"


def get_stream_key() -> str:
    """Return the stream key from env var or default 'matanglawin'."""
    return os.environ.get("MATANGLAWIN_STREAM_KEY", "matanglawin").strip() or "matanglawin"


def get_rtmp_address() -> str:
    """
    Build the RTMP ingest URL for the DJI Fly app.

    Format: rtmp://<host-ip>:1935/<stream_key>
    """
    host_ip = get_host_ip()
    stream_key = get_stream_key()
    return f"rtmp://{host_ip}:1935/{stream_key}"


def get_network_info() -> dict:
    """
    Return a dict with all network configuration info for the dashboard.

    Keys: host_ip, rtmp_address, stream_key, rtsp_url, webrtc_url
    """
    host_ip = get_host_ip()
    stream_key = get_stream_key()
    rtmp_address = f"rtmp://{host_ip}:1935/{stream_key}"
    rtsp_url = f"rtsp://localhost:8554/{stream_key}"
    mediamtx_url = os.environ.get("MEDIAMTX_URL", "http://localhost:8889").strip()
    webrtc_url = f"{mediamtx_url}/{stream_key}"

    return {
        "host_ip": host_ip,
        "rtmp_address": rtmp_address,
        "stream_key": stream_key,
        "rtsp_url": rtsp_url,
        "webrtc_url": webrtc_url,
    }


def check_mediamtx_reachable() -> dict:
    """
    Check if MediaMTX is reachable by attempting TCP connections to its
    RTSP port (8554) and WebRTC/HTTP port (8889) on localhost.

    Returns a dict with per-port reachability booleans and an overall status.
    """
    rtsp_ok = _check_tcp_port("127.0.0.1", 8554, timeout=2.0)
    webrtc_ok = _check_tcp_port("127.0.0.1", 8889, timeout=2.0)

    return {
        "rtsp_reachable": rtsp_ok,
        "webrtc_reachable": webrtc_ok,
        "mediamtx_reachable": rtsp_ok or webrtc_ok,
    }


def get_stream_status() -> dict:
    """
    Get combined stream status including network info and MediaMTX reachability.
    """
    info = get_network_info()
    reachability = check_mediamtx_reachable()
    return {**info, **reachability}


def _check_tcp_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """Attempt a TCP connection to host:port with the given timeout."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except (OSError, socket.error):
            return False
        finally:
            s.close()
    except (OSError, socket.error):
        return False
