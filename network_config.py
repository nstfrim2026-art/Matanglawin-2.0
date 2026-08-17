"""
network_config.py - Centralized, dynamic network configuration for MatanglaWIN.

This module is the SINGLE source of truth for every network-facing value
the app needs: the PC's current LAN IPv4 address, the DJI RTMP address the
drone should publish to, the MediaMTX stream key/path, and the localhost
RTSP/WebRTC URLs the backend/website consume.

Design goals (see project requirements):

- No LAN IP is ever hardcoded. The PC may move between completely
  different networks (home Wi-Fi, mobile hotspot, another router, a
  different subnet) and this module must reflect whatever network is
  active *right now*, every time it is asked.
- ``MATANGLAWIN_HOST_IP`` environment variable, if set, always wins.
- Otherwise the active LAN/Wi-Fi IPv4 address is auto-detected using a
  routing-table trick that works identically on Windows, macOS, and
  Linux (no extra dependencies, no shelling out to ``ipconfig``/``ifconfig``).
- If no usable network is found, we fail soft: ``host_ip`` is ``None`` and
  every derived URL that depends on it is also ``None``. The app must not
  crash just because the drone/network isn't connected yet.
- MediaMTX/RTSP/WebRTC run on the SAME machine as this app, so those URLs
  always use ``localhost`` - only the DJI-facing RTMP URL uses the LAN IP,
  because that's the only endpoint an external device (the drone) needs
  to reach across the network.
- The MediaMTX configuration file (mediamtx.yml) is *generated* from these
  values (see :func:`render_mediamtx_yaml`), so it always matches the
  current host IP without anyone editing source or config by hand.

Environment variables:

    MATANGLAWIN_HOST_IP     Optional. Force a specific LAN IP instead of
                             auto-detecting one. Example: 172.20.10.3
    MATANGLAWIN_STREAM_KEY   Optional. MediaMTX stream path/key.
                             Default: "matanglawin"
    MATANGLAWIN_RTMP_PORT    Optional. Default: 1935
    MATANGLAWIN_RTSP_PORT    Optional. Default: 8554
    MATANGLAWIN_WEBRTC_PORT  Optional. Default: 8889
    MATANGLAWIN_API_PORT     Optional. Default: 9997
"""

from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import dataclass, asdict
from typing import Optional

DEFAULT_STREAM_KEY = "matanglawin"
DEFAULT_RTMP_PORT = 1935
DEFAULT_RTSP_PORT = 8554
DEFAULT_WEBRTC_PORT = 8889
DEFAULT_API_PORT = 9997  # MediaMTX's HTTP control API (enabled by default in MediaMTX)

# Addresses that are never a usable "another device on the LAN can reach
# me here" address.
_UNUSABLE_PREFIXES = (
    "127.",       # loopback
    "0.",         # unspecified / invalid
    "169.254.",   # link-local (no DHCP lease / disconnected adapter)
)


def _is_usable_lan_ip(ip: str) -> bool:
    """True if `ip` looks like a real, routable LAN IPv4 address."""
    if not ip:
        return False
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if addr.version != 4:
        return False
    if any(ip.startswith(p) for p in _UNUSABLE_PREFIXES):
        return False
    if addr.is_loopback or addr.is_link_local or addr.is_unspecified:
        return False
    if addr.is_multicast or addr.is_reserved:
        return False
    return True


def detect_lan_ip() -> Optional[str]:
    """
    Auto-detect the active LAN/Wi-Fi IPv4 address of this machine.

    Uses the "connect a UDP socket to a public address" trick: this does
    NOT send any packets (UDP ``connect()`` just asks the OS routing table
    which local interface/IP it *would* use), so it works without
    internet access and without admin/root privileges. It transparently
    picks whichever interface currently owns the machine's default
    route - Wi-Fi, Ethernet, or a mobile hotspot - which is exactly the
    "active" interface DJI Fly needs to reach.

    Works the same way on Windows, macOS, and Linux; no ``ipconfig``/
    ``ifconfig``/``netifaces`` dependency required.

    Returns None if no usable LAN IPv4 address can be determined (e.g.
    no network adapter is connected at all).
    """
    candidates = []

    # Primary strategy: ask the OS routing table via a UDP "connect".
    for target in ("10.255.255.255", "1.1.1.1", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(0.2)
                s.connect((target, 1))
                ip = s.getsockname()[0]
                if _is_usable_lan_ip(ip):
                    candidates.append(ip)
        except OSError:
            continue

    if candidates:
        return candidates[0]

    # Fallback strategy: enumerate addresses via getaddrinfo on the
    # machine's own hostname. Less reliable (can return loopback or a
    # stale/secondary address), but better than nothing when the primary
    # strategy fails (e.g. sandboxed environments with no routing table).
    try:
        hostname = socket.gethostname()
        _, _, ips = socket.gethostbyname_ex(hostname)
        for ip in ips:
            if _is_usable_lan_ip(ip):
                return ip
    except OSError:
        pass

    return None


@dataclass
class NetworkInfo:
    host_ip: Optional[str]
    host_ip_source: str  # "env" | "auto" | "unavailable" | "invalid_env"
    stream_key: str
    rtmp_port: int
    rtsp_port: int
    webrtc_port: int
    api_port: int
    rtmp_address: Optional[str]   # rtmp://<host_ip>:<rtmp_port>  (no path - this is what DJI Fly expects)
    rtmp_url: Optional[str]       # rtmp://<host_ip>:<rtmp_port>/<stream_key>  (full publish URL)
    rtsp_url: str                 # rtsp://localhost:<rtsp_port>/<stream_key>  (always localhost)
    webrtc_url: str               # http://localhost:<webrtc_port>/<stream_key>  (always localhost)

    def to_dict(self) -> dict:
        return asdict(self)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_network_info() -> NetworkInfo:
    """
    Build a fresh `NetworkInfo` snapshot from current environment
    variables + a live LAN IP detection pass.

    Call this fresh whenever the current network info is needed (e.g. on
    every ``/api/network`` request) rather than caching it for the lifetime
    of the process, so the app reflects a Wi-Fi switch or new DHCP lease
    without needing a restart.
    """
    stream_key = os.environ.get("MATANGLAWIN_STREAM_KEY", "").strip() or DEFAULT_STREAM_KEY
    rtmp_port = _int_env("MATANGLAWIN_RTMP_PORT", DEFAULT_RTMP_PORT)
    rtsp_port = _int_env("MATANGLAWIN_RTSP_PORT", DEFAULT_RTSP_PORT)
    webrtc_port = _int_env("MATANGLAWIN_WEBRTC_PORT", DEFAULT_WEBRTC_PORT)
    api_port = _int_env("MATANGLAWIN_API_PORT", DEFAULT_API_PORT)

    env_ip = os.environ.get("MATANGLAWIN_HOST_IP", "").strip()
    host_ip: Optional[str]
    host_ip_source: str

    if env_ip:
        if _is_usable_lan_ip(env_ip):
            host_ip = env_ip
            host_ip_source = "env"
        else:
            # An explicitly-set-but-invalid override must not silently
            # fall back to a *different* IP - that would be confusing -
            # but it also must not crash the app. Surface it as
            # unavailable so /api/network can report the problem.
            host_ip = None
            host_ip_source = "invalid_env"
    else:
        host_ip = detect_lan_ip()
        host_ip_source = "auto" if host_ip else "unavailable"

    if host_ip:
        rtmp_address = f"rtmp://{host_ip}:{rtmp_port}"
        rtmp_url = f"{rtmp_address}/{stream_key}"
    else:
        rtmp_address = None
        rtmp_url = None

    # MediaMTX runs on the same machine as this app - the AI pipeline and
    # the internal player must always use localhost, regardless of what
    # LAN IP the drone is told to use.
    rtsp_url = f"rtsp://localhost:{rtsp_port}/{stream_key}"
    webrtc_url = f"http://localhost:{webrtc_port}/{stream_key}"

    return NetworkInfo(
        host_ip=host_ip,
        host_ip_source=host_ip_source,
        stream_key=stream_key,
        rtmp_port=rtmp_port,
        rtsp_port=rtsp_port,
        webrtc_port=webrtc_port,
        api_port=api_port,
        rtmp_address=rtmp_address,
        rtmp_url=rtmp_url,
        rtsp_url=rtsp_url,
        webrtc_url=webrtc_url,
    )


def check_tcp_port(host: str, port: int, timeout: float = 0.75) -> bool:
    """True if a TCP connection to host:port succeeds (something is listening)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def get_mediamtx_status(info: Optional[NetworkInfo] = None, timeout: float = 0.3) -> dict:
    """
    Best-effort check of whether MediaMTX appears to be reachable on this
    machine, by probing its well-known RTMP/RTSP/WebRTC ports on
    localhost. This does not start, stop, or reconfigure MediaMTX - it
    only reports what it observes, so we never risk launching a second
    instance and hitting "Only one usage of each socket address...".
    """
    info = info or get_network_info()
    rtmp_ok = check_tcp_port("localhost", info.rtmp_port, timeout=timeout)
    rtsp_ok = check_tcp_port("localhost", info.rtsp_port, timeout=timeout)
    webrtc_ok = check_tcp_port("localhost", info.webrtc_port, timeout=timeout)
    return {
        "rtmp_reachable": rtmp_ok,
        "rtsp_reachable": rtsp_ok,
        "webrtc_reachable": webrtc_ok,
        "reachable": rtmp_ok and rtsp_ok and webrtc_ok,
    }


# Publisher-state values reported for the display-only POV.
POV_LIVE = "LIVE"        # a publisher (the drone) is connected and the path is ready
POV_OFFLINE = "OFFLINE"  # MediaMTX API reachable, but no publisher on the path
POV_UNKNOWN = "UNKNOWN"  # MediaMTX API not reachable / not enabled - can't tell


def get_stream_status(info: Optional[NetworkInfo] = None, timeout: float = 0.4) -> dict:
    """
    Best-effort, DISPLAY-ONLY check of whether the drone is currently
    publishing to MediaMTX, by querying MediaMTX's own HTTP control API
    (``/v3/paths/get/<stream_key>``, default port 9997).

    This does NOT run YOLO, open the RTSP stream, or read any frames -
    it only asks MediaMTX whether the path has an active publisher, so
    the dashboard can show a LIVE/OFFLINE badge next to the (separately
    served) WebRTC POV. If MediaMTX's API isn't reachable/enabled, we
    return UNKNOWN and the UI simply shows the raw feed without a
    definitive badge - never an error.
    """
    import json as _json
    import urllib.error
    import urllib.request

    info = info or get_network_info()
    url = f"http://localhost:{info.api_port}/v3/paths/get/{info.stream_key}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 404:
                return {"pov_state": POV_OFFLINE, "api_reachable": True}
            data = _json.loads(resp.read().decode("utf-8"))
        ready = bool(data.get("ready", False))
        return {"pov_state": POV_LIVE if ready else POV_OFFLINE, "api_reachable": True}
    except urllib.error.HTTPError as exc:  # noqa: PERF203
        # 404 => path not created yet (no publisher). Any other HTTP
        # error still means the API is reachable.
        if exc.code == 404:
            return {"pov_state": POV_OFFLINE, "api_reachable": True}
        return {"pov_state": POV_OFFLINE, "api_reachable": True}
    except Exception:  # noqa: BLE001 - API disabled/unreachable/timeout
        return {"pov_state": POV_UNKNOWN, "api_reachable": False}


def render_mediamtx_yaml(info: Optional[NetworkInfo] = None) -> str:
    """
    Generate a ready-to-use ``mediamtx.yml`` from the *current* network
    configuration. MediaMTX itself binds to all interfaces (0.0.0.0), so
    the file never contains a hardcoded LAN IP - the drone reaches it via
    whatever the current ``rtmp_address`` is (shown on the dashboard).

    The generated config:
      - keeps the HTTP control API enabled (so the LIVE/OFFLINE badge works),
      - enables WebRTC (so the browser can play the POV with no plugins),
      - defines the stream path/key from the current configuration.

    This is a convenience artifact the operator can download from the
    dashboard and hand to MediaMTX; it always matches the live config.
    """
    info = info or get_network_info()
    return (
        "# mediamtx.yml - generated by MatanglaWIN network_config.\n"
        "# Regenerate from the dashboard whenever ports/stream key change.\n"
        "# MediaMTX binds to all interfaces; the DJI drone publishes to the\n"
        f"# RTMP address shown on the dashboard (currently: "
        f"{info.rtmp_address or 'unavailable - connect to a network'}).\n"
        "\n"
        "logLevel: info\n"
        "\n"
        "# --- HTTP control API (used only for the display-only LIVE/OFFLINE badge) ---\n"
        "api: yes\n"
        f"apiAddress: :{info.api_port}\n"
        "\n"
        "# --- RTMP ingest (DJI Fly publishes here) ---\n"
        "rtmp: yes\n"
        f"rtmpAddress: :{info.rtmp_port}\n"
        "\n"
        "# --- RTSP (localhost consumers only) ---\n"
        "rtsp: yes\n"
        f"rtspAddress: :{info.rtsp_port}\n"
        "\n"
        "# --- WebRTC (browser POV player) ---\n"
        "webrtc: yes\n"
        f"webrtcAddress: :{info.webrtc_port}\n"
        "\n"
        "paths:\n"
        f"  {info.stream_key}:\n"
        "    # A plain ingest path: the drone publishes, the browser plays.\n"
        "    # No recording, no re-encoding, no AI - the live POV stays clean.\n"
        "    source: publisher\n"
    )
