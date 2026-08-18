#!/usr/bin/env python3
"""
dji_gps_collector.py - Offline companion collector for aircraft telemetry.

Watches a folder of flight-telemetry files, extracts the latest aircraft
position via the pluggable flightrecord_parser, and POSTs it to MatanglaWIN's
LAN endpoint (POST /api/telemetry). It keeps the most recent valid sample in
memory and pushes new samples as the file changes.

Honesty (see .kiro/specs/dji-gps-geotagging/investigation.md): the raw DJI
Fly `DJIFlightRecord_*.txt` for the Neo 2 is AES-encrypted and cannot be
decoded offline. This collector DETECTS that container and logs that it needs
a decrypted CSV export - it never fabricates coordinates. Point it at:
    * a plaintext/CSV telemetry export (from an offline decrypter or an
      external GPS logger), or
    * any folder your offline toolchain writes position CSVs into.

Robustness: tolerates missing/locked/partial files, bad rows, and a PC that
is temporarily unreachable (it just retries on the next scan). It never
modifies or deletes the source files.

Stdlib only (urllib). Runs on Android via Termux or on any machine that can
see both the telemetry folder and the PC. The PC address is configurable
(--server / env) - never hardcoded.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import flightrecord_parser as frp

DEFAULT_ENDPOINT = "/api/telemetry"
DEFAULT_POLL_INTERVAL = 1.0
WATCH_EXTS = (".csv", ".txt", ".log")

log = logging.getLogger("dji_gps_collector")


def http_post_json(server: str, endpoint: str, payload: dict, timeout: float = 5.0):
    """POST JSON to server+endpoint. Returns (ok, detail). Never raises."""
    url = server.rstrip("/") + endpoint
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, f"{resp.status}"
    except (urllib.error.URLError, OSError) as exc:
        return False, f"unreachable: {exc}"


def _iso_utc(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


class TelemetryCollector:
    def __init__(
        self,
        watch_dir: str,
        server: str,
        endpoint: str = DEFAULT_ENDPOINT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        poster: Optional[Callable[[str, str, dict], tuple]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.watch_dir = Path(watch_dir)
        self.server = server
        self.endpoint = endpoint
        self.poll_interval = poll_interval
        self._poster = poster or (lambda s, e, p: http_post_json(s, e, p))
        self._sleep = sleep
        self.last_sent_ms: Optional[float] = None
        self.latest_point: Optional[frp.TelemetryPoint] = None

    def _newest_file(self) -> Optional[Path]:
        try:
            files = [p for p in self.watch_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in WATCH_EXTS]
        except OSError:
            return None
        if not files:
            return None
        # Newest by modification time = the active/most-recent flight.
        return max(files, key=lambda p: _safe_mtime(p))

    def scan_once(self) -> dict:
        """
        One pass: find the newest telemetry file, parse it, and POST the
        latest valid position if it is newer than what we last sent. Returns
        an action dict; never raises.
        """
        newest = self._newest_file()
        if newest is None:
            return {"action": "no_file"}

        result = frp.parse_file(str(newest))
        if not result.decodable:
            return {"action": "skip", "file": newest.name, "fmt": result.fmt,
                    "reason": result.reason}
        point = result.latest
        if point is None:
            return {"action": "no_point", "file": newest.name}

        self.latest_point = point
        if self.last_sent_ms is not None and point.timestamp_ms <= self.last_sent_ms:
            return {"action": "unchanged", "file": newest.name}

        payload = {
            "latitude": point.latitude,
            "longitude": point.longitude,
            "altitude_m": point.altitude_m,
            "timestamp": _iso_utc(point.timestamp_ms),
            "source": "dji_flight_record",
        }
        ok, detail = self._poster(self.server, self.endpoint, payload)
        if ok:
            self.last_sent_ms = point.timestamp_ms
            log.info("[GPS] sent %.6f,%.6f alt=%s (%s)", point.latitude,
                     point.longitude, point.altitude_m, newest.name)
            return {"action": "sent", "file": newest.name, "detail": detail}
        log.warning("[GPS] PC unreachable, will retry (%s)", detail)
        return {"action": "failed", "file": newest.name, "detail": detail}

    def run(self) -> None:
        log.info("[GPS] watching %s -> %s%s", self.watch_dir, self.server.rstrip("/"), self.endpoint)
        while True:
            try:
                self.scan_once()
            except Exception as exc:  # noqa: BLE001 - collector must never die
                log.error("[GPS] scan error: %s", exc)
            self._sleep(self.poll_interval)


def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def discover_server(explicit: Optional[str]) -> Optional[str]:
    """
    Resolve the MatanglaWIN base URL without hardcoding an IP:
      1. explicit --server / env value wins;
      2. else try mDNS/zeroconf (optional dependency) for _http._tcp
         'matanglawin'; returns None if not resolvable.
    """
    if explicit:
        return explicit
    try:
        from zeroconf import Zeroconf, ServiceBrowser  # type: ignore  # noqa: F401
        # Best-effort mDNS; kept optional so the collector has no hard dep.
        # (A full resolver is intentionally out of scope here.)
    except Exception:  # noqa: BLE001
        return None
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline DJI aircraft-GPS collector for MatanglaWIN.")
    p.add_argument("--server", default=os.environ.get("MATANGLAWIN_TELEMETRY_SERVER", ""),
                   help="MatanglaWIN base URL, e.g. http://<PC-LAN-IP>:5000 "
                        "(or MATANGLAWIN_TELEMETRY_SERVER). Never hardcode the IP in source.")
    p.add_argument("--watch-dir", default=os.environ.get("MATANGLAWIN_TELEMETRY_DIR", ""),
                   help="Folder of telemetry files (CSV export, etc.). "
                        "Or MATANGLAWIN_TELEMETRY_DIR.")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--interval", type=float,
                   default=float(os.environ.get("MATANGLAWIN_TELEMETRY_INTERVAL", DEFAULT_POLL_INTERVAL)))
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    server = discover_server(args.server)
    if not server or not args.watch_dir:
        log.error("Need --server (or MATANGLAWIN_TELEMETRY_SERVER) and --watch-dir "
                  "(or MATANGLAWIN_TELEMETRY_DIR).")
        return 2
    if not Path(args.watch_dir).exists():
        log.error("Watch dir does not exist: %s", args.watch_dir)
        return 2
    TelemetryCollector(args.watch_dir, server, endpoint=args.endpoint,
                       poll_interval=args.interval).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
