"""
Repo-wide audit: no LAN IP address should be hardcoded anywhere in the
application's runtime source code.

Per the project requirements, example LAN IPs (172.20.10.3, 10.0.254.31,
etc.) may appear ONLY in documentation (README.md) or in this test
suite itself, where they are clearly example/test values, not values
the running application depends on.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Runtime source files the application actually executes. Deliberately
# excludes README.md (documentation examples are fine), this tests/
# directory (test fixtures use example IPs on purpose), and any build
# artifacts / virtualenvs.
RUNTIME_PY_FILES = [
    "app.py",
    "network_config.py",
    "gps_provider.py",
    "inference_core.py",
    "infer_overlay.py",
    "detector.py",
    "inspection_db.py",
    "inspection_service.py",
    "photo_import.py",
    "report_generator.py",
]

# A private-use IPv4 literal, e.g. 172.20.10.3, 192.168.1.20, 10.0.254.31.
# We look for private-range prefixes specifically (172.16-31.x.x,
# 192.168.x.x, 10.x.x.x) since those are the LAN ranges devices like the
# PC/DJI Neo 2 would actually use - not just any dotted-quad (which would
# also flag version numbers, unrelated constants, etc.).
PRIVATE_IP_PATTERN = re.compile(
    r"\b(?:"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r")\b"
)


def test_no_hardcoded_private_ip_in_runtime_source():
    """
    Any private-range IPv4 literal found in runtime source must be one
    of two explicitly-reviewed, non-configuration exceptions:

    1. A documentation/comment example (contains "Example" nearby) -
       e.g. `MATANGLAWIN_HOST_IP=172.20.10.3` in a docstring, purely
       illustrating the env var format.
    2. The `10.255.255.255` UDP "connect" probe target used by
       detect_lan_ip() - this address is NEVER actually contacted (UDP
       connect() just asks the OS routing table which local interface
       it would use); it is not a LAN IP the app depends on or reports
       anywhere, so it is not a "hardcoded network assumption" in the
       sense the requirement is guarding against.

    Anything else is a genuine failure - it would mean a real,
    reachable-looking LAN IP is baked into the code.
    """
    ALLOWED_PROBE_TARGET = "10.255.255.255"  # never actually contacted - see docstring above

    offenders = []
    for filename in RUNTIME_PY_FILES:
        path = REPO_ROOT / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not PRIVATE_IP_PATTERN.search(line):
                continue
            if ALLOWED_PROBE_TARGET in line:
                continue  # the UDP-trick probe target - not a config value, see above
            if "example" in line.lower():
                continue  # documented illustrative example, not a real config value
            offenders.append(f"{filename}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Found hardcoded LAN IP address(es) in runtime source code:\n"
        + "\n".join(offenders)
    )


def test_no_hardcoded_ip_in_templates_or_static():
    offenders = []
    for subdir in ["templates", "static"]:
        base = REPO_ROOT / subdir
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".html", ".js", ".css"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if PRIVATE_IP_PATTERN.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Found hardcoded LAN IP address(es) in frontend assets:\n" + "\n".join(offenders)
    )


def test_network_config_is_the_single_source_of_lan_ip_logic():
    """
    Sanity check for requirement #30 (centralize network config, don't
    duplicate IP-detection logic across files): no other runtime module
    should re-implement its own socket-based LAN IP detection.
    """
    offenders = []
    for filename in RUNTIME_PY_FILES:
        if filename == "network_config.py":
            continue
        path = REPO_ROOT / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        # A crude but effective signal: another module opening its own
        # UDP socket to a public IP purely to read getsockname(), which
        # is the exact trick network_config.detect_lan_ip() uses.
        if "getsockname()[0]" in text or "10.255.255.255" in text:
            offenders.append(filename)

    assert not offenders, (
        "LAN IP detection logic duplicated outside network_config.py in: "
        + ", ".join(offenders)
    )
