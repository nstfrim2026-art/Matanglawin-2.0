"""
Repo-wide audit: no LAN IP address should be hardcoded anywhere in the
application's runtime source code. The PC must keep working when it moves
to a different network, so the only source of the host IP is
network_config (env override or live auto-detection).

Example LAN IPs may appear ONLY in documentation (README.md) or in this
test suite, where they are clearly example/test values.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent

RUNTIME_PY_FILES = [
    "app.py",
    "network_config.py",
    "inference_core.py",
    "infer_overlay.py",
    "detector.py",
    "inspection_db.py",
    "inspection_service.py",
    "photo_import.py",
]

PRIVATE_IP_PATTERN = re.compile(
    r"\b(?:"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r")\b"
)


def test_no_hardcoded_private_ip_in_runtime_source():
    ALLOWED_PROBE_TARGET = "10.255.255.255"  # never actually contacted (UDP connect trick)

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
                continue
            if "example" in line.lower():
                continue
            offenders.append(f"{filename}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Found hardcoded LAN IP address(es) in runtime source code:\n" + "\n".join(offenders)
    )


def test_no_hardcoded_ip_in_templates_or_static():
    offenders = []
    for subdir in ["templates", "static"]:
        base = REPO_ROOT / subdir
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in {".html", ".js", ".css"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if PRIVATE_IP_PATTERN.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Found hardcoded LAN IP address(es) in frontend assets:\n" + "\n".join(offenders)
    )


def test_network_config_is_the_single_source_of_lan_ip_logic():
    offenders = []
    for filename in RUNTIME_PY_FILES:
        if filename == "network_config.py":
            continue
        path = REPO_ROOT / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "getsockname()[0]" in text or "10.255.255.255" in text:
            offenders.append(filename)

    assert not offenders, (
        "LAN IP detection logic duplicated outside network_config.py in: " + ", ".join(offenders)
    )
