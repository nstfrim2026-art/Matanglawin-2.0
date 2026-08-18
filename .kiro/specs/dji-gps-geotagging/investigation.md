# Investigation — Automated Offline DJI Neo 2 GPS Geotagging

> This document is written **before** the implementation, as required. Its
> most important job is to answer honestly whether the DJI Fly FlightRecord
> can be used as an **offline, real-time** aircraft-GPS source — and to make
> the design match reality instead of a hopeful assumption.

## TL;DR

Reading the DJI Fly FlightRecord **fully offline is not possible for the
DJI Neo 2.** DJI Fly (`dji.go.v5`) writes an **encrypted** flight log whose
per-record telemetry (GPS, altitude, timestamps) is **AES-256-CBC encrypted
starting at log version 13**, and the decryption keychain must be fetched
from **DJI's cloud API using an API key**. The Neo 2 is a current-generation
aircraft on DJI Fly, so its logs are v13+/encrypted. This directly conflicts
with the "no internet / no cloud / no DJI SDK" core requirement.

Therefore:

- **LIVE TELEMETRY MODE from the FlightRecord: NOT ACHIEVABLE offline.**
- **POST-FLIGHT MODE from the FlightRecord: also blocked offline**, because
  even the finished file cannot be decrypted without DJI's keychain service.

I will **not** claim live aircraft GPS from the FlightRecord, because that
would be false. Instead I build an offline-correct, **source-agnostic**
geotagging backbone (telemetry ingest + nearest-sample matching + capture
association + offline map) and a **pluggable FlightRecord parser** so that
*any* aircraft-GPS source that a user can produce offline (a plaintext/CSV
telemetry export, a licensed offline decrypter's output, an external NMEA
GPS feeding the same endpoint, etc.) drops straight in. The encrypted DJI
`.txt` itself is detected and reported as "not decryptable offline" rather
than silently faked.

## Sources (public, verified 2026-08)

- `lvauvillier/dji-log-parser`, `JrVolt/DJI-LogParser`, `pydjirecord`
  (PyPI), and DJI's own `FlightRecordParsingLib`: all state that **v13+
  records are AES-encrypted and require a keychain obtained from the DJI
  API / an App Key**. `pydjirecord` explicitly: "AES-256-CBC encryption
  (v13-14) with per-feature-point keys fetched from the DJI API."
- DJI support + community threads: FlightRecords live at
  `Android/data/dji.go.v5/file/FlightRecord/` (a.k.a. `.../DJI/dji.go.v5/
  FlightRecord/`), named `DJIFlightRecord_YYYY-MM-DD_[HH-MM-SS].txt`, sampled
  at ~10 Hz, and "flight records are encrypted so you need a way to decrypt
  them" (airdata / phantomhelp / Flight Reader — all key-holding tools).
- Content rephrased for compliance with source licensing.

---

## PART 1 answers — what DJI Fly writes

| # | Question | Finding |
|---|---|---|
| 1 | File format | Proprietary DJI flight-log container (`.txt`), header + serialized telemetry frames. **Encrypted** (v13+ AES-256-CBC). |
| 2 | Filename pattern | `DJIFlightRecord_YYYY-MM-DD_[HH-MM-SS].txt` |
| 3 | GPS present? | Yes — latitude/longitude per frame (once decrypted). |
| 4 | Altitude present? | Yes — relative/absolute altitude per frame. |
| 5 | Timestamps present? | Yes — per-frame time offsets + a flight start time. |
| 6 | Coordinate format | Decimal degrees (WGS84) after decode (radians internally in some versions). |
| 7 | Sample frequency | ~10 Hz. |
| 8 | Updated during flight? | The file is written progressively, but this is unconfirmed for Neo 2 and **moot**: the frames are encrypted, so a growing file yields no readable samples offline. |
| 9 | Safe to read newest while recording? | Read-only tailing is safe in principle, but **irrelevant offline** without the keychain. |
| 10 | Parse locally without internet? | **NO for v13+.** The AES keychain must be pulled from DJI's cloud API. Older XOR-encoded versions (v7–12) could be decoded offline, but the Neo 2 is not one of those. |

**Conclusion:** the FlightRecord is a real ~10 Hz aircraft-GPS source, but it
is **locked behind DJI's cloud keychain** for this aircraft. It cannot serve
as an offline telemetry source on its own.

---

## What CAN be built offline (and is, in this feature)

The valuable, honest, offline pieces are everything *around* the raw
FlightRecord decode:

1. **Telemetry ingest API** (`POST /api/telemetry`) on the PC, retaining the
   latest valid aircraft position in memory + DB. Fully offline, LAN-only,
   PC IP auto-discovered via the existing `/api/network` (no hardcoded IP).
2. **Nearest-sample matching** (`telemetry_store.py`): buffer time-stamped
   samples and, at capture time, pick the sample **closest to the capture
   timestamp**, recording `gps_available` and `gps_time_delta_ms` — exactly
   the Part 5 logic, independent of where samples come from.
3. **Capture → GPS association**: when an inspection is created, stamp the
   nearest telemetry sample; store coordinates internally; **never block
   crack analysis when GPS is missing** (`gps_available=false`).
4. **Offline inspection map** (`/map`): a locally-vendored **Leaflet** map
   (no Google/cloud provider, no cloud tiles required) with a red marker for
   CRACK DETECTED and green for NO CRACK DETECTED, popups showing the
   original + highlighted images, result, and time.
5. **Pluggable FlightRecord parser** (`flightrecord_parser.py`): a clean
   interface + a working **plaintext/CSV** parser (for any offline-decrypted
   or exported telemetry) and honest **detection of the encrypted DJI `.txt`
   container**, which returns a clear "needs offline keychain — cannot decode
   offline" status rather than fabricating coordinates.
6. **Companion collector** (`dji_gps_collector.py`): watches a folder, picks
   the newest record, parses via the pluggable parser, keeps the latest valid
   sample, and POSTs it to the auto-discovered PC — robust to partial/locked
   files, never modifying DJI's files.

This means the moment a user has *any* offline way to obtain the aircraft's
position stream (a licensed offline decrypter that emits CSV, an on-aircraft
/ external NMEA GPS, a future firmware/SRT telemetry path, etc.), the whole
pipeline works end-to-end with **no code changes** — just point the collector
at that source.

---

## PART 12 — Deliverable status

```
FlightRecord format:
  DJIFlightRecord_YYYY-MM-DD_[HH-MM-SS].txt, ~10 Hz telemetry frames,
  AES-256-CBC encrypted (log version 13+, which the Neo 2 uses).

Does it update during flight:
  Written progressively, but UNVERIFIED for Neo 2 and irrelevant offline
  (frames are encrypted). Not usable as a live source without DJI's cloud.

GPS fields available:
  latitude, longitude, altitude, per-frame timestamps (AFTER decryption).

Sample frequency:
  ~10 Hz.

Live GPS collection (offline, from FlightRecord):
  NO. Blocked by AES v13+ encryption whose keychain requires the DJI cloud
  API. Neither LIVE nor POST-FLIGHT offline decode is possible for Neo 2.

Android collector:
  BUILT, source-agnostic (dji_gps_collector.py) — watches a folder, parses
  via a pluggable parser, POSTs latest sample to the PC. Works today with a
  plaintext/CSV telemetry source; the encrypted DJI .txt is detected and
  reported as not-offline-decodable (not faked).

Windows telemetry endpoint:
  BUILT — POST /api/telemetry, GET /api/telemetry/latest; latest valid
  sample retained in memory + DB; PC auto-discovered (no hardcoded IP).

Capture/GPS synchronization:
  BUILT — nearest-sample-to-capture matching with gps_available +
  gps_time_delta_ms quality flags; analysis never blocked by missing GPS.

Offline map:
  BUILT — locally vendored Leaflet, no cloud provider/tiles required.

Inspection marker:
  BUILT — red = CRACK DETECTED, green = NO CRACK DETECTED, popup with
  images + result + time.

Automated end-to-end test:
  PASS for the offline backbone (telemetry store, parser on fixtures,
  collector with injected poster, endpoint, capture association, DB, map/geo
  API, missing-GPS behavior). The real-drone FlightRecord decode step is
  NOT part of automated tests because it cannot run offline for the Neo 2.
```

### Mode actually delivered

```
POST-FLIGHT / SOURCE-AGNOSTIC TELEMETRY MODE (offline).
NOT live FlightRecord telemetry — that is not possible offline for the Neo 2.
```

## Recommended offline paths to real aircraft GPS (for the operator to choose)

1. **Licensed offline decrypter → CSV → collector.** Tools like "Flight
   Reader" can process encrypted logs offline (they hold keys under
   license). Export CSV per flight, drop it in the collector's watch folder;
   this feature then geotags via nearest-timestamp matching (post-flight).
2. **External GPS feeding `POST /api/telemetry`.** A small GPS unit (or a
   phone app with location permission) on/near the aircraft that pushes NMEA/
   JSON to the endpoint gives **live** samples — but that is not
   *aircraft-derived* in the strict sense, so it is offered only as an option.
3. **One-time online keychain fetch, then offline.** If a single
   internet-connected keychain fetch is acceptable, `pydjirecord`/
   `dji-log-parser` can cache keys; subsequent parsing is offline. This
   breaks the strict "never internet" rule once, so it is documented, not
   enabled by default.

The build below implements #1/#2 cleanly and leaves #3 as a documented,
opt-in extension via the pluggable parser.
