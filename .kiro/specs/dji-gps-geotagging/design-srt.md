# Design — SRT-based Offline Geotagging (authoritative source)

> Supersedes the FlightRecord approach for offline use. See
> `investigation.md`: the DJI Fly FlightRecord for the Neo 2 is AES-encrypted
> and cannot be decoded offline. **DJI SRT (Video Subtitles) is plaintext**,
> so it IS parseable offline — it is now the authoritative aircraft-position
> source for MatanglaWIN geotagging.

## Why SRT

When "Video Subtitles" is enabled in DJI Fly, DJI writes a plaintext `.SRT`
next to the recording with per-frame telemetry including latitude/longitude
and timestamps. No cloud, no keychain, no SDK — it parses fully offline.

We extract only what geotagging needs: **latitude, longitude, timestamp**
(altitude/heading/speed intentionally ignored and never shown to the operator).

## Architecture (reuses PR #5/#6 infra — no duplication)

```
DJI Fly (Video Subtitles ON)  ─writes─▶  *.SRT in a local folder
                                              │  (MATANGLAWIN_SRT_DIR /
                                              │   MATANGLAWIN_CAPTURE_DIR)
                                              ▼
   srt_watcher.SrtWatcher (in-process, automatic)
        parse via srt_telemetry.parse_file
        ├─ feed (lat,lon,timestamp) ─▶ telemetry_store  (PR #6)
        └─ backfill pending inspections (nearest-time match) ─▶ inspection_db.update_geo
                                              │
 CAPTURE ─▶ analyze (best.pt) ─▶ inspection  │  captured_at ALWAYS stored
        nearest-sample match at capture time ┘  (gps_available, gps_time_delta_ms)
                                              ▼
   /map  (offline Leaflet, PR #6)  red = crack / green = no-crack /
                                   blue = current drone (fresh only)
```

New modules (small, focused):
- `srt_telemetry.py` — robust DJI SRT parser (bracketed `[latitude: …]`,
  loose `latitude: …`, and `GPS(lon,lat,…)` variants; absolute datetime or
  relative offset + configurable timezone; partial/malformed-block safe).
- `srt_watcher.py` — in-process directory watcher; feeds `telemetry_store`
  and backfills inspection coordinates; tracks Mode A vs B.

Reused unchanged: `telemetry_store` (nearest-in-time matching + quality
flags), `inspection_db` (gps columns), `/api/telemetry*`, `/map`,
`/api/inspections/geo`, the vendored offline Leaflet map, and the whole
segmentation/POV pipeline.

## Two telemetry modes (detected, never assumed)

- **Mode A — `AUTOMATED LIVE AIRCRAFT POSITION`**: the newest SRT file is
  observed growing *while recently modified* → live samples; the current
  drone (blue) marker updates and captures match immediately.
- **Mode B — `AUTOMATED POST-CAPTURE GEOTAGGING`**: SRT only appears after
  recording → captures are saved with `gps_available=false`, then the
  watcher **backfills** their coordinates by nearest-timestamp match when the
  SRT arrives, and the markers appear automatically.

The mode is reported honestly in `/api/telemetry/latest` (`srt_mode`) and the
watcher `status()`; the code never labels post-recording data as "live".

## Timestamp matching

`telemetry_store.match_for_capture(capture_ms)` returns the sample nearest in
time to the capture, with `gps_time_delta_ms` (stored internally, never shown
in the operator UI). Capture timestamps carry an explicit offset (e.g.
`+08:00`); naive SRT datetimes are interpreted in the machine's local zone,
or a fixed `MATANGLAWIN_SRT_TZ_OFFSET_MIN` if set (timezone-ambiguity guard).

## Configuration (no hardcoded IP / username)

- `MATANGLAWIN_SRT_DIR`, `MATANGLAWIN_CAPTURE_DIR` — one or more folders
  (os.pathsep-separated) watched for `*.srt`/`*.SRT`. `~`, `%USERPROFILE%`,
  and `$HOME` are expanded; nothing is hardcoded. Default: `<data>/srt`.
- `MATANGLAWIN_SRT_TZ_OFFSET_MIN` — optional fixed tz for naive SRT times.
- PC/LAN address continues to use the project's dynamic `network_config`.

## Acceptance mapping

Detect SRT locally (watcher) → extract lat/lon (parser) → record exact
capture time (service) → nearest-match / backfill (store + watcher) → analyze
(unchanged best.pt) → inspection has image + result + lat/lon → offline map
marker appears automatically → click shows image/result → all offline, no
manual SRT command, no DJI SDK, existing POV + analysis intact.
