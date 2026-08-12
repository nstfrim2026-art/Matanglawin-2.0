/**
 * dashboard.js - Drives the display-only dashboard.
 *
 * Two independent concerns, matching the backend split:
 *   1. Live Drone POV  - the raw MediaMTX WebRTC feed, loaded directly
 *      into an <iframe>. The backend never proxies/re-encodes/annotates
 *      it, and NO detection runs on it. A LIVE/OFFLINE/UNKNOWN badge is
 *      derived from MediaMTX's own publisher state (/api/stream/status),
 *      which does not run YOLO.
 *   2. Latest Inspection - the most recent PHOTO analysis result
 *      (/api/inspection/latest): status + original/highlighted/crack
 *      images. Never shows confidence.
 */

const REFRESH_INTERVAL_MS = 3000;
let lastWebrtcUrl = null;

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

async function refreshNetwork() {
  try {
    const res = await fetch('/api/network');
    const info = await res.json();
    setText('net-host-ip', info.host_ip || 'Unavailable (no network detected)');
    setText('net-rtmp-address', info.rtmp_address || 'Unavailable - connect to a network first');
    setText('net-stream-key', info.stream_key || '\u2014');
    setText('net-webrtc-url', info.webrtc_url || '\u2014');

    if (info.webrtc_url && info.webrtc_url !== lastWebrtcUrl) {
      lastWebrtcUrl = info.webrtc_url;
      const iframe = document.getElementById('pov-iframe');
      if (iframe) iframe.src = info.webrtc_url;
    }
  } catch (err) {
    setText('net-host-ip', 'Error loading network info');
  }

  try {
    const res = await fetch('/api/mediamtx/status');
    const status = await res.json();
    setText('net-mediamtx', status.reachable ? 'Reachable' : 'Not reachable');
  } catch (err) {
    setText('net-mediamtx', 'Unknown');
  }
}

function setPovBadge(state) {
  const badge = document.getElementById('stream-badge');
  const placeholder = document.getElementById('pov-placeholder');
  const iframe = document.getElementById('pov-iframe');
  if (!badge) return;

  badge.textContent = state;
  badge.className = 'badge ' + (
    state === 'LIVE' ? 'badge-live' : state === 'OFFLINE' ? 'badge-offline' : 'badge-connecting'
  );

  // Show the WebRTC player whenever we have a URL; the badge just
  // annotates whether a publisher is currently detected. When the
  // MediaMTX API isn't reachable (UNKNOWN) we still show the feed.
  const showFeed = state === 'LIVE' || state === 'UNKNOWN';
  if (placeholder) placeholder.style.display = showFeed ? 'none' : 'block';
  if (iframe) iframe.style.display = showFeed ? 'block' : 'none';
}

async function refreshPovStatus() {
  try {
    const res = await fetch('/api/stream/status');
    const status = await res.json();
    setPovBadge(status.pov_state || 'UNKNOWN');
  } catch (err) {
    setPovBadge('UNKNOWN');
  }
}

function renderLatest(rec) {
  const empty = document.getElementById('latest-empty');
  const content = document.getElementById('latest-content');
  if (!rec) {
    if (empty) empty.style.display = 'block';
    if (content) content.style.display = 'none';
    return;
  }
  if (empty) empty.style.display = 'none';
  if (content) content.style.display = 'block';

  const banner = document.getElementById('latest-status');
  if (banner) {
    banner.textContent = rec.status;
    banner.className = 'status-banner ' + (rec.has_crack ? 'status-crack' : 'status-ok');
  }
  setText('latest-timestamp', rec.timestamp);
  setText('latest-source', rec.source === 'import' ? 'DJI import' : 'Manual upload');
  setText(
    'latest-gps',
    rec.gps_available ? `${rec.latitude.toFixed(5)}, ${rec.longitude.toFixed(5)}` : 'Unavailable'
  );

  const urls = rec.urls || {};
  const original = document.getElementById('latest-original');
  if (original && urls.original) original.src = urls.original;

  const hlCol = document.getElementById('latest-highlighted-col');
  const hl = document.getElementById('latest-highlighted');
  if (rec.has_crack && urls.highlighted) {
    if (hl) hl.src = urls.highlighted;
    if (hlCol) hlCol.style.display = 'block';
  } else if (hlCol) {
    hlCol.style.display = 'none';
  }

  const cracksCol = document.getElementById('latest-cracks-col');
  const cracks = document.getElementById('latest-cracks');
  if (rec.has_crack && urls.cracks && urls.cracks.length) {
    if (cracks) {
      cracks.innerHTML = urls.cracks
        .map((u, i) => `<div class="crack-item"><img src="${u}" class="crack-img" alt="Crack ${i + 1}"><span class="hint">Crack #${i + 1}</span></div>`)
        .join('');
    }
    if (cracksCol) cracksCol.style.display = 'block';
  } else if (cracksCol) {
    cracksCol.style.display = 'none';
  }

  const view = document.getElementById('latest-view');
  if (view) view.href = `/inspection/${rec.id}`;
  const report = document.getElementById('latest-report');
  if (report) report.href = `/reports/inspection/${rec.id}.pdf`;
}

async function refreshLatest() {
  try {
    const res = await fetch('/api/inspection/latest');
    renderLatest(await res.json());
  } catch (err) {
    /* leave last known result on screen */
  }
}

function refreshAll() {
  refreshNetwork();
  refreshPovStatus();
  refreshLatest();
}

refreshAll();
setInterval(refreshAll, REFRESH_INTERVAL_MS);
