/**
 * dashboard.js - Drives the display-only inspection dashboard.
 *
 * Two independent concerns, matching the backend's two independent pipelines:
 *
 *   1. Live Drone POV (Pipeline A) - the raw MediaMTX WebRTC feed, loaded
 *      directly into an <iframe>. The backend never proxies, re-encodes,
 *      or annotates it, and NO detection ever runs on it. A LIVE/OFFLINE/
 *      UNKNOWN badge is derived purely from MediaMTX's own publisher state
 *      (/api/stream/status) - never from reading video frames.
 *
 *   2. Latest Inspection (Pipeline B) - the most recent PHOTO analysis
 *      result (/api/inspection/latest): status + original + red-highlighted
 *      image. Polled continuously so a freshly captured DJI photo appears
 *      automatically, with NO page refresh and NO Analyze click. Confidence,
 *      counts, boxes, and other metrics are never shown.
 */

const REFRESH_INTERVAL_MS = 3000;
let webrtcUrl = null;
let lastInspectionId = null;
let seenAnyInspection = false;
let toastTimer = null;

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// ---------------------------------------------------------------- network
async function refreshNetwork() {
  try {
    const res = await fetch('/api/network');
    const info = await res.json();
    setText('net-host-ip', info.host_ip || 'Unavailable (no network detected)');
    setText('net-rtmp-address', info.rtmp_address || 'Connect to a network first');
    setText('net-stream-key', info.stream_key || '\u2014');
    // Remember the WebRTC URL; the iframe is only actually loaded once a
    // publisher is confirmed LIVE (see setPovBadge), so an offline stream
    // never shows a broken player - the clean dark placeholder stays up.
    webrtcUrl = info.webrtc_url || null;
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

// -------------------------------------------------------------- POV badge
function setPovBadge(state) {
  const badge = document.getElementById('stream-badge');
  const placeholder = document.getElementById('pov-placeholder');
  const iframe = document.getElementById('pov-iframe');
  if (!badge) return;

  badge.textContent = state;
  badge.className = 'badge ' + (
    state === 'LIVE' ? 'badge-live' : state === 'OFFLINE' ? 'badge-offline' : 'badge-connecting'
  );

  // Only load/show the live player when MediaMTX confirms a publisher is
  // actually LIVE. In OFFLINE/UNKNOWN we keep the clean dark placeholder
  // (and unload the iframe) so the operator never sees a broken player.
  const showFeed = state === 'LIVE' && !!webrtcUrl;
  if (iframe) {
    if (showFeed) {
      if (iframe.src !== webrtcUrl) iframe.src = webrtcUrl;
      iframe.style.display = 'block';
    } else {
      if (iframe.src) iframe.removeAttribute('src');
      iframe.style.display = 'none';
    }
  }
  if (placeholder) placeholder.style.display = showFeed ? 'none' : 'flex';
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

// ------------------------------------------------- photo bridge status
async function refreshBridgeStatus() {
  const badge = document.getElementById('bridge-badge');
  if (!badge) return;
  let state = 'OFFLINE';
  try {
    const res = await fetch('/api/bridge/status');
    const info = await res.json();
    state = info.state || 'OFFLINE';
  } catch (err) {
    state = 'OFFLINE';
  }
  badge.textContent = 'PHOTO BRIDGE: ' + state;
  badge.className = 'badge ' + (
    state === 'READY' ? 'badge-live'
      : state === 'OFFLINE' ? 'badge-offline'
      : 'badge-connecting'  // WAITING FOR PHOTO
  );
}

// -------------------------------------------------- latest inspection (B)
function showToast(message) {
  const toast = document.getElementById('latest-toast');
  if (!toast) return;
  toast.textContent = message;
  toast.style.display = 'block';
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.style.display = 'none'; }, 6000);
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
  if (content) content.style.display = 'grid';

  const banner = document.getElementById('latest-status');
  if (banner) {
    banner.textContent = rec.status;
    banner.className = 'status-banner ' + (rec.has_crack ? 'status-crack' : 'status-ok');
  }
  setText('latest-timestamp', rec.timestamp);
  setText('latest-source', rec.source_label || (rec.source === 'import' ? 'DJI import' : 'Manual upload'));

  const urls = rec.urls || {};
  const original = document.getElementById('latest-original');
  // Cache-bust per inspection id so the browser always shows the new photo.
  if (original && urls.original) original.src = urls.original + '?v=' + rec.id;

  const hlCol = document.getElementById('latest-highlighted-col');
  const hl = document.getElementById('latest-highlighted');
  if (rec.has_crack && urls.highlighted) {
    if (hl) hl.src = urls.highlighted + '?v=' + rec.id;
    if (hlCol) hlCol.style.display = 'block';
  } else if (hlCol) {
    hlCol.style.display = 'none';
  }

  const view = document.getElementById('latest-view');
  if (view) view.href = `/inspection/${rec.id}`;

  // Automatic-update announcement: a new inspection arrived on its own.
  if (rec.id !== lastInspectionId) {
    if (seenAnyInspection && lastInspectionId !== null) {
      const via = rec.source === 'import' ? 'DJI capture' : 'manual upload';
      showToast(`Analysis complete \u2014 new inspection #${rec.id} (${via}): ${rec.status}`);
    }
    lastInspectionId = rec.id;
    seenAnyInspection = true;
  }
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
  refreshBridgeStatus();
  refreshLatest();
}

refreshAll();
setInterval(refreshAll, REFRESH_INTERVAL_MS);
