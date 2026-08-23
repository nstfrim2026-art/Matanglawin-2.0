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
// We only need the WebRTC URL for the live-feed iframe; the operator-facing
// connection panel was removed, so nothing else is displayed here.
async function refreshWebrtcUrl() {
  try {
    const res = await fetch('/api/network');
    const info = await res.json();
    webrtcUrl = info.webrtc_url || null;
  } catch (err) {
    /* offline-safe: keep the clean placeholder */
  }
}

// -------------------------------------------------- crack beep + alert
// A short WebAudio beep (no audio file needed - fully offline) plus a
// prominent, auto-hiding notification. Fires once per NEWLY completed crack
// inspection - never per refresh, never when there is no crack.
let audioCtx = null;
function _unlockAudio() {
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();
  } catch (e) { /* audio unavailable */ }
}
document.addEventListener('click', _unlockAudio);
document.addEventListener('keydown', _unlockAudio);

function beep() {
  try {
    _unlockAudio();
    if (!audioCtx) return;
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    o.type = 'square';
    o.frequency.value = 880;
    o.connect(g); g.connect(audioCtx.destination);
    const t = audioCtx.currentTime;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.25, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.18);
    o.start(t);
    o.stop(t + 0.2);
  } catch (e) {
    /* audio blocked (no user gesture yet) - the visual alert still shows */
  }
}

let crackAlertTimer = null;
function showCrackAlert(rec) {
  const el = document.getElementById('crack-alert');
  if (!el) return;
  const sub = document.getElementById('crack-alert-sub');
  if (sub) sub.textContent = 'Inspection #' + rec.id;
  el.style.display = 'flex';
  if (crackAlertTimer) clearTimeout(crackAlertTimer);
  crackAlertTimer = setTimeout(() => { el.style.display = 'none'; }, 6000);
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

  // GPS location of the capture (shown once the photo has been processed).
  // Only present when an aircraft/phone GPS sample was matched at capture time.
  const hasGps = rec.gps_available && rec.latitude != null && rec.longitude != null;
  setText('latest-gps', hasGps
    ? `${rec.latitude.toFixed(6)}, ${rec.longitude.toFixed(6)}`
    : 'Not recorded');

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
  // The block runs once per NEW inspection id, so a crack beeps exactly once
  // (never on a refresh of the same result, never when there's no crack).
  if (rec.id !== lastInspectionId) {
    if (seenAnyInspection && lastInspectionId !== null) {
      const via = rec.source === 'import' ? 'DJI capture' : 'manual upload';
      showToast(`Analysis complete \u2014 new inspection #${rec.id} (${via}): ${rec.status}`);
      if (rec.has_crack) {
        beep();
        showCrackAlert(rec);
      }
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
  refreshWebrtcUrl();
  refreshPovStatus();
  refreshBridgeStatus();
  refreshLatest();
}

refreshAll();
setInterval(refreshAll, REFRESH_INTERVAL_MS);
