/**
 * dashboard.js - Drives the Live Inspection Dashboard.
 *
 * Layout (matches the reference): the clean live drone POV on top, the
 * Capture Photo button directly under the video, and the Latest Inspection
 * section beneath it.
 *
 *   1. Live Drone POV (display only) - the raw MediaMTX WebRTC feed in an
 *      <iframe>. The backend never proxies, re-encodes, or annotates it, and
 *      NO detection ever runs on it (no masks/boxes/labels/confidence). The
 *      LIVE / "No connection" state comes from MediaMTX's own publisher state
 *      (/api/stream/status), never from reading frames.
 *
 *   2. Capture Photo - POSTs /api/capture. One press = one capture = one
 *      inspection. Double-clicks / simultaneous requests are prevented.
 *
 *   3. Latest Inspection - status + latitude/longitude/date-time/source +
 *      analyzed image, refreshed automatically after each capture. A crack
 *      beeps + alerts ONCE per new inspection (never on refresh/polling,
 *      never for a clear result).
 *
 * Connection failures are NORMAL: the live area falls back to a clean
 * "No connection" and reconnects automatically with a bounded backoff. A
 * dropped stream never reloads the page and never touches inspection history.
 */

const REFRESH_INTERVAL_MS = 3000;
let webrtcUrl = null;
let lastInspectionId = null;
let seenAnyInspection = false;

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// Coordinate + timestamp formatting.
function fmtLat(v) { return Math.abs(v).toFixed(6) + '\u00B0 ' + (v >= 0 ? 'N' : 'S'); }
function fmtLon(v) { return Math.abs(v).toFixed(6) + '\u00B0 ' + (v >= 0 ? 'E' : 'W'); }
function fmtDateTime(ts) { return ts ? String(ts).replace('T', ' ').slice(0, 16) : '\u2014'; }

// ---------------------------------------------------------------- network
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

// ----------------------------------------------- live POV + reconnection
// Internal stream-health monitoring (kept for debugging; the operator only
// ever sees a clean LIVE / "No connection" state).
const streamHealth = {
  status: 'UNKNOWN',
  lastFrameTs: null,
  reconnectAttempts: 0,
  lastError: null,
};
const RECONNECT_BASE_MS = 2000;
const RECONNECT_MAX_MS = 15000;
let reconnectTimer = null;

function setPovOffline(label) {
  const badge = document.getElementById('stream-badge');
  const placeholder = document.getElementById('pov-placeholder');
  const iframe = document.getElementById('pov-iframe');
  if (badge) {
    badge.textContent = label === 'CONNECTING' ? 'Connecting\u2026' : 'No connection';
    badge.className = 'badge ' + (label === 'CONNECTING' ? 'badge-connecting' : 'badge-offline');
  }
  if (iframe) {
    if (iframe.src) iframe.removeAttribute('src');
    iframe.style.display = 'none';
  }
  if (placeholder) placeholder.style.display = 'flex';
  setText('pov-placeholder-text', 'No connection');
}

function setPovLive() {
  const badge = document.getElementById('stream-badge');
  const placeholder = document.getElementById('pov-placeholder');
  const iframe = document.getElementById('pov-iframe');
  if (badge) { badge.textContent = 'LIVE'; badge.className = 'badge badge-live'; }
  if (iframe && webrtcUrl) {
    if (iframe.src !== webrtcUrl) iframe.src = webrtcUrl;   // (re)connect the player
    iframe.style.display = 'block';
  }
  if (placeholder) placeholder.style.display = (iframe && webrtcUrl) ? 'none' : 'flex';
  streamHealth.lastFrameTs = Date.now();
  streamHealth.reconnectAttempts = 0;
  streamHealth.lastError = null;
}

// Bounded exponential backoff: when disconnected we keep retrying the status
// probe, but never hammer the network or reload the page. The steady poll also
// calls this, so a stream that comes back is picked up automatically.
function scheduleReconnect() {
  if (reconnectTimer) return;
  const delay = Math.min(
    RECONNECT_BASE_MS * Math.pow(2, streamHealth.reconnectAttempts),
    RECONNECT_MAX_MS
  );
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    streamHealth.reconnectAttempts += 1;
    refreshPovStatus();
  }, delay);
}

async function refreshPovStatus() {
  let state = 'UNKNOWN';
  try {
    const res = await fetch('/api/stream/status');
    const status = await res.json();
    state = status.pov_state || 'UNKNOWN';
  } catch (err) {
    streamHealth.lastError = String(err);
    state = 'UNKNOWN';
  }
  streamHealth.status = state;

  if (state === 'LIVE' && webrtcUrl) {
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    setPovLive();
  } else {
    setPovOffline(state === 'UNKNOWN' ? 'CONNECTING' : 'OFFLINE');
    scheduleReconnect();
  }
}

// A live player that fails to load must not leave a broken frame.
(function wireIframeError() {
  const iframe = document.getElementById('pov-iframe');
  if (iframe) {
    iframe.addEventListener('error', () => {
      streamHealth.lastError = 'iframe load error';
      setPovOffline('OFFLINE');
      scheduleReconnect();
    });
  }
})();

// ---------------------------------------------------------- Capture Photo
// Button states: idle -> loading -> success/error -> (back to idle).
let capturing = false;
let captureResetTimer = null;

function setButtonState(state, labelText) {
  const btn = document.getElementById('capture-btn');
  const label = document.getElementById('capture-label');
  if (!btn) return;
  btn.classList.remove('is-idle', 'is-loading', 'is-success', 'is-error');
  btn.classList.add('is-' + state);
  btn.disabled = (state === 'loading');
  const icons = { idle: '\uD83D\uDCF7', loading: '\u23F3', success: '\u2713', error: '\u26A0' };
  const icon = btn.querySelector('.capture-icon');
  if (icon) icon.textContent = icons[state] || icons.idle;
  if (label) label.textContent = labelText;
}

function setCaptureStatus(text, kind) {
  const el = document.getElementById('capture-status');
  if (!el) return;
  el.textContent = text || '';
  el.className = 'capture-status' + (kind ? ' capture-status-' + kind : '');
}

function resetButtonSoon() {
  if (captureResetTimer) clearTimeout(captureResetTimer);
  captureResetTimer = setTimeout(() => setButtonState('idle', 'Capture Photo'), 2500);
}

async function onCapture() {
  // Prevent double-clicks / multiple simultaneous capture requests: one press
  // must equal exactly one capture and one inspection.
  if (capturing) return;
  capturing = true;
  if (captureResetTimer) clearTimeout(captureResetTimer);
  _unlockAudio();  // first user gesture -> allow the alert beep to play
  setButtonState('loading', 'Capturing\u2026');
  setCaptureStatus('', null);
  try {
    const res = await fetch('/api/capture', { method: 'POST' });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.ok === false) {
      const detail = data.detail || 'Live feed unavailable';
      setButtonState('error', 'Capture Failed');
      setCaptureStatus(detail, 'error');
      resetButtonSoon();
    } else {
      setButtonState('success', 'Photo Captured');
      const link = '<a href="/inspection/' + data.id + '">#' + data.id + '</a>';
      const el = document.getElementById('capture-status');
      if (el) {
        el.className = 'capture-status ' + (data.has_crack ? 'capture-status-crack' : 'capture-status-ok');
        el.innerHTML = data.status + ' \u2014 inspection ' + link;
      }
      // Update the Latest Inspection section + fire the alert immediately.
      refreshLatest();
      resetButtonSoon();
    }
  } catch (err) {
    setButtonState('error', 'Capture Failed');
    setCaptureStatus('Network error', 'error');
    resetButtonSoon();
  } finally {
    capturing = false;
  }
}

// -------------------------------------------- Latest Inspection + alert
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
    banner.textContent = rec.has_crack ? 'CRACK DETECTED' : 'CLEAR';
    banner.className = 'status-banner ' + (rec.has_crack ? 'status-crack' : 'status-ok');
  }

  const hasGps = rec.gps_available && rec.latitude != null && rec.longitude != null;
  setText('latest-lat', hasGps ? fmtLat(rec.latitude) : 'Not recorded');
  setText('latest-lon', hasGps ? fmtLon(rec.longitude) : 'Not recorded');
  setText('latest-datetime', fmtDateTime(rec.timestamp));
  setText('latest-source', rec.source_label || 'Capture');

  const urls = rec.urls || {};
  const img = document.getElementById('latest-result');
  const src = (rec.has_crack && urls.highlighted) ? urls.highlighted : urls.original;
  if (img && src) img.src = src + '?v=' + rec.id;   // cache-bust per inspection
  setText('latest-caption', rec.has_crack ? 'Analyzed photo (crack highlighted)' : 'Analyzed photo');

  const view = document.getElementById('latest-view');
  if (view) view.href = '/inspection/' + rec.id;

  // Fire the beep + visible alert ONCE per new inspection id (never on a
  // refresh of the same result, never for a clear result).
  if (rec.id !== lastInspectionId) {
    if (seenAnyInspection && lastInspectionId !== null && rec.has_crack) {
      beep();
      showCrackAlert(rec);
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
    /* offline-safe: leave the last known result on screen */
  }
}

// ---------------------------------------------------------------- loop
function refreshAll() {
  refreshWebrtcUrl();
  refreshPovStatus();
  refreshLatest();
}

document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('capture-btn');
  if (btn) btn.addEventListener('click', onCapture);
  setButtonState('idle', 'Capture Photo');
});

refreshAll();
setInterval(refreshAll, REFRESH_INTERVAL_MS);
