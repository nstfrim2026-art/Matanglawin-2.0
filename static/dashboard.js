/**
 * dashboard.js - Drives the clean, display-only Live Inspection Dashboard.
 *
 * The dashboard has exactly three concerns:
 *
 *   1. Live Drone POV (display only) - the raw MediaMTX WebRTC feed in an
 *      <iframe>. The backend never proxies, re-encodes, or annotates it, and
 *      NO detection ever runs on it (no masks/boxes/labels/confidence). A
 *      LIVE/OFFLINE badge is derived purely from MediaMTX's own publisher
 *      state (/api/stream/status), never from reading frames.
 *
 *   2. Capture Photo - POSTs /api/capture. The capture service grabs the
 *      current MediaMTX frame, saves a JPEG, associates the latest usable
 *      phone GPS, runs best.pt once, and creates one inspection. No manual
 *      upload, no Snipping Tool, no separate Analyze step.
 *
 *   3. Automatic crack alert - a short beep + visible banner that fires ONCE
 *      per newly created crack inspection (never on refresh, never for a
 *      clear result). Inspection counts, history, results, and the map live
 *      in the Inspection section, not here.
 *
 * Connection failures are treated as NORMAL: the live area falls back to a
 * clean "No connection" and reconnects automatically with a bounded backoff.
 * A dropped stream never crashes the page and never touches inspection/GPS/
 * map history (those are served by independent endpoints).
 */

const REFRESH_INTERVAL_MS = 3000;
let webrtcUrl = null;
let lastInspectionId = null;
let seenAnyInspection = false;

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

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
// Internal stream-health monitoring (requirement 4C): last successful frame
// time, current status, reconnect attempts, last error. Kept for debugging;
// the operator only ever sees a clean LIVE / "No connection" state.
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
// probe, but never hammer the network. The steady poll also calls this, so a
// stream that comes back is picked up automatically.
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
    // OFFLINE / UNKNOWN (or LIVE but URL not resolved yet): clean placeholder
    // and keep trying to reconnect on a bounded backoff.
    setPovOffline(state === 'UNKNOWN' ? 'CONNECTING' : 'OFFLINE');
    scheduleReconnect();
  }
}

// A live player that fails to load must not leave a broken frame: fall back to
// the clean placeholder and let the reconnect loop bring it back.
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
let capturing = false;
function setCaptureStatus(text, kind) {
  const el = document.getElementById('capture-status');
  if (!el) return;
  el.textContent = text || '';
  el.className = 'capture-status' + (kind ? ' capture-status-' + kind : '');
}

async function onCapture() {
  if (capturing) return;
  capturing = true;
  const btn = document.getElementById('capture-btn');
  if (btn) btn.disabled = true;
  _unlockAudio();  // first user gesture -> allow the alert beep to play
  setCaptureStatus('Capturing\u2026', 'busy');
  try {
    const res = await fetch('/api/capture', { method: 'POST' });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.ok === false) {
      // Clean, human-readable error - never a stack trace. The website,
      // server, MediaMTX, and inspection history are all unaffected.
      const detail = data.detail || 'Live feed unavailable';
      setCaptureStatus('Capture failed: ' + detail, 'error');
    } else {
      const link = '<a href="/inspection/' + data.id + '">#' + data.id + '</a>';
      setCaptureStatus('', null);
      const el = document.getElementById('capture-status');
      if (el) {
        el.className = 'capture-status ' + (data.has_crack ? 'capture-status-crack' : 'capture-status-ok');
        el.innerHTML = data.status + ' \u2014 inspection ' + link +
          ' \u00B7 <a href="/inspections">view in Inspections</a>';
      }
      // Refresh the latest poll immediately so the crack alert/beep fires now.
      refreshLatestForAlert();
    }
  } catch (err) {
    setCaptureStatus('Capture failed: network error', 'error');
  } finally {
    capturing = false;
    if (btn) btn.disabled = false;
  }
}

// -------------------------------------------- crack alert (latest poll)
// We poll the latest inspection ONLY to drive the once-per-new-crack alert.
// The full result / counts / history live in the Inspection section.
function handleLatestForAlert(rec) {
  if (!rec) return;
  if (rec.id !== lastInspectionId) {
    if (seenAnyInspection && lastInspectionId !== null && rec.has_crack) {
      beep();
      showCrackAlert(rec);
    }
    lastInspectionId = rec.id;
    seenAnyInspection = true;
  }
}

async function refreshLatestForAlert() {
  try {
    const res = await fetch('/api/inspection/latest');
    handleLatestForAlert(await res.json());
  } catch (err) {
    /* offline-safe */
  }
}

// ---------------------------------------------------------------- loop
function refreshAll() {
  refreshWebrtcUrl();
  refreshPovStatus();
  refreshLatestForAlert();
}

document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('capture-btn');
  if (btn) btn.addEventListener('click', onCapture);
});

refreshAll();
setInterval(refreshAll, REFRESH_INTERVAL_MS);
