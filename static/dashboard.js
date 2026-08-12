/**
 * dashboard.js - Polls the backend for network config, stream status, and
 * the latest capture, and updates the dashboard UI accordingly.
 *
 * The live drone POV video itself is loaded directly from MediaMTX's own
 * WebRTC endpoint (network.webrtc_url) inside an <iframe> - the Flask
 * backend never proxies or re-encodes the video stream. This keeps the
 * "no OBS / no screen mirroring / video comes straight from MediaMTX"
 * requirement true at the browser level, not just on paper.
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
    setText('net-rtsp-url', info.rtsp_url || '\u2014');
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

function setStreamBadge(state) {
  const badge = document.getElementById('stream-badge');
  const placeholder = document.getElementById('pov-placeholder');
  const iframe = document.getElementById('pov-iframe');
  if (!badge) return;

  badge.textContent = state;
  badge.className = 'badge ' + (
    state === 'LIVE' ? 'badge-live' : state === 'CONNECTING' ? 'badge-connecting' : 'badge-offline'
  );

  if (state === 'LIVE') {
    if (placeholder) placeholder.style.display = 'none';
    if (iframe) iframe.style.display = 'block';
  } else {
    if (placeholder) placeholder.style.display = 'block';
    if (iframe) iframe.style.display = 'none';
  }
}

async function refreshStreamStatus() {
  try {
    const res = await fetch('/api/stream/status');
    const status = await res.json();
    setStreamBadge(status.stream_state || 'OFFLINE');
    setText('detector-status', status.detector_ready ? 'Ready' : (status.detector_error || 'Loading model...'));
    setText('frames-processed', String(status.frames_processed || 0));
    setText(
      'last-detection-at',
      status.last_detection_at
        ? new Date(status.last_detection_at * 1000).toLocaleTimeString()
        : 'None yet'
    );
  } catch (err) {
    setStreamBadge('OFFLINE');
  }
}

async function refreshLatestCapture() {
  try {
    const res = await fetch('/api/inspections/latest');
    const rec = await res.json();
    const img = document.getElementById('latest-capture-img');
    const empty = document.getElementById('latest-capture-empty');
    const reportLink = document.getElementById('latest-report-link');

    if (!rec) {
      if (img) img.style.display = 'none';
      if (empty) empty.style.display = 'block';
      if (reportLink) reportLink.style.display = 'none';
      return;
    }

    if (empty) empty.style.display = 'none';
    if (img) {
      const filename = (rec.cropped_image_path || '').split('/').pop();
      img.src = '/data/captures/' + filename;
      img.style.display = 'block';
    }
    setText('latest-confidence', (rec.confidence * 100).toFixed(1) + '%');
    setText('latest-timestamp', rec.timestamp);
    setText(
      'latest-gps',
      rec.gps_available ? `${rec.latitude.toFixed(5)}, ${rec.longitude.toFixed(5)}` : 'Unavailable'
    );
    if (reportLink) {
      reportLink.href = `/reports/inspection/${rec.id}.pdf`;
      reportLink.style.display = 'inline-block';
    }
  } catch (err) {
    // Leave the last known values on screen; this is expected while the
    // drone/backend is offline.
  }
}

function refreshAll() {
  refreshNetwork();
  refreshStreamStatus();
  refreshLatestCapture();
}

refreshAll();
setInterval(refreshAll, REFRESH_INTERVAL_MS);
