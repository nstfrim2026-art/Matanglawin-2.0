/**
 * dashboard.js - MATANGLAWIN live dashboard frontend logic.
 *
 * Responsibilities only:
 *   - Fade the loading screen into the dashboard once ready.
 *   - Poll /status and reflect camera-connected / crack-present state:
 *       - LIVE indicator (green while camera connected)
 *       - Status panel text (Monitoring... / Potential Crack Detected)
 *       - Alert banner show/hide (blinking while a crack is visible)
 *       - Alert sound played ONCE per rising edge (crack appears),
 *         not on every poll while it stays present.
 *   - Let the operator switch camera source (webcam / droidcam / drone)
 *     via a simple POST to /set_source; the backend/video pipeline does
 *     the rest.
 *
 * No confidence scores, measurements, or analytics are shown - this file
 * intentionally only toggles the two states described above.
 */

(function () {
  "use strict";

  const POLL_INTERVAL_MS = 1000;

  const loadingScreen = document.getElementById("loadingScreen");
  const dashboard = document.getElementById("dashboard");
  const liveDot = document.getElementById("liveDot");
  const alertBanner = document.getElementById("alertBanner");
  const statusIndicator = document.getElementById("statusIndicator");
  const statusText = document.getElementById("statusText");
  const alertSound = document.getElementById("alertSound");
  const sourceOptions = document.getElementById("sourceOptions");

  let crackWasPresent = false;

  function showDashboard() {
    loadingScreen.classList.add("fade-out");
    dashboard.classList.add("visible");
    setTimeout(() => {
      loadingScreen.style.display = "none";
    }, 650);
  }

  function applyStatus(data) {
    const cameraConnected = Boolean(data.camera_connected);
    const crackPresent = Boolean(data.crack_present);

    // LIVE indicator.
    liveDot.classList.toggle("online", cameraConnected);
    liveDot.classList.toggle("offline", !cameraConnected);

    // Status panel + alert banner.
    if (crackPresent) {
      statusIndicator.classList.remove("monitoring");
      statusIndicator.classList.add("alert");
      statusText.textContent = "Potential Crack Detected";

      alertBanner.classList.add("visible");
      alertBanner.setAttribute("aria-hidden", "false");
    } else {
      statusIndicator.classList.remove("alert");
      statusIndicator.classList.add("monitoring");
      statusText.textContent = "Monitoring...";

      alertBanner.classList.remove("visible");
      alertBanner.setAttribute("aria-hidden", "true");
    }

    // Play the alert sound only on the rising edge (crack just appeared),
    // never continuously while it stays present.
    if (crackPresent && !crackWasPresent) {
      playAlertSound();
    }
    crackWasPresent = crackPresent;
  }

  function playAlertSound() {
    try {
      alertSound.currentTime = 0;
      const playPromise = alertSound.play();
      if (playPromise && typeof playPromise.catch === "function") {
        // Autoplay may be blocked until the user interacts with the page
        // at least once - this is expected browser behavior, not a bug.
        playPromise.catch(() => {});
      }
    } catch (err) {
      /* ignore playback errors */
    }
  }

  async function pollStatus() {
    try {
      const res = await fetch("/status", { cache: "no-store" });
      if (!res.ok) throw new Error("bad status response");
      const data = await res.json();
      applyStatus(data);
      // First successful poll: reveal the dashboard.
      if (loadingScreen.style.display !== "none") {
        showDashboard();
      }
    } catch (err) {
      liveDot.classList.remove("online");
      liveDot.classList.add("offline");
    }
  }

  function setupSourceSelector() {
    if (!sourceOptions) return;
    sourceOptions.addEventListener("change", async (event) => {
      const target = event.target;
      if (target && target.name === "source") {
        try {
          await fetch("/set_source", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: target.value }),
          });
        } catch (err) {
          /* ignore - status polling will reflect connection state */
        }
      }
    });
  }

  function init() {
    setupSourceSelector();
    pollStatus();
    setInterval(pollStatus, POLL_INTERVAL_MS);

    // Safety net: always reveal the dashboard after a short delay even if
    // the very first status poll is slow, so the operator isn't stuck on
    // the loading screen.
    setTimeout(() => {
      if (loadingScreen.style.display !== "none") {
        showDashboard();
      }
    }, 4000);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
