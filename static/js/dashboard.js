/**
 * dashboard.js - MATANGLAWIN frontend logic.
 *
 * Responsibilities:
 *   - Loading screen fade-out on first successful /status poll or timeout.
 *   - Smooth scrolling for navigation anchor links.
 *   - Poll /status every 1000ms: update hero badge, live indicator.
 *   - Poll /capture/status every 1500ms: render crack analysis, manage alerts.
 *   - Camera source selector: POST to /set_source on change.
 *   - Mobile nav toggle handler.
 *   - Alert sound ONLY on analysis_complete state (confirmed crack), with 5s cooldown and auto-recovery.
 *   - Blinking crack alert banner on confirmed crack detection.
 *   - Scroll-based nav link highlighting.
 */

(function () {
  "use strict";

  var STATUS_POLL_MS = 1000;
  var ANALYSIS_POLL_MS = 1500;
  var ALERT_COOLDOWN_MS = 5000;

  // Elements
  var loadingScreen = document.getElementById("loadingScreen");
  var heroStatusBadge = document.getElementById("heroStatusBadge");
  var heroBadgeText = document.getElementById("heroBadgeText");
  var liveDot = document.getElementById("liveDot");
  var analysisStatus = document.getElementById("analysisStatus");
  var analysisContent = document.getElementById("analysisContent");
  var sourceOptions = document.getElementById("sourceOptions");
  var navToggle = document.getElementById("navToggle");
  var navLinks = document.querySelectorAll(".nav-link");
  var videoFeed = document.getElementById("videoFeed");
  var crackAlertBanner = document.getElementById("crackAlertBanner");

  var loadingDismissed = false;
  var cameraWasConnected = false;

  // -----------------------------------------------------------------------
  // Audio Manager - handles alert sound with cooldown and auto-recovery
  // -----------------------------------------------------------------------

  var AudioManager = {
    _audio: null,
    _lastPlayTime: 0,
    _alertFiredForCurrentCapture: false,

    init: function () {
      this._createAudio();
    },

    _createAudio: function () {
      try {
        var el = document.getElementById("alertSound");
        if (el) {
          this._audio = el;
        } else {
          this._audio = new Audio("/static/sounds/alert.wav");
          this._audio.preload = "auto";
        }
      } catch (e) {
        this._audio = null;
      }
    },

    play: function () {
      var now = Date.now();
      // Enforce cooldown
      if (now - this._lastPlayTime < ALERT_COOLDOWN_MS) {
        return;
      }
      // Prevent duplicate for same capture
      if (this._alertFiredForCurrentCapture) {
        return;
      }

      this._lastPlayTime = now;
      this._alertFiredForCurrentCapture = true;

      if (!this._audio) {
        this._createAudio();
      }

      try {
        this._audio.currentTime = 0;
        var playPromise = this._audio.play();
        if (playPromise && typeof playPromise.catch === "function") {
          var self = this;
          playPromise.catch(function () {
            // Auto-recovery: recreate audio element on failure
            self._audio = null;
            self._createAudio();
          });
        }
      } catch (err) {
        // Auto-recovery: recreate audio element
        this._audio = null;
        this._createAudio();
      }
    },

    resetForNewCapture: function () {
      this._alertFiredForCurrentCapture = false;
    }
  };

  // -----------------------------------------------------------------------
  // Loading screen
  // -----------------------------------------------------------------------

  function dismissLoading() {
    if (loadingDismissed) return;
    loadingDismissed = true;
    loadingScreen.classList.add("fade-out");
    setTimeout(function () {
      loadingScreen.style.display = "none";
    }, 700);
  }

  // -----------------------------------------------------------------------
  // Smooth scrolling for nav links
  // -----------------------------------------------------------------------

  function setupSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
      anchor.addEventListener("click", function (e) {
        var targetId = this.getAttribute("href");
        if (!targetId || targetId === "#") return;
        var target = document.querySelector(targetId);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: "smooth" });
          // Close mobile nav if open
          if (navToggle) navToggle.checked = false;
        }
      });
    });
  }

  // -----------------------------------------------------------------------
  // Scroll-based nav highlighting
  // -----------------------------------------------------------------------

  function updateActiveNav() {
    var sections = document.querySelectorAll("section[id]");
    var scrollPos = window.scrollY + 120;

    sections.forEach(function (section) {
      var top = section.offsetTop;
      var height = section.offsetHeight;
      var id = section.getAttribute("id");

      if (scrollPos >= top && scrollPos < top + height) {
        navLinks.forEach(function (link) {
          link.classList.remove("active");
          if (link.getAttribute("href") === "#" + id) {
            link.classList.add("active");
          }
        });
      }
    });
  }

  // -----------------------------------------------------------------------
  // Crack Alert Banner
  // -----------------------------------------------------------------------

  function showAlertBanner() {
    if (crackAlertBanner) {
      crackAlertBanner.classList.add("active");
    }
  }

  function hideAlertBanner() {
    if (crackAlertBanner) {
      crackAlertBanner.classList.remove("active");
    }
  }

  // -----------------------------------------------------------------------
  // Status polling
  // -----------------------------------------------------------------------

  function applyStatus(data) {
    var cameraConnected = Boolean(data.camera_connected);
    var crackPresent = Boolean(data.crack_present);

    // Reconnect MJPEG feed when camera recovers
    if (cameraConnected && !cameraWasConnected && videoFeed) {
      videoFeed.src = "/video_feed?" + Date.now();
    }
    cameraWasConnected = cameraConnected;

    // Live indicator
    if (liveDot) {
      if (cameraConnected) {
        liveDot.classList.add("online");
      } else {
        liveDot.classList.remove("online");
      }
    }

    // Hero status badge
    if (heroStatusBadge && heroBadgeText) {
      if (crackPresent) {
        heroStatusBadge.classList.add("crack-detected");
        heroBadgeText.textContent = "CRACK DETECTED";
      } else {
        heroStatusBadge.classList.remove("crack-detected");
        heroBadgeText.textContent = "MONITORING...";
      }
    }
  }

  async function pollStatus() {
    try {
      var res = await fetch("/status", { cache: "no-store" });
      if (!res.ok) throw new Error("bad response");
      var data = await res.json();
      applyStatus(data);
      if (!loadingDismissed) dismissLoading();
    } catch (err) {
      if (liveDot) liveDot.classList.remove("online");
      cameraWasConnected = false;
    }
  }

  // -----------------------------------------------------------------------
  // Capture status polling
  // -----------------------------------------------------------------------

  var thresholdSlider = document.getElementById("thresholdSlider");
  var thresholdLabel = document.getElementById("thresholdLabel");

  var lastCaptureTimestamp = null;
  var thresholdSyncedFromBackend = false;
  var lastState = "monitoring";

  var STATE_LABELS = {
    monitoring: "Monitoring",
    threshold_reached: "Threshold Met",
    capturing: "Capturing...",
    analyzing: "Analyzing...",
    analysis_complete: "Complete",
    cooldown: "Cooldown"
  };

  function getConfidenceColor(confidence) {
    if (confidence < 50) return "green";
    if (confidence < 75) return "yellow";
    return "red";
  }

  function getClassificationClass(classification) {
    if (!classification) return "";
    var lower = classification.toLowerCase();
    if (lower.indexOf("structural") !== -1) return "structural";
    if (lower.indexOf("surface") !== -1) return "surface-level";
    if (lower.indexOf("hairline") !== -1) return "hairline";
    return "structural";
  }

  function updateAnalysisStatus(state) {
    if (!analysisStatus) return;
    var label = STATE_LABELS[state] || "Monitoring";
    analysisStatus.textContent = label;

    // Remove all state classes
    analysisStatus.classList.remove(
      "alert",
      "status-monitoring",
      "status-threshold",
      "status-capturing",
      "status-analyzing",
      "status-complete",
      "status-cooldown"
    );

    switch (state) {
      case "monitoring":
        analysisStatus.classList.add("status-monitoring");
        break;
      case "threshold_reached":
        analysisStatus.classList.add("alert", "status-threshold");
        break;
      case "capturing":
        analysisStatus.classList.add("alert", "status-capturing");
        break;
      case "analyzing":
        analysisStatus.classList.add("status-analyzing");
        break;
      case "analysis_complete":
        analysisStatus.classList.add("status-complete");
        break;
      case "cooldown":
        analysisStatus.classList.add("status-cooldown");
        break;
    }
  }

  function renderCaptureAnalysis(analysis) {
    if (!analysisContent) return;

    var confidence = (analysis.confidence || 0) * 100;
    var classification = analysis.classification || "Unknown";
    var timestamp = analysis.timestamp || "";
    var colorClass = getConfidenceColor(confidence);
    var classClass = getClassificationClass(classification);

    var html = "";
    html += '<div class="capture-image-container">';
    html +=
      '<img class="capture-image" src="/capture/image?t=' +
      encodeURIComponent(timestamp) +
      '" alt="Captured crack analysis">';
    html += "</div>";
    html += '<div class="crack-entry">';
    html +=
      '<span class="crack-classification ' +
      classClass +
      '">' +
      classification +
      "</span>";
    html += '<div class="severity-row">';
    html += '<div class="severity-label">';
    html += "<span>Confidence</span>";
    html +=
      '<span class="severity-value">' +
      Math.round(confidence) +
      "%</span>";
    html += "</div>";
    html += '<div class="severity-bar">';
    html +=
      '<div class="severity-fill ' +
      colorClass +
      '" style="width: ' +
      confidence +
      '%"></div>';
    html += "</div>";
    html += "</div>";
    if (timestamp) {
      html +=
        '<div class="capture-timestamp">Captured: ' +
        timestamp +
        "</div>";
    }
    html += "</div>";

    analysisContent.innerHTML = html;
  }

  async function pollAnalysis() {
    try {
      var res = await fetch("/capture/status", { cache: "no-store" });
      if (!res.ok) throw new Error("bad response");
      var data = await res.json();

      // Sync threshold slider from backend on first successful poll
      if (!thresholdSyncedFromBackend && data.threshold != null) {
        thresholdSyncedFromBackend = true;
        var backendPercent = Math.round(data.threshold * 100);
        if (thresholdSlider) {
          thresholdSlider.value = backendPercent;
        }
        if (thresholdLabel) {
          thresholdLabel.textContent = backendPercent + "%";
        }
      }

      var state = data.state || "monitoring";
      updateAnalysisStatus(state);

      // Alert sound and banner: ONLY on analysis_complete (confirmed crack)
      if (state === "analysis_complete") {
        showAlertBanner();
        AudioManager.play();
      }

      // Hide banner when state returns to monitoring or cooldown
      if (state === "monitoring" || state === "cooldown") {
        hideAlertBanner();
        AudioManager.resetForNewCapture();
      }

      // Track state transitions
      lastState = state;

      // Render capture results or empty state
      if (state === "analysis_complete" && data.latest_analysis) {
        var ts = data.latest_analysis.timestamp;
        if (ts !== lastCaptureTimestamp) {
          lastCaptureTimestamp = ts;
          renderCaptureAnalysis(data.latest_analysis);
        }
      } else if (
        state !== "analysis_complete" &&
        !lastCaptureTimestamp
      ) {
        // No capture has occurred yet - show monitoring state
        if (analysisContent) {
          analysisContent.innerHTML =
            '<div class="analysis-empty"><p>No cracks detected - Monitoring...</p></div>';
        }
      }
    } catch (err) {
      /* ignore - status polling will handle connectivity */
    }
  }

  // -----------------------------------------------------------------------
  // Camera source selector
  // -----------------------------------------------------------------------

  function setupSourceSelector() {
    if (!sourceOptions) return;
    sourceOptions.addEventListener("change", async function (event) {
      var target = event.target;
      if (target && target.name === "source") {
        try {
          await fetch("/set_source", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: target.value }),
          });
        } catch (err) {
          /* ignore */
        }
      }
    });
  }

  // -----------------------------------------------------------------------
  // Threshold slider
  // -----------------------------------------------------------------------

  function setupThresholdSlider() {
    if (!thresholdSlider || !thresholdLabel) return;

    thresholdSlider.addEventListener("input", function () {
      thresholdLabel.textContent = thresholdSlider.value + "%";
    });

    thresholdSlider.addEventListener("change", async function () {
      var value = parseFloat(thresholdSlider.value) / 100;
      try {
        await fetch("/capture/threshold", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ threshold: value }),
        });
      } catch (err) {
        /* ignore */
      }
    });
  }

  // -----------------------------------------------------------------------
  // Mobile nav toggle
  // -----------------------------------------------------------------------

  function setupMobileNav() {
    // The CSS checkbox hack handles the toggle via label,
    // but we also close the menu when a link is clicked.
    // This is handled in setupSmoothScroll above.
  }

  // -----------------------------------------------------------------------
  // Init
  // -----------------------------------------------------------------------

  function init() {
    AudioManager.init();
    setupSmoothScroll();
    setupSourceSelector();
    setupThresholdSlider();
    setupMobileNav();

    // Start polling
    pollStatus();
    pollAnalysis();
    setInterval(pollStatus, STATUS_POLL_MS);
    setInterval(pollAnalysis, ANALYSIS_POLL_MS);

    // Scroll-based nav highlighting
    window.addEventListener("scroll", updateActiveNav);
    updateActiveNav();

    // Safety: dismiss loading after 4s timeout
    setTimeout(function () {
      if (!loadingDismissed) dismissLoading();
    }, 4000);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
