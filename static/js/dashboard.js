/**
 * dashboard.js - MATANGLAWIN frontend logic.
 *
 * Responsibilities:
 *   - Loading screen fade-out on first successful /status poll or timeout.
 *   - Smooth scrolling for navigation anchor links.
 *   - Poll /status every 1000ms: update hero badge, live indicator, alert state.
 *   - Poll /analysis every 1500ms: render crack entries in analysis panel.
 *   - Camera source selector: POST to /set_source on change.
 *   - Mobile nav toggle handler.
 *   - Alert sound on crack detection rising edge.
 *   - Scroll-based nav link highlighting.
 */

(function () {
  "use strict";

  const STATUS_POLL_MS = 1000;
  const ANALYSIS_POLL_MS = 1500;

  // Elements
  const loadingScreen = document.getElementById("loadingScreen");
  const heroStatusBadge = document.getElementById("heroStatusBadge");
  const heroBadgeText = document.getElementById("heroBadgeText");
  const liveDot = document.getElementById("liveDot");
  const analysisStatus = document.getElementById("analysisStatus");
  const analysisContent = document.getElementById("analysisContent");
  const alertSound = document.getElementById("alertSound");
  const sourceOptions = document.getElementById("sourceOptions");
  const navToggle = document.getElementById("navToggle");
  const navLinks = document.querySelectorAll(".nav-link");
  const videoFeed = document.getElementById("videoFeed");

  let crackWasPresent = false;
  let loadingDismissed = false;
  let cameraWasConnected = false;

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

    // Analysis status indicator
    if (analysisStatus) {
      if (crackPresent) {
        analysisStatus.textContent = "Alert";
        analysisStatus.classList.add("alert");
      } else {
        analysisStatus.textContent = "Monitoring";
        analysisStatus.classList.remove("alert");
      }
    }

    // Alert sound on rising edge
    if (crackPresent && !crackWasPresent) {
      playAlertSound();
    }
    crackWasPresent = crackPresent;
  }

  function playAlertSound() {
    try {
      alertSound.currentTime = 0;
      var playPromise = alertSound.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch(function () {});
      }
    } catch (err) {
      /* ignore playback errors */
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
  // Analysis polling
  // -----------------------------------------------------------------------

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

  function renderAnalysis(cracks) {
    if (!analysisContent) return;

    if (!cracks || cracks.length === 0) {
      analysisContent.innerHTML =
        '<div class="analysis-empty"><p>No cracks detected - Monitoring...</p></div>';
      return;
    }

    var html = "";
    cracks.forEach(function (crack) {
      var confidence = (crack.confidence || 0) * 100;
      var classification = crack.classification || "Unknown";
      var area = crack.area_px || 0;
      var length = crack.estimated_length_px || 0;
      var width = crack.estimated_width_px || 0;
      var colorClass = getConfidenceColor(confidence);
      var classClass = getClassificationClass(classification);

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
      html += '<div class="crack-dimensions">';
      html +=
        '<span class="crack-dim-item">Length: <span>' +
        Math.round(length) +
        "px</span></span>";
      html +=
        '<span class="crack-dim-item">Width: <span>' +
        Math.round(width) +
        "px</span></span>";
      html +=
        '<span class="crack-dim-item">Area: <span>' +
        Math.round(area) +
        "px</span></span>";
      html += "</div>";
      html += "</div>";
    });

    analysisContent.innerHTML = html;
  }

  async function pollAnalysis() {
    try {
      var res = await fetch("/analysis", { cache: "no-store" });
      if (!res.ok) throw new Error("bad response");
      var data = await res.json();
      renderAnalysis(data);
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
    setupSmoothScroll();
    setupSourceSelector();
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
