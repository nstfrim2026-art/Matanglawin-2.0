// app.js - Frontend logic for the Matanglawin crack-detection website.
// Talks to the FastAPI backend's /api/detect endpoint.

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const uploadPrompt = document.getElementById("uploadPrompt");
const previewImg = document.getElementById("previewImg");

const confRange = document.getElementById("confRange");
const confVal = document.getElementById("confVal");
const alphaRange = document.getElementById("alphaRange");
const alphaVal = document.getElementById("alphaVal");
const colorPick = document.getElementById("colorPick");
const outlineRange = document.getElementById("outlineRange");
const outlineVal = document.getElementById("outlineVal");

const detectBtn = document.getElementById("detectBtn");
const statusEl = document.getElementById("status");
const resultPanel = document.getElementById("resultPanel");
const resultSummary = document.getElementById("resultSummary");
const resultImg = document.getElementById("resultImg");
const downloadLink = document.getElementById("downloadLink");

let selectedFile = null;

// --- Slider label sync -----------------------------------------------------
confRange.addEventListener("input", () => (confVal.textContent = confRange.value));
alphaRange.addEventListener("input", () => (alphaVal.textContent = alphaRange.value));
outlineRange.addEventListener("input", () => (outlineVal.textContent = outlineRange.value));

// --- File selection ---------------------------------------------------------
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) {
    handleFile(fileInput.files[0]);
  }
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showStatus("Please choose an image file.", true);
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
    uploadPrompt.hidden = true;
  };
  reader.readAsDataURL(file);
  detectBtn.disabled = false;
  hideStatus();
  resultPanel.hidden = true;
}

// --- Detection --------------------------------------------------------------
function hexToRgbString(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `${r},${g},${b}`;
}

function showStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.hidden = false;
  statusEl.classList.toggle("error", isError);
}

function hideStatus() {
  statusEl.hidden = true;
}

detectBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  detectBtn.disabled = true;
  showStatus("Running crack detection... this can take a few seconds.");

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("conf", confRange.value);
  formData.append("imgsz", "640");
  formData.append("alpha", alphaRange.value);
  formData.append("color", hexToRgbString(colorPick.value));
  formData.append("outline", outlineRange.value);

  try {
    const res = await fetch("/api/detect", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${res.status})`);
    }
    const data = await res.json();

    resultImg.src = data.image;
    downloadLink.href = data.image;
    resultSummary.textContent =
      data.detections > 0
        ? `Detected ${data.detections} crack instance(s) in a ${data.width}x${data.height} image.`
        : `No cracks detected in this ${data.width}x${data.height} image. Try lowering the confidence threshold.`;
    resultPanel.hidden = false;
    hideStatus();
  } catch (err) {
    showStatus(`Error: ${err.message}`, true);
  } finally {
    detectBtn.disabled = false;
  }
});
