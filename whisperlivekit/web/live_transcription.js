/* Theme, WebSocket, recording, rendering logic extracted from inline script and adapted for segmented theme control and WS caption */

let isRecording = false;
let websocket = null;
let recorder = null;
let chunkDuration = 100;
let websocketUrl = "ws://localhost:8000/asr";
let userClosing = false;
let wakeLock = null;
let startTime = null;
let timerInterval = null;
let audioContext = null;
let analyser = null;
let microphone = null;
let waveCanvas = document.getElementById("waveCanvas");
let waveCtx = waveCanvas.getContext("2d");
let animationFrame = null;
let waitingForStop = false;
let lastReceivedData = null;
let lastSignature = null;

waveCanvas.width = 60 * (window.devicePixelRatio || 1);
waveCanvas.height = 30 * (window.devicePixelRatio || 1);
waveCtx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);

const statusText = document.getElementById("status");
const recordButton = document.getElementById("recordButton");
const chunkSelector = document.getElementById("chunkSelector");
const websocketInput = document.getElementById("websocketInput");
const websocketDefaultSpan = document.getElementById("wsDefaultUrl");
const linesTranscriptDiv = document.getElementById("linesTranscript");
const timerElement = document.querySelector(".timer");
const themeRadios = document.querySelectorAll('input[name="theme"]');

function getWaveStroke() {
  const styles = getComputedStyle(document.documentElement);
  const v = styles.getPropertyValue("--wave-stroke").trim();
  return v || "#000";
}

let waveStroke = getWaveStroke();
function updateWaveStroke() {
  waveStroke = getWaveStroke();
}

function applyTheme(pref) {
  if (pref === "light") {
    document.documentElement.setAttribute("data-theme", "light");
  } else if (pref === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  updateWaveStroke();
}

// Persisted theme preference
const savedThemePref = localStorage.getItem("themePreference") || "system";
applyTheme(savedThemePref);
if (themeRadios.length) {
  themeRadios.forEach((r) => {
    r.checked = r.value === savedThemePref;
    r.addEventListener("change", () => {
      if (r.checked) {
        localStorage.setItem("themePreference", r.value);
        applyTheme(r.value);
      }
    });
  });
}

// React to OS theme changes when in "system" mode
const darkMq = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)");
const handleOsThemeChange = () => {
  const pref = localStorage.getItem("themePreference") || "system";
  if (pref === "system") updateWaveStroke();
};
if (darkMq && darkMq.addEventListener) {
  darkMq.addEventListener("change", handleOsThemeChange);
} else if (darkMq && darkMq.addListener) {
  // deprecated, but included for Safari compatibility
  darkMq.addListener(handleOsThemeChange);
}

// Helpers
function fmt1(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n.toFixed(1) : x;
}

// Default WebSocket URL computation
const host = window.location.hostname || "localhost";
const port = window.location.port;
const protocol = window.location.protocol === "https:" ? "wss" : "ws";
const defaultWebSocketUrl = `${protocol}://${host}${port ? ":" + port : ""}/asr`;

// Populate default caption and input
if (websocketDefaultSpan) websocketDefaultSpan.textContent = defaultWebSocketUrl;
websocketInput.value = defaultWebSocketUrl;
websocketUrl = defaultWebSocketUrl;

// Optional chunk selector (guard for presence)
if (chunkSelector) {
  chunkSelector.addEventListener("change", () => {
    chunkDuration = parseInt(chunkSelector.value);
  });
}

// WebSocket input change handling
websocketInput.addEventListener("change", () => {
  const urlValue = websocketInput.value.trim();
  if (!urlValue.startsWith("ws://") && !urlValue.startsWith("wss://")) {
    statusText.textContent = "Invalid WebSocket URL (must start with ws:// or wss://)";
    return;
  }
  websocketUrl = urlValue;
  statusText.textContent = "WebSocket URL updated. Ready to connect.";
});

function setupWebSocket() {
  return new Promise((resolve, reject) => {
    try {
      websocket = new WebSocket(websocketUrl);
    } catch (error) {
      statusText.textContent = "Invalid WebSocket URL. Please check and try again.";
      reject(error);
      return;
    }

    websocket.onopen = () => {
      statusText.textContent = "Connected to server.";
      resolve();
    };

    websocket.onclose = () => {
      if (userClosing) {
        if (waitingForStop) {
          statusText.textContent = "Processing finalized or connection closed.";
          if (lastReceivedData) {
            renderLinesWithBuffer(
              lastReceivedData.lines || [],
              lastReceivedData.buffer_diarization || "",
              lastReceivedData.buffer_transcription || "",
              0,
              0,
              true
            );
          }
        }
      } else {
        statusText.textContent = "Disconnected from the WebSocket server. (Check logs if model is loading.)";
        if (isRecording) {
          stopRecording();
        }
      }
      isRecording = false;
      waitingForStop = false;
      userClosing = false;
      lastReceivedData = null;
      websocket = null;
      updateUI();
    };

    websocket.onerror = () => {
      statusText.textContent = "Error connecting to WebSocket.";
      reject(new Error("Error connecting to WebSocket"));
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "ready_to_stop") {
        console.log("Ready to stop received, finalizing display and closing WebSocket.");
        waitingForStop = false;

        if (lastReceivedData) {
          renderLinesWithBuffer(
            lastReceivedData.lines || [],
            lastReceivedData.buffer_diarization || "",
            lastReceivedData.buffer_transcription || "",
            0,
            0,
            true
          );
        }
        statusText.textContent = "Finished processing audio! Ready to record again.";
        recordButton.disabled = false;

        if (websocket) {
          websocket.close();
        }
        return;
      }

      lastReceivedData = data;

      const {
        lines = [],
        buffer_transcription = "",
        buffer_diarization = "",
        remaining_time_transcription = 0,
        remaining_time_diarization = 0,
        status = "active_transcription",
      } = data;

      renderLinesWithBuffer(
        lines,
        buffer_diarization,
        buffer_transcription,
        remaining_time_diarization,
        remaining_time_transcription,
        false,
        status
      );
    };
  });
}

function renderLinesWithBuffer(
  lines,
  buffer_diarization,
  buffer_transcription,
  remaining_time_diarization,
  remaining_time_transcription,
  isFinalizing = false,
  current_status = "active_transcription"
) {
  if (current_status === "no_audio_detected") {
    linesTranscriptDiv.innerHTML =
      "<p style='text-align: center; color: var(--muted); margin-top: 20px;'><em>No audio detected...</em></p>";
    return;
  }

  const showLoading = !isFinalizing && (lines || []).some((it) => it.speaker == 0);
  const showTransLag = !isFinalizing && remaining_time_transcription > 0;
  const showDiaLag = !isFinalizing && !!buffer_diarization && remaining_time_diarization > 0;
  const signature = JSON.stringify({
    lines: (lines || []).map((it) => ({ speaker: it.speaker, text: it.text, beg: it.beg, end: it.end })),
    buffer_transcription: buffer_transcription || "",
    buffer_diarization: buffer_diarization || "",
    status: current_status,
    showLoading,
    showTransLag,
    showDiaLag,
    isFinalizing: !!isFinalizing,
  });
  if (lastSignature === signature) {
    const t = document.querySelector(".lag-transcription-value");
    if (t) t.textContent = fmt1(remaining_time_transcription);
    const d = document.querySelector(".lag-diarization-value");
    if (d) d.textContent = fmt1(remaining_time_diarization);
    const ld = document.querySelector(".loading-diarization-value");
    if (ld) ld.textContent = fmt1(remaining_time_diarization);
    return;
  }
  lastSignature = signature;

  const linesHtml = (lines || [])
    .map((item, idx) => {
      let timeInfo = "";
      if (item.beg !== undefined && item.end !== undefined) {
        timeInfo = ` ${item.beg} - ${item.end}`;
      }

      let speakerLabel = "";
      if (item.speaker === -2) {
        speakerLabel = `<span class="silence">Silence<span id='timeInfo'>${timeInfo}</span></span>`;
      } else if (item.speaker == 0 && !isFinalizing) {
        speakerLabel = `<span class='loading'><span class="spinner"></span><span id='timeInfo'><span class="loading-diarization-value">${fmt1(
          remaining_time_diarization
        )}</span> second(s) of audio are undergoing diarization</span></span>`;
      } else if (item.speaker !== 0) {
        speakerLabel = `<span id="speaker">Speaker ${item.speaker}<span id='timeInfo'>${timeInfo}</span></span>`;
      }

      let currentLineText = item.text || "";

      if (idx === lines.length - 1) {
        if (!isFinalizing && item.speaker !== -2) {
          if (remaining_time_transcription > 0) {
            speakerLabel += `<span class="label_transcription"><span class="spinner"></span>Transcription lag <span id='timeInfo'><span class="lag-transcription-value">${fmt1(
              remaining_time_transcription
            )}</span>s</span></span>`;
          }
          if (buffer_diarization && remaining_time_diarization > 0) {
            speakerLabel += `<span class="label_diarization"><span class="spinner"></span>Diarization lag<span id='timeInfo'><span class="lag-diarization-value">${fmt1(
              remaining_time_diarization
            )}</span>s</span></span>`;
          }
        }

        if (buffer_diarization) {
          if (isFinalizing) {
            currentLineText +=
              (currentLineText.length > 0 && buffer_diarization.trim().length > 0 ? " " : "") + buffer_diarization.trim();
          } else {
            currentLineText += `<span class="buffer_diarization">${buffer_diarization}</span>`;
          }
        }
        if (buffer_transcription) {
          if (isFinalizing) {
            currentLineText +=
              (currentLineText.length > 0 && buffer_transcription.trim().length > 0 ? " " : "") +
              buffer_transcription.trim();
          } else {
            currentLineText += `<span class="buffer_transcription">${buffer_transcription}</span>`;
          }
        }
      }

      return currentLineText.trim().length > 0 || speakerLabel.length > 0
        ? `<p>${speakerLabel}<br/><div class='textcontent'>${currentLineText}</div></p>`
        : `<p>${speakerLabel}<br/></p>`;
    })
    .join("");

  linesTranscriptDiv.innerHTML = linesHtml;
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
}

function updateTimer() {
  if (!startTime) return;

  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  const minutes = Math.floor(elapsed / 60).toString().padStart(2, "0");
  const seconds = (elapsed % 60).toString().padStart(2, "0");
  timerElement.textContent = `${minutes}:${seconds}`;
}

function drawWaveform() {
  if (!analyser) return;

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteTimeDomainData(dataArray);

  waveCtx.clearRect(
    0,
    0,
    waveCanvas.width / (window.devicePixelRatio || 1),
    waveCanvas.height / (window.devicePixelRatio || 1)
  );
  waveCtx.lineWidth = 1;
  waveCtx.strokeStyle = waveStroke;
  waveCtx.beginPath();

  const sliceWidth = (waveCanvas.width / (window.devicePixelRatio || 1)) / bufferLength;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    const v = dataArray[i] / 128.0;
    const y = (v * (waveCanvas.height / (window.devicePixelRatio || 1))) / 2;

    if (i === 0) {
      waveCtx.moveTo(x, y);
    } else {
      waveCtx.lineTo(x, y);
    }

    x += sliceWidth;
  }

  waveCtx.lineTo(
    waveCanvas.width / (window.devicePixelRatio || 1),
    (waveCanvas.height / (window.devicePixelRatio || 1)) / 2
  );
  waveCtx.stroke();

  animationFrame = requestAnimationFrame(drawWaveform);
}

async function startRecording() {
  try {
    try {
      wakeLock = await navigator.wakeLock.request("screen");
    } catch (err) {
      console.log("Error acquiring wake lock.");
    }

    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      } 
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000
    });
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    microphone = audioContext.createMediaStreamSource(stream);
    microphone.connect(analyser);

    // Try different audio formats for better compatibility
    let mimeType = "audio/webm";
    if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
      mimeType = "audio/webm;codecs=opus";
    } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
      mimeType = "audio/mp4";
    } else if (MediaRecorder.isTypeSupported("audio/wav")) {
      mimeType = "audio/wav";
    }
    
    console.log("Using audio format:", mimeType);
    recorder = new MediaRecorder(stream, { mimeType: mimeType });
    recorder.ondataavailable = (e) => {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(e.data);
      }
    };
    recorder.start(chunkDuration);

    startTime = Date.now();
    timerInterval = setInterval(updateTimer, 1000);
    drawWaveform();

    isRecording = true;
    updateUI();
  } catch (err) {
    if (window.location.hostname === "0.0.0.0") {
      statusText.textContent =
        "Error accessing microphone. Browsers may block microphone access on 0.0.0.0. Try using localhost:8000 instead.";
    } else {
      statusText.textContent = "Error accessing microphone. Please allow microphone access.";
    }
    console.error(err);
  }
}

async function stopRecording() {
  if (wakeLock) {
    try {
      await wakeLock.release();
    } catch (e) {
      // ignore
    }
    wakeLock = null;
  }

  userClosing = true;
  waitingForStop = true;

  if (websocket && websocket.readyState === WebSocket.OPEN) {
    const emptyBlob = new Blob([], { type: "audio/webm" });
    websocket.send(emptyBlob);
    statusText.textContent = "Recording stopped. Processing final audio...";
  }

  if (recorder) {
    recorder.stop();
    recorder = null;
  }

  if (microphone) {
    microphone.disconnect();
    microphone = null;
  }

  if (analyser) {
    analyser = null;
  }

  if (audioContext && audioContext.state !== "closed") {
    try {
      await audioContext.close();
    } catch (e) {
      console.warn("Could not close audio context:", e);
    }
    audioContext = null;
  }

  if (animationFrame) {
    cancelAnimationFrame(animationFrame);
    animationFrame = null;
  }

  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  timerElement.textContent = "00:00";
  startTime = null;

  isRecording = false;
  updateUI();
}

async function toggleRecording() {
  if (!isRecording) {
    if (waitingForStop) {
      console.log("Waiting for stop, early return");
      return;
    }
    console.log("Connecting to WebSocket");
    try {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        await startRecording();
      } else {
        await setupWebSocket();
        await startRecording();
      }
    } catch (err) {
      statusText.textContent = "Could not connect to WebSocket or access mic. Aborted.";
      console.error(err);
    }
  } else {
    console.log("Stopping recording");
    stopRecording();
  }
}

function updateUI() {
  recordButton.classList.toggle("recording", isRecording);
  recordButton.disabled = waitingForStop;

  if (waitingForStop) {
    if (statusText.textContent !== "Recording stopped. Processing final audio...") {
      statusText.textContent = "Please wait for processing to complete...";
    }
  } else if (isRecording) {
    statusText.textContent = "Recording...";
  } else {
    if (
      statusText.textContent !== "Finished processing audio! Ready to record again." &&
      statusText.textContent !== "Processing finalized or connection closed."
    ) {
      statusText.textContent = "Click to start transcription";
    }
  }
  if (!waitingForStop) {
    recordButton.disabled = false;
  }
}

recordButton.addEventListener("click", toggleRecording);

// File Upload Functionality
const liveModeBtn = document.getElementById("liveMode");
const uploadModeBtn = document.getElementById("uploadMode");
const youtubeModeBtn = document.getElementById("youtubeMode");
const liveSection = document.getElementById("liveSection");
const uploadSection = document.getElementById("uploadSection");
const youtubeSection = document.getElementById("youtubeSection");
const uploadArea = document.getElementById("uploadArea");
const audioFileInput = document.getElementById("audioFileInput");
const browseButton = document.getElementById("browseButton");
const uploadProgress = document.getElementById("uploadProgress");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");
const fileInfo = document.getElementById("fileInfo");
const fileDetails = document.getElementById("fileDetails");
const uploadModeDescription = document.getElementById("uploadModeDescription");
const uploadModeRadios = document.querySelectorAll('input[name="uploadMode"]');

// Mode switching
liveModeBtn.addEventListener("click", () => {
  switchMode("live");
});

uploadModeBtn.addEventListener("click", () => {
  switchMode("upload");
});

// Add null check and debugging for YouTube button
if (youtubeModeBtn) {
  console.log("✅ YouTube mode button found, adding event listener");
  youtubeModeBtn.addEventListener("click", () => {
    console.log("🎬 YouTube button clicked!");
    switchMode("youtube");
  });
} else {
  console.error("❌ YouTube mode button not found!");
}

function switchMode(mode) {
  console.log("🔄 Switching to mode:", mode);
  
  // Reset all modes
  liveModeBtn.classList.remove("active");
    uploadModeBtn.classList.remove("active");
  youtubeModeBtn.classList.remove("active");
  liveSection.classList.remove("active");
    uploadSection.classList.remove("active");
  youtubeSection.classList.remove("active");
    
  if (mode === "live") {
    console.log("📱 Activating live mode");
    liveModeBtn.classList.add("active");
    liveSection.classList.add("active");
    clearUploadResults();
    clearYouTubeResults();
  } else if (mode === "upload") {
    console.log("📁 Activating upload mode");
    uploadModeBtn.classList.add("active");
    uploadSection.classList.add("active");
    clearTranscriptionResults();
    clearYouTubeResults();
  } else if (mode === "youtube") {
    console.log("🎬 Activating YouTube mode");
    console.log("youtubeModeBtn:", youtubeModeBtn);
    console.log("youtubeSection:", youtubeSection);
    
    if (youtubeModeBtn) youtubeModeBtn.classList.add("active");
    if (youtubeSection) youtubeSection.classList.add("active");
    clearTranscriptionResults();
    clearUploadResults();
  }
}

function clearUploadResults() {
  uploadProgress.classList.add("hidden");
  fileInfo.classList.add("hidden");
  progressFill.style.width = "0%";
  progressText.textContent = "Uploading...";
  audioFileInput.value = "";
}

function clearTranscriptionResults() {
  linesTranscriptDiv.innerHTML = "";
  statusText.textContent = "Click to start transcription";
}

// File upload handling
browseButton.addEventListener("click", () => {
  audioFileInput.click();
});

uploadArea.addEventListener("click", () => {
  audioFileInput.click();
});

audioFileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) {
    const uploadMode = document.querySelector('input[name="uploadMode"]:checked').value;
    if (uploadMode === "stream") {
      handleStreamingUpload(file);
    } else {
      handleFileUpload(file);
    }
  }
});

// Upload mode selection handling
uploadModeRadios.forEach(radio => {
  radio.addEventListener("change", (event) => {
    const mode = event.target.value;
    if (mode === "stream") {
      uploadModeDescription.textContent = "Streaming mode: See transcription appear in real-time as file is processed";
    } else {
      uploadModeDescription.textContent = "Batch mode: Process entire file at once (recommended)";
    }
  });
});

// Drag and drop functionality
uploadArea.addEventListener("dragover", (event) => {
  event.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (event) => {
  event.preventDefault();
  uploadArea.classList.remove("dragover");
  
  const files = event.dataTransfer.files;
  if (files.length > 0) {
    const file = files[0];
    if (file.type.startsWith("audio/")) {
      audioFileInput.files = files;
      const uploadMode = document.querySelector('input[name="uploadMode"]:checked').value;
      if (uploadMode === "stream") {
        handleStreamingUpload(file);
      } else {
        handleFileUpload(file);
      }
    } else {
      alert("Please select an audio file.");
    }
  }
});

async function handleStreamingUpload(file) {
  // Clear previous results
  linesTranscriptDiv.innerHTML = "";
  fileInfo.classList.add("hidden");
  
  // Show progress
  uploadProgress.classList.remove("hidden");
  progressFill.style.width = "0%";
  progressText.textContent = "Connecting...";
  
  try {
    // Connect to streaming WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/upload-stream`;
    const ws = new WebSocket(wsUrl);
    
    let transcriptionStarted = false;
    
    ws.onopen = async () => {
      progressText.textContent = "Connected. Sending file metadata...";
      
      // Send file metadata
      ws.send(JSON.stringify({
        filename: file.name,
        size: file.size,
        type: file.type
      }));
      
      // Read and send file in chunks
      const chunkSize = 8192; // 8KB chunks
      const reader = new FileReader();
      let offset = 0;
      
      const sendNextChunk = () => {
        if (offset >= file.size) {
          // Send empty message to signal end
          ws.send(new ArrayBuffer(0));
          return;
        }
        
        const chunk = file.slice(offset, offset + chunkSize);
        reader.onload = (e) => {
          ws.send(e.target.result);
          offset += chunkSize;
          setTimeout(sendNextChunk, 50); // Small delay between chunks
        };
        reader.readAsArrayBuffer(chunk);
      };
      
      // Start sending chunks after a short delay
      setTimeout(sendNextChunk, 100);
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case "status":
          progressText.textContent = data.message;
          break;
          
        case "progress":
          progressFill.style.width = data.progress + "%";
          progressText.textContent = `Processing... ${Math.round(data.progress)}%`;
          break;
          
        case "transcription":
          if (!transcriptionStarted) {
            transcriptionStarted = true;
            progressText.textContent = "Transcribing in real-time...";
          }
          // Display transcription in real-time
          displayRealtimeTranscription(data.data);
          break;
          
        case "transcription_complete":
          progressText.textContent = "Transcription completed!";
          break;
          
        case "upload_complete":
          uploadProgress.classList.add("hidden");
          showFileInfo(file, "Streaming upload completed successfully!");
          statusText.textContent = "File transcribed successfully with real-time streaming!";
          ws.close();
          break;
          
        case "error":
          handleUploadError(data.message);
          ws.close();
          break;
      }
    };
    
    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      handleUploadError("Connection error occurred");
    };
    
    ws.onclose = () => {
      if (uploadProgress && !uploadProgress.classList.contains("hidden")) {
        progressText.textContent = "Connection closed";
      }
    };
    
  } catch (error) {
    handleUploadError(error.message);
  }
}

async function handleFileUpload(file) {
  // Clear previous results
  linesTranscriptDiv.innerHTML = "";
  fileInfo.classList.add("hidden");
  
  // Show progress
  uploadProgress.classList.remove("hidden");
  progressFill.style.width = "0%";
  progressText.textContent = "Uploading...";
  
  // Prepare form data
  const formData = new FormData();
  formData.append("file", file);
  
  try {
    // Upload with progress tracking
    const xhr = new XMLHttpRequest();
    
    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) {
        const percentComplete = (event.loaded / event.total) * 100;
        progressFill.style.width = percentComplete + "%";
        progressText.textContent = `Uploading... ${Math.round(percentComplete)}%`;
      }
    });
    
    xhr.addEventListener("load", () => {
      if (xhr.status === 200) {
        const response = JSON.parse(xhr.responseText);
        handleUploadSuccess(response, file);
      } else {
        const error = JSON.parse(xhr.responseText);
        handleUploadError(error.detail || "Upload failed");
      }
    });
    
    xhr.addEventListener("error", () => {
      handleUploadError("Network error occurred");
    });
    
    // Start upload
    progressText.textContent = "Processing...";
    xhr.open("POST", "/upload");
    xhr.send(formData);
    
  } catch (error) {
    handleUploadError(error.message);
  }
}

function handleUploadSuccess(response, file) {
  // Hide progress
  uploadProgress.classList.add("hidden");
  
  // Show file info
  fileInfo.classList.remove("hidden");
  
  const fileSizeMB = (response.file_size / (1024 * 1024)).toFixed(2);
  const durationFormatted = formatDuration(response.duration);
  
  fileDetails.innerHTML = `
    <p><strong>Filename:</strong> ${response.filename}</p>
    <p><strong>Size:</strong> ${fileSizeMB} MB</p>
    <p><strong>Duration:</strong> ${durationFormatted}</p>
    <p><strong>Status:</strong> ✅ Transcription completed</p>
  `;
  
  // Display transcription results
  displayTranscriptionResult(response);
  
  statusText.textContent = "File transcribed successfully!";
}

function handleUploadError(errorMessage) {
  uploadProgress.classList.add("hidden");
  statusText.textContent = `Upload failed: ${errorMessage}`;
  
  // Show error in file info
  fileInfo.classList.remove("hidden");
  fileDetails.innerHTML = `
    <p style="color: #dc3545;"><strong>Error:</strong> ${errorMessage}</p>
    <p>Please try again with a different file or check the file format.</p>
  `;
}

function displayTranscriptionResult(response) {
  // Clear previous results
  linesTranscriptDiv.innerHTML = "";
  
  if (response.transcription && response.transcription.trim()) {
    // Create a result container
    const resultContainer = document.createElement("div");
    resultContainer.className = "transcription-result";
    resultContainer.style.cssText = `
      background: var(--upload-bg);
      border: 1px solid var(--upload-border);
      border-radius: 8px;
      padding: 16px;
      margin: 10px 0;
    `;
    
    // Add timestamp
    const timestamp = document.createElement("div");
    timestamp.style.cssText = `
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    `;
    timestamp.textContent = `Transcribed at ${new Date().toLocaleTimeString()}`;
    
    // Add transcription text
    const transcriptionText = document.createElement("div");
    transcriptionText.style.cssText = `
      color: var(--text);
      font-size: 16px;
      line-height: 1.5;
      white-space: pre-wrap;
    `;
    transcriptionText.textContent = response.transcription;
    
    resultContainer.appendChild(timestamp);
    resultContainer.appendChild(transcriptionText);
    linesTranscriptDiv.appendChild(resultContainer);
    
    // If there are segments with timestamps, display them too
    if (response.segments && response.segments.length > 0) {
      const segmentsContainer = document.createElement("div");
      segmentsContainer.style.cssText = `
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid var(--upload-border);
      `;
      
      const segmentsTitle = document.createElement("h4");
      segmentsTitle.textContent = "Detailed Segments:";
      segmentsTitle.style.cssText = `
        color: var(--text);
        font-size: 14px;
        margin: 0 0 8px 0;
      `;
      segmentsContainer.appendChild(segmentsTitle);
      
      response.segments.forEach((segment, index) => {
        if (segment.text && segment.text.trim()) {
          const segmentDiv = document.createElement("div");
          segmentDiv.style.cssText = `
            margin: 4px 0;
            padding: 8px;
            background: var(--silence-bg);
            border-radius: 4px;
            font-size: 14px;
          `;
          
          const segmentText = document.createElement("span");
          segmentText.style.color = "var(--text)";
          segmentText.textContent = segment.text;
          
          segmentDiv.appendChild(segmentText);
          segmentsContainer.appendChild(segmentDiv);
        }
      });
      
      resultContainer.appendChild(segmentsContainer);
    }
  } else {
    // No transcription found
    const noResultDiv = document.createElement("div");
    noResultDiv.style.cssText = `
      text-align: center;
      color: var(--muted);
      font-style: italic;
      padding: 20px;
    `;
    noResultDiv.textContent = "No speech detected in the audio file.";
    linesTranscriptDiv.appendChild(noResultDiv);
  }
}

function displayRealtimeTranscription(transcriptionData) {
  // Handle real-time transcription data similar to live recording
  if (transcriptionData.type === "transcript" && transcriptionData.text) {
    const transcriptDiv = document.createElement("div");
    transcriptDiv.className = "line";
    transcriptDiv.style.cssText = `
      background: var(--upload-bg);
      border: 1px solid var(--upload-border);
      border-radius: 8px;
      padding: 12px;
      margin: 8px 0;
    `;
    
    const timestamp = document.createElement("div");
    timestamp.style.cssText = `
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 6px;
    `;
    timestamp.textContent = `${new Date().toLocaleTimeString()}`;
    
    const textDiv = document.createElement("div");
    textDiv.style.cssText = `
      color: var(--text);
      font-size: 16px;
      line-height: 1.4;
    `;
    textDiv.textContent = transcriptionData.text;
    
    transcriptDiv.appendChild(timestamp);
    transcriptDiv.appendChild(textDiv);
    linesTranscriptDiv.appendChild(transcriptDiv);
    
    // Auto-scroll to bottom
    linesTranscriptDiv.scrollTop = linesTranscriptDiv.scrollHeight;
  }
}

function showFileInfo(file, message) {
  fileInfo.classList.remove("hidden");
  
  const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
  
  fileDetails.innerHTML = `
    <p><strong>Filename:</strong> ${file.name}</p>
    <p><strong>Size:</strong> ${fileSizeMB} MB</p>
    <p><strong>Status:</strong> ✅ ${message}</p>
  `;
}

function formatDuration(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// YouTube Audio Capture Functionality
let youtubePlayer = null;
let youtubeAudioContext = null;
let youtubeMediaSource = null;
let youtubeRecorder = null;
let isYouTubeCapturing = false;
let youtubeWebSocket = null;

// YouTube DOM elements - with debugging
const youtubeUrl = document.getElementById("youtubeUrl");
const loadYouTubeBtn = document.getElementById("loadYouTubeBtn");
const captureYouTubeBtn = document.getElementById("captureYouTubeBtn");
const stopYouTubeBtn = document.getElementById("stopYouTubeBtn");
const youtubePlayerDiv = document.getElementById("youtubePlayer");
const youtubeStatus = document.getElementById("youtubeStatus");
const youtubeDetails = document.getElementById("youtubeDetails");

// Debug YouTube elements
console.log("🔍 YouTube Elements Debug:");
console.log("youtubeUrl:", youtubeUrl);
console.log("loadYouTubeBtn:", loadYouTubeBtn);
console.log("captureYouTubeBtn:", captureYouTubeBtn);
console.log("stopYouTubeBtn:", stopYouTubeBtn);
console.log("youtubePlayerDiv:", youtubePlayerDiv);
console.log("youtubeStatus:", youtubeStatus);
console.log("youtubeDetails:", youtubeDetails);

function clearYouTubeResults() {
  console.log("🧹 Clearing YouTube results");
  
  if (youtubePlayer) {
    if (youtubePlayerDiv) youtubePlayerDiv.innerHTML = "";
    youtubePlayer = null;
  }
  
  if (youtubePlayerDiv) youtubePlayerDiv.classList.add("hidden");
  if (youtubeDetails) youtubeDetails.classList.add("hidden");
  if (youtubeStatus) {
    youtubeStatus.textContent = "Enter a YouTube URL to get started";
    youtubeStatus.className = "status-text";
  }
  if (captureYouTubeBtn) captureYouTubeBtn.disabled = true;
  if (stopYouTubeBtn) stopYouTubeBtn.disabled = true;
  
  if (isYouTubeCapturing) {
    stopYouTubeCapture();
  }
}

function extractVideoId(url) {
  const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/;
  const match = url.match(regex);
  return match ? match[1] : null;
}

function loadYouTubeVideo() {
  const url = youtubeUrl.value.trim();
  if (!url) {
    youtubeStatus.textContent = "Please enter a YouTube URL";
    youtubeStatus.className = "status-text youtube-status error";
    return;
  }

  const videoId = extractVideoId(url);
  if (!videoId) {
    youtubeStatus.textContent = "Invalid YouTube URL. Please check the URL and try again.";
    youtubeStatus.className = "status-text youtube-status error";
    return;
  }

  youtubeStatus.textContent = "Loading video...";
  youtubeStatus.className = "status-text youtube-status loading";

  // Create iframe for YouTube video
  const iframe = document.createElement("iframe");
  iframe.src = `https://www.youtube.com/embed/${videoId}?enablejsapi=1&origin=${window.location.origin}`;
  iframe.frameBorder = "0";
  iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture";
  iframe.allowFullscreen = true;

  youtubePlayerDiv.innerHTML = "";
  youtubePlayerDiv.appendChild(iframe);
  youtubePlayerDiv.classList.remove("hidden");

  // Show video details
  youtubeDetails.innerHTML = `
    <h4>Video Loaded</h4>
    <p><strong>Video ID:</strong> ${videoId}</p>
    <p><strong>Status:</strong> Ready for audio capture</p>
  `;
  youtubeDetails.classList.remove("hidden");

  youtubeStatus.textContent = "Video loaded successfully! Click 'Start Audio Capture' to begin transcription.";
  youtubeStatus.className = "status-text youtube-status success";
  
  captureYouTubeBtn.disabled = false;
  youtubePlayer = iframe;
}

async function startYouTubeCapture() {
  try {
    youtubeStatus.textContent = "Starting audio capture...";
    youtubeStatus.className = "status-text youtube-status loading";

    // Method 1: Try to capture audio from the iframe using Web Audio API
    try {
      // Get display media (screen/tab capture)
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          sampleRate: 16000
        }
      });

      // Create audio context
      youtubeAudioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });

      // Create media source from the stream
      youtubeMediaSource = youtubeAudioContext.createMediaStreamSource(stream);
      
      // Add audio level monitoring
      const analyser = youtubeAudioContext.createAnalyser();
      analyser.fftSize = 256;
      youtubeMediaSource.connect(analyser);
      
      // Create destination for recording
      const destination = youtubeAudioContext.createMediaStreamDestination();
      youtubeMediaSource.connect(destination);
      
      // Monitor audio levels
      const monitorAudioLevels = () => {
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyser.getByteFrequencyData(dataArray);
        
        // Calculate average volume
        const average = dataArray.reduce((a, b) => a + b) / bufferLength;
        
        if (average > 10) { // Only log if there's significant audio
          console.log(`🎵 Audio level: ${average.toFixed(1)}`);
        }
        
        if (isYouTubeCapturing) {
          setTimeout(monitorAudioLevels, 1000); // Check every second
        }
      };
      
      // Start monitoring
      setTimeout(monitorAudioLevels, 1000);

      // Set up WebSocket connection
      await setupYouTubeWebSocket();

      // Create recorder
      let mimeType = "audio/webm";
      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
        mimeType = "audio/webm;codecs=opus";
      } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
        mimeType = "audio/mp4";
      } else if (MediaRecorder.isTypeSupported("audio/wav")) {
        mimeType = "audio/wav";
      }

      youtubeRecorder = new MediaRecorder(destination.stream, { mimeType: mimeType });
      youtubeRecorder.ondataavailable = (e) => {
        if (youtubeWebSocket && youtubeWebSocket.readyState === WebSocket.OPEN) {
          youtubeWebSocket.send(e.data);
        }
      };

      youtubeRecorder.start(chunkDuration);
      isYouTubeCapturing = true;

      youtubeStatus.textContent = "🎤 Capturing audio from YouTube video... Transcription will appear below.";
      youtubeStatus.className = "status-text youtube-status success";
      
      captureYouTubeBtn.disabled = true;
      stopYouTubeBtn.disabled = false;

      // Update details
      youtubeDetails.innerHTML = `
        <h4>Audio Capture Active</h4>
        <p><strong>Status:</strong> ✅ Capturing and transcribing audio</p>
        <p><strong>Method:</strong> Screen/Tab Audio Capture</p>
        <p><strong>Format:</strong> ${mimeType}</p>
      `;

    } catch (displayError) {
      console.warn("Display media capture failed:", displayError);
      throw new Error("Screen capture not available or denied. Please allow screen sharing to capture YouTube audio.");
    }

  } catch (error) {
    console.error("YouTube capture error:", error);
    youtubeStatus.textContent = `Error: ${error.message}`;
    youtubeStatus.className = "status-text youtube-status error";
    
    // Reset state
    isYouTubeCapturing = false;
    captureYouTubeBtn.disabled = false;
    stopYouTubeBtn.disabled = true;
  }
}

function setupYouTubeWebSocket() {
  return new Promise((resolve, reject) => {
    try {
      youtubeWebSocket = new WebSocket(websocketUrl);
    } catch (error) {
      reject(error);
      return;
    }

    youtubeWebSocket.onopen = () => {
      resolve();
    };

    youtubeWebSocket.onclose = () => {
      if (isYouTubeCapturing) {
        youtubeStatus.textContent = "Connection lost. Please restart capture.";
        youtubeStatus.className = "status-text youtube-status error";
        stopYouTubeCapture();
      }
    };

    youtubeWebSocket.onerror = () => {
      reject(new Error("WebSocket connection failed"));
    };

    youtubeWebSocket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "ready_to_stop") {
        youtubeStatus.textContent = "Finished processing audio!";
        youtubeStatus.className = "status-text youtube-status success";
        
        if (lastReceivedData) {
          renderLinesWithBuffer(
            lastReceivedData.lines || [],
            lastReceivedData.buffer_diarization || "",
            lastReceivedData.buffer_transcription || "",
            0,
            0,
            true
          );
        }

        if (youtubeWebSocket) {
          youtubeWebSocket.close();
        }
        return;
      }

      lastReceivedData = data;

      const {
        lines = [],
        buffer_transcription = "",
        buffer_diarization = "",
        remaining_time_transcription = 0,
        remaining_time_diarization = 0,
        status = "active_transcription",
      } = data;

      renderLinesWithBuffer(
        lines,
        buffer_diarization,
        buffer_transcription,
        remaining_time_diarization,
        remaining_time_transcription,
        false,
        status
      );
    };
  });
}

function stopYouTubeCapture() {
  isYouTubeCapturing = false;

  if (youtubeWebSocket && youtubeWebSocket.readyState === WebSocket.OPEN) {
    const emptyBlob = new Blob([], { type: "audio/webm" });
    youtubeWebSocket.send(emptyBlob);
  }

  if (youtubeRecorder) {
    youtubeRecorder.stop();
    youtubeRecorder = null;
  }

  if (youtubeMediaSource) {
    youtubeMediaSource.disconnect();
    youtubeMediaSource = null;
  }

  if (youtubeAudioContext && youtubeAudioContext.state !== "closed") {
    youtubeAudioContext.close();
    youtubeAudioContext = null;
  }

  youtubeStatus.textContent = "Audio capture stopped. Video remains loaded for restart.";
  youtubeStatus.className = "status-text";
  
  captureYouTubeBtn.disabled = false;
  stopYouTubeBtn.disabled = true;

  // Update details
  if (youtubeDetails && !youtubeDetails.classList.contains("hidden")) {
    youtubeDetails.innerHTML = `
      <h4>Video Loaded</h4>
      <p><strong>Status:</strong> Ready for audio capture</p>
      <p><strong>Action:</strong> Click 'Start Audio Capture' to resume</p>
    `;
  }
}

// Event listeners for YouTube functionality
if (loadYouTubeBtn) {
  loadYouTubeBtn.addEventListener("click", loadYouTubeVideo);
  console.log("✅ Load YouTube button listener added");
} else {
  console.error("❌ loadYouTubeBtn not found");
}

if (captureYouTubeBtn) {
  captureYouTubeBtn.addEventListener("click", startYouTubeCapture);
  console.log("✅ Capture YouTube button listener added");
} else {
  console.error("❌ captureYouTubeBtn not found");
}

if (stopYouTubeBtn) {
  stopYouTubeBtn.addEventListener("click", stopYouTubeCapture);
  console.log("✅ Stop YouTube button listener added");
} else {
  console.error("❌ stopYouTubeBtn not found");
}

if (youtubeUrl) {
  youtubeUrl.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      loadYouTubeVideo();
    }
  });
  console.log("✅ YouTube URL input listener added");
} else {
  console.error("❌ youtubeUrl input not found");
}

// Initialize with live mode - wrapped in DOM ready check
document.addEventListener('DOMContentLoaded', function() {
  console.log("🚀 DOM fully loaded, initializing...");
  
  // Re-check YouTube elements after DOM is ready
  const youtubeUrlCheck = document.getElementById("youtubeUrl");
  const loadYouTubeBtnCheck = document.getElementById("loadYouTubeBtn");
  const captureYouTubeBtnCheck = document.getElementById("captureYouTubeBtn");
  const stopYouTubeBtnCheck = document.getElementById("stopYouTubeBtn");
  const youtubePlayerDivCheck = document.getElementById("youtubePlayer");
  const youtubeStatusCheck = document.getElementById("youtubeStatus");
  const youtubeDetailsCheck = document.getElementById("youtubeDetails");
  const youtubeSectionCheck = document.getElementById("youtubeSection");
  const youtubeModeBtnCheck = document.getElementById("youtubeMode");
  
  console.log("🔍 DOM Ready - YouTube Elements Check:");
  console.log("youtubeUrl:", youtubeUrlCheck);
  console.log("loadYouTubeBtn:", loadYouTubeBtnCheck);
  console.log("captureYouTubeBtn:", captureYouTubeBtnCheck);
  console.log("stopYouTubeBtn:", stopYouTubeBtnCheck);
  console.log("youtubePlayerDiv:", youtubePlayerDivCheck);
  console.log("youtubeStatus:", youtubeStatusCheck);
  console.log("youtubeDetails:", youtubeDetailsCheck);
  console.log("youtubeSection:", youtubeSectionCheck);
  console.log("youtubeModeBtn:", youtubeModeBtnCheck);
  
  // Re-attach YouTube mode button listener if it wasn't attached before
  if (youtubeModeBtnCheck && !youtubeModeBtnCheck.hasAttribute('data-listener-attached')) {
    console.log("🔧 Re-attaching YouTube mode button listener");
    youtubeModeBtnCheck.addEventListener("click", () => {
      console.log("🎬 YouTube button clicked (DOM ready listener)!");
      switchMode("youtube");
    });
    youtubeModeBtnCheck.setAttribute('data-listener-attached', 'true');
  }
  
switchMode("live");
});

// Fallback initialization (in case DOMContentLoaded already fired)
if (document.readyState === 'loading') {
  console.log("📄 Document still loading, waiting for DOMContentLoaded...");
} else {
  console.log("📄 Document already loaded, initializing immediately...");
  switchMode("live");
}

// Debug function - call this from browser console to test YouTube button
window.testYouTubeButton = function() {
  console.log("🧪 Testing YouTube button manually...");
  
  const btn = document.getElementById("youtubeMode");
  console.log("Button element:", btn);
  
  if (btn) {
    console.log("Button found! Triggering click...");
    btn.click();
  } else {
    console.error("Button not found!");
  }
};

// Debug function - call this to manually switch to YouTube mode
window.forceYouTubeMode = function() {
  console.log("🔧 Forcing YouTube mode...");
  switchMode("youtube");
};

console.log("🎬 YouTube capture functionality loaded!");
console.log("💡 Debug commands available:");
console.log("   testYouTubeButton() - Test the YouTube button");
console.log("   forceYouTubeMode() - Force switch to YouTube mode");
