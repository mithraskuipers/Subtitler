(() => {
  "use strict";

  const els = {
    folderInput: document.getElementById("folder-input"),
    browseBtn: document.getElementById("browse-btn"),
    scanBtn: document.getElementById("scan-btn"),
    languageSelect: document.getElementById("language-select"),
    runBtn: document.getElementById("run-btn"),
    stopBtn: document.getElementById("stop-btn"),
    overallProgressFill: document.getElementById("overall-progress-fill"),
    overallProgressLabel: document.getElementById("overall-progress-label"),
    elapsedValue: document.getElementById("elapsed-value"),
    remainingValue: document.getElementById("remaining-value"),
    currentVideoName: document.getElementById("current-video-name"),
    stageFill: document.getElementById("stage-fill"),
    stageLabel: document.getElementById("stage-label"),
    videoTableBody: document.getElementById("video-table-body"),
    selectAllBtn: document.getElementById("select-all-btn"),
    selectNoneBtn: document.getElementById("select-none-btn"),
    consoleBody: document.getElementById("console-body"),
    modelsList: document.getElementById("models-list"),
    refreshModelsBtn: document.getElementById("refresh-models-btn"),
    devicePill: document.getElementById("device-pill"),
    deviceDot: document.getElementById("device-dot"),
    deviceLabel: document.getElementById("device-label"),
    connectionDot: document.getElementById("connection-dot"),
    connectionLabel: document.getElementById("connection-label"),
    interruptModal: document.getElementById("interrupt-modal"),
    interruptFilename: document.getElementById("interrupt-filename"),
    interruptKeepBtn: document.getElementById("interrupt-keep-btn"),
    interruptDeleteBtn: document.getElementById("interrupt-delete-btn"),
    summaryModal: document.getElementById("summary-modal"),
    summarySuccessCount: document.getElementById("summary-success-count"),
    summaryFailedCount: document.getElementById("summary-failed-count"),
    summarySkippedCount: document.getElementById("summary-skipped-count"),
    summaryTimeCount: document.getElementById("summary-time-count"),
    summaryDetails: document.getElementById("summary-details"),
    summaryCloseBtn: document.getElementById("summary-close-btn"),
  };

  const STAGE_LABELS = {
    queued: "Queued",
    loading_model: "Loading model",
    extracting_audio: "Extracting audio",
    transcribing: "Transcribing",
    generating_subtitles: "Generating subtitles",
    saving: "Saving",
    finished: "Finished",
    error: "Error",
  };

  let latestState = null;
  let latestModels = [];
  let summaryShownForCurrentBatch = false;
  let renderedLogCount = 0;
  let lastKnownDeviceLabel = null;
  let modelPollTimer = null;

  function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = value ?? "";
    return div.innerHTML;
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    const units = ["KB", "MB", "GB", "TB"];
    let value = bytes;
    let unitIndex = -1;
    do {
      value /= 1024;
      unitIndex += 1;
    } while (value >= 1024 && unitIndex < units.length - 1);
    return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unitIndex]}`;
  }

  function formatDuration(totalSeconds) {
    const seconds = Math.max(0, Math.round(totalSeconds || 0));
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  function formatShortDuration(totalSeconds) {
    const seconds = Math.max(0, Math.round(totalSeconds || 0));
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  async function apiCall(path, method = "GET", body) {
    const response = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const data = await response.json();
        detail = data.detail || detail;
      } catch (_) {
        /* ignore parse errors */
      }
      throw new Error(detail);
    }
    return response.json();
  }

  function setBusyMessage(message) {
    if (!message) return;
    appendTransientLog(message);
  }

  function appendTransientLog(message) {
    const line = document.createElement("div");
    line.className = "log-line";
    line.innerHTML = `<span class="log-time">--:--:--</span><span class="log-warning">${escapeHtml(message)}</span>`;
    els.consoleBody.appendChild(line);
    els.consoleBody.scrollTop = els.consoleBody.scrollHeight;
  }

  // -- rendering -----------------------------------------------------------
  function render(state) {
    latestState = state;
    renderDevicePill(state);
    renderControls(state);
    renderVideoTable(state);
    renderProgress(state);
    renderCurrentVideo(state);
    renderLogs(state);
    renderInterruptModal(state);
    renderSummaryModal(state);

    if (state.deviceLabel && state.deviceLabel !== lastKnownDeviceLabel) {
      lastKnownDeviceLabel = state.deviceLabel;
      loadModels();
    }
  }

  function renderDevicePill(state) {
    if (state.deviceLabel) {
      els.deviceLabel.textContent = state.deviceLabel;
      els.deviceDot.className = "pill-dot " + (state.deviceLabel.startsWith("CUDA") ? "on-gpu" : "on-cpu");
    } else {
      els.deviceLabel.textContent = "Model not loaded";
      els.deviceDot.className = "pill-dot";
    }
  }

  function modelsReady() {
    // Nothing registered yet (status not loaded) - don't block on an unknown.
    if (!latestModels || latestModels.length === 0) return true;
    return latestModels.every((m) => m.installed);
  }

  function renderControls(state) {
    const running = state.status === "running" || state.status === "stopping" || state.status === "awaiting_interrupt_decision";
    const selectedCount = state.videos.filter((v) => v.selected).length;
    const ready = modelsReady();
    const canRun = !running && selectedCount > 0 && state.status !== "scanning" && ready;

    els.runBtn.disabled = !canRun;
    els.runBtn.title = ready ? "" : "Download the model in the Models panel before running a batch.";
    els.stopBtn.disabled = !(state.status === "running");
    els.scanBtn.disabled = running;
    els.browseBtn.disabled = running;
    els.folderInput.disabled = running;
    els.languageSelect.disabled = running;
    els.selectAllBtn.disabled = running;
    els.selectNoneBtn.disabled = running;

    if (document.activeElement !== els.folderInput && state.sourceDirectory) {
      els.folderInput.value = state.sourceDirectory;
    }
  }

  const STATUS_LABELS = {
    pending: "Pending",
    processing: "Processing",
    done: "Done",
    failed: "Failed",
    skipped: "Skipped",
    interrupted: "Interrupted",
  };

  function renderVideoTable(state) {
    const running = state.status === "running" || state.status === "stopping" || state.status === "awaiting_interrupt_decision";

    if (state.videos.length === 0) {
      els.videoTableBody.innerHTML = '<tr class="empty-row"><td colspan="5">No videos yet — scan a folder to get started.</td></tr>';
      return;
    }

    const rows = state.videos.map((video) => {
      const isActive = state.currentProgress.videoId === video.id;
      return `
        <tr data-video-id="${video.id}" class="${isActive ? "row-active" : ""}">
          <td class="col-check">
            <input type="checkbox" class="checkbox video-checkbox" data-video-id="${video.id}"
              ${video.selected ? "checked" : ""} ${running ? "disabled" : ""} />
          </td>
          <td class="filename-cell" title="${escapeHtml(video.filename)}">${escapeHtml(video.filename)}</td>
          <td class="path-cell" title="${escapeHtml(video.relativePath)}">${escapeHtml(video.relativePath)}</td>
          <td class="col-size">${formatBytes(video.sizeBytes)}</td>
          <td class="col-status">
            <span class="status-badge status-${video.status}">${STATUS_LABELS[video.status] || video.status}</span>
          </td>
        </tr>`;
    });

    els.videoTableBody.innerHTML = rows.join("");

    els.videoTableBody.querySelectorAll(".video-checkbox").forEach((checkbox) => {
      checkbox.addEventListener("change", async (event) => {
        const videoId = event.target.getAttribute("data-video-id");
        const selected = event.target.checked;
        try {
          const newState = await apiCall("/api/videos/selection", "POST", { videoId, selected });
          render(newState);
        } catch (err) {
          appendTransientLog(`Could not update selection: ${err.message}`);
          event.target.checked = !selected;
        }
      });
    });
  }

  function renderProgress(state) {
    const selected = state.videos.filter((v) => v.selected);
    const processed = selected.filter((v) => ["done", "failed", "skipped", "interrupted"].includes(v.status)).length;
    const total = selected.length;
    const fraction = total > 0 ? processed / total : 0;

    els.overallProgressFill.style.width = `${Math.round(fraction * 100)}%`;
    els.overallProgressLabel.textContent = `${processed} / ${total} videos`;

    els.elapsedValue.textContent = formatDuration(state.elapsedSeconds);

    if (state.status === "running" && processed > 0 && state.elapsedSeconds > 0) {
      const perVideo = state.elapsedSeconds / processed;
      const remainingSeconds = perVideo * (total - processed);
      els.remainingValue.textContent = formatDuration(remainingSeconds);
    } else if (state.status === "running") {
      els.remainingValue.textContent = "Estimating…";
    } else {
      els.remainingValue.textContent = "—";
    }
  }

  function renderCurrentVideo(state) {
    const progress = state.currentProgress;
    if (progress.filename) {
      els.currentVideoName.textContent = progress.filename;
      els.stageFill.style.width = `${Math.round((progress.progressFraction || 0) * 100)}%`;
      els.stageLabel.textContent = STAGE_LABELS[progress.stage] || progress.stage;
    } else {
      els.currentVideoName.textContent = state.status === "finished" ? "Batch finished" : "Idle";
      els.stageFill.style.width = "0%";
      els.stageLabel.textContent = state.status === "running" ? "Preparing…" : "Waiting for a batch to start";
    }
  }

  function renderLogs(state) {
    if (state.logs.length === renderedLogCount && renderedLogCount !== 0) return;

    const wasScrolledToBottom =
      els.consoleBody.scrollHeight - els.consoleBody.scrollTop - els.consoleBody.clientHeight < 40;

    els.consoleBody.innerHTML = state.logs
      .map((entry) => {
        const time = new Date(entry.timestamp * 1000).toLocaleTimeString();
        return `<div class="log-line"><span class="log-time">${time}</span><span class="log-${entry.level}">${escapeHtml(entry.message)}</span></div>`;
      })
      .join("");

    renderedLogCount = state.logs.length;

    if (wasScrolledToBottom || renderedLogCount === state.logs.length) {
      els.consoleBody.scrollTop = els.consoleBody.scrollHeight;
    }
  }

  function renderInterruptModal(state) {
    if (state.pendingInterruptDecision) {
      els.interruptFilename.textContent = state.pendingInterruptDecision.filename;
      els.interruptModal.classList.remove("hidden");
    } else {
      els.interruptModal.classList.add("hidden");
    }
  }

  function renderSummaryModal(state) {
    if (!state.summary) {
      summaryShownForCurrentBatch = false;
      return;
    }
    if (summaryShownForCurrentBatch) return;
    summaryShownForCurrentBatch = true;

    els.summarySuccessCount.textContent = state.summary.successful.length;
    els.summaryFailedCount.textContent = state.summary.failed.length;
    els.summarySkippedCount.textContent = state.summary.skipped.length;
    els.summaryTimeCount.textContent = formatShortDuration(state.summary.totalProcessingSeconds);

    const failedLines = state.summary.failed.map(
      (f) => `<div class="detail-line error-line">✕ ${escapeHtml(f.filename)} — ${escapeHtml(f.error)}</div>`
    );
    const skippedLines = state.summary.skipped.map((f) => `<div class="detail-line">– ${escapeHtml(f)} (skipped)</div>`);
    els.summaryDetails.innerHTML = failedLines.concat(skippedLines).join("") || '<div class="detail-line">Everything completed without issues.</div>';

    els.summaryModal.classList.remove("hidden");
  }

  function renderModels(models) {
    if (!models || models.length === 0) {
      els.modelsList.innerHTML = '<div class="model-row"><div class="model-row-text"><span class="model-row-label">No models registered</span></div></div>';
      return;
    }
    els.modelsList.innerHTML = models
      .map((model) => {
        const languages = model.languages.join(", ").toUpperCase();
        let actionHtml;
        if (model.installed) {
          actionHtml = '<span class="model-badge installed">Downloaded</span>';
        } else if (model.downloading) {
          actionHtml = '<span class="model-badge downloading">Downloading…</span>';
        } else {
          actionHtml = `<button class="btn btn-tiny model-download-btn" data-model-name="${escapeHtml(model.name)}" type="button">Download</button>`;
        }
        return `
          <div class="model-row">
            <div class="model-row-text">
              <span class="model-row-label" title="${escapeHtml(model.label)}">${escapeHtml(model.label)}</span>
              <span class="model-row-meta">${languages} · ~${model.approxSizeMb} MB</span>
            </div>
            ${actionHtml}
          </div>`;
      })
      .join("");

    els.modelsList.querySelectorAll(".model-download-btn").forEach((button) => {
      button.addEventListener("click", async () => {
        const modelName = button.getAttribute("data-model-name");
        button.disabled = true;
        button.textContent = "Starting…";
        try {
          await apiCall(`/api/models/${encodeURIComponent(modelName)}/download`, "POST");
          startModelPolling();
          await loadModels();
        } catch (err) {
          appendTransientLog(`Could not start model download: ${err.message}`);
          button.disabled = false;
          button.textContent = "Download";
        }
      });
    });
  }

  function startModelPolling() {
    if (modelPollTimer) return;
    modelPollTimer = setInterval(async () => {
      await loadModels();
      if (!latestModels.some((m) => m.downloading)) {
        clearInterval(modelPollTimer);
        modelPollTimer = null;
      }
    }, 2000);
  }

  async function loadModels() {
    try {
      const models = await apiCall("/api/models");
      latestModels = models;
      renderModels(models);
      if (latestState) renderControls(latestState);
      if (models.some((m) => m.downloading)) startModelPolling();
    } catch (err) {
      appendTransientLog(`Could not load model status: ${err.message}`);
    }
  }

  els.refreshModelsBtn.addEventListener("click", loadModels);

  // -- event handlers --------------------------------------------------------
  els.browseBtn.addEventListener("click", async () => {
    try {
      els.browseBtn.disabled = true;
      const result = await apiCall("/api/browse-folder", "POST");
      els.folderInput.value = result.directory;
    } catch (err) {
      appendTransientLog(`Folder picker: ${err.message}`);
    } finally {
      els.browseBtn.disabled = false;
    }
  });

  els.scanBtn.addEventListener("click", async () => {
    const directory = els.folderInput.value.trim();
    if (!directory) {
      appendTransientLog("Please choose a folder before scanning.");
      return;
    }
    try {
      els.scanBtn.disabled = true;
      const state = await apiCall("/api/scan", "POST", { directory });
      render(state);
    } catch (err) {
      appendTransientLog(`Scan failed: ${err.message}`);
    } finally {
      els.scanBtn.disabled = false;
    }
  });

  els.selectAllBtn.addEventListener("click", async () => {
    try {
      const state = await apiCall("/api/videos/selection/all", "POST", { selected: true });
      render(state);
    } catch (err) {
      appendTransientLog(err.message);
    }
  });

  els.selectNoneBtn.addEventListener("click", async () => {
    try {
      const state = await apiCall("/api/videos/selection/all", "POST", { selected: false });
      render(state);
    } catch (err) {
      appendTransientLog(err.message);
    }
  });

  els.runBtn.addEventListener("click", async () => {
    try {
      const languageCode = els.languageSelect.value;
      const state = await apiCall("/api/start", "POST", { languageCode });
      render(state);
    } catch (err) {
      appendTransientLog(`Could not start batch: ${err.message}`);
    }
  });

  els.stopBtn.addEventListener("click", async () => {
    try {
      els.stopBtn.disabled = true;
      const state = await apiCall("/api/stop", "POST");
      render(state);
    } catch (err) {
      appendTransientLog(err.message);
    }
  });

  els.interruptKeepBtn.addEventListener("click", async () => {
    try {
      const state = await apiCall("/api/interrupt-decision", "POST", { keep: true });
      render(state);
    } catch (err) {
      appendTransientLog(err.message);
    }
  });

  els.interruptDeleteBtn.addEventListener("click", async () => {
    try {
      const state = await apiCall("/api/interrupt-decision", "POST", { keep: false });
      render(state);
    } catch (err) {
      appendTransientLog(err.message);
    }
  });

  els.summaryCloseBtn.addEventListener("click", () => {
    els.summaryModal.classList.add("hidden");
  });

  // -- bootstrap ---------------------------------------------------------
  async function loadLanguages() {
    try {
      const languages = await apiCall("/api/languages");
      els.languageSelect.innerHTML = languages.map((l) => `<option value="${l.code}">${escapeHtml(l.label)}</option>`).join("");
    } catch (err) {
      els.languageSelect.innerHTML = '<option value="nl">Dutch</option>';
    }
  }

  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

    socket.addEventListener("open", () => {
      els.connectionDot.className = "pill-dot connected";
      els.connectionLabel.textContent = "Connected";
    });

    socket.addEventListener("message", (event) => {
      const state = JSON.parse(event.data);
      render(state);
    });

    socket.addEventListener("close", () => {
      els.connectionDot.className = "pill-dot disconnected";
      els.connectionLabel.textContent = "Reconnecting…";
      setTimeout(connectWebSocket, 1500);
    });

    socket.addEventListener("error", () => {
      socket.close();
    });
  }

  async function bootstrap() {
    await loadLanguages();
    await loadModels();
    try {
      const state = await apiCall("/api/state");
      render(state);
    } catch (err) {
      appendTransientLog(`Could not load initial state: ${err.message}`);
    }
    connectWebSocket();
  }

  bootstrap();
})();
