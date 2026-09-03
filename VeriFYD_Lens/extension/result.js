const AGENT = "http://127.0.0.1:8765";

let current = null;
let pollTimer = null;
let originalHandled = false;

const AUTH = {
  riskMax: 39,
  reviewMin: 40,
  reviewMax: 54,
  supportedMin: 55
};

function formatScore(value) {
  return Number.isFinite(value) ? `${Math.round(value)}/100` : "Not analyzed";
}

function setTone(el, tone) {
  if (!el) return;
  el.classList.remove("tone-green", "tone-orange", "tone-red", "tone-neutral");
  el.classList.add(`tone-${tone}`);
}

function authenticityPresentation(score, label) {
  if (!Number.isFinite(score)) {
    return { tone: "neutral", state: label || "Not analyzed" };
  }
  if (score <= AUTH.riskMax) {
    return { tone: "red", state: "AI / Tampering Detected" };
  }
  if (score <= AUTH.reviewMax) {
    return { tone: "orange", state: "UNDETERMINED / Review Recommended" };
  }
  return { tone: "green", state: "REAL / Authenticity Supported" };
}

function securityPresentation(score, defenderStatus, findings) {
  const list = Array.isArray(findings) ? findings.join(" ").toLowerCase() : "";
  const threat =
    list.includes("malware") ||
    list.includes("malicious") ||
    list.includes("threat detected") ||
    list.includes("phishing");

  if (threat) {
    return { tone: "red", state: "High Security Risk" };
  }
  if (!Number.isFinite(score)) {
    return { tone: "neutral", state: "Not analyzed" };
  }
  if (score < 40) {
    return { tone: "red", state: "High Security Risk" };
  }
  if (score < 70) {
    return { tone: "orange", state: "Security Review Recommended" };
  }
  if (defenderStatus === "UNAVAILABLE") {
    return { tone: "green", state: "Low Concern — AV Scan Unavailable" };
  }
  return { tone: "green", state: "Low Security Concern" };
}

function trustPresentation(score) {
  if (!Number.isFinite(score)) {
    return { tone: "neutral", state: "Not analyzed" };
  }
  if (score < 40) return { tone: "red", state: "Low Trust" };
  if (score < 70) return { tone: "orange", state: "Moderate Trust / Review" };
  return { tone: "green", state: "High Trust" };
}

function overallPresentation(sec, auth, trust) {
  if (sec.tone === "red") {
    return {
      tone: "red",
      title: "HIGH SECURITY RISK",
      note: "Potentially dangerous security indicators were detected. Do not open the file until reviewed."
    };
  }
  if (auth.tone === "red") {
    return {
      tone: "red",
      title: "AI / TAMPERING DETECTED",
      note: "VeriFYD found stronger indicators of AI generation, synthetic media, manipulation, or tampering."
    };
  }
  if (sec.tone === "orange" || auth.tone === "orange" || trust.tone === "orange") {
    return {
      tone: "orange",
      title: "REVIEW RECOMMENDED",
      note: "One or more signals are mixed, borderline, limited, or uncertain. Human review is recommended."
    };
  }
  if (sec.tone === "green" && auth.tone === "green" && trust.tone === "green") {
    return {
      tone: "green",
      title: "LOW CONCERN",
      note: "Available security, authenticity, and trust signals support release with low concern."
    };
  }
  return {
    tone: "neutral",
    title: current?.status || "SCANNING…",
    note: "Analysis is still in progress or one or more checks are unavailable."
  };
}

async function loadStoredResult() {
  const data = await chrome.storage.local.get("lastResult");
  return data.lastResult || null;
}

async function saveStoredResult(record) {
  await chrome.storage.local.set({ lastResult: record });
}

async function cancelOriginalDownloadIfNeeded() {
  if (originalHandled || !current || !Number.isInteger(current.browserDownloadId)) return;
  try { await chrome.downloads.cancel(current.browserDownloadId); } catch (_) {}
  try { await chrome.downloads.erase({ id: current.browserDownloadId }); } catch (_) {}
  originalHandled = true;
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function pollScan() {
  if (!current || !current.scan_id) return;
  try {
    const response = await fetch(`${AGENT}/scan/${current.scan_id}`, { cache: "no-store" });
    if (!response.ok) {
      console.warn("VeriFYD Lens poll failed:", response.status);
      return;
    }

    const next = await response.json();
    next.browserDownloadId = current.browserDownloadId;
    next.protectionMode = current.protectionMode;

    current = next;
    await saveStoredResult(current);
    await render(false);

    if (["QUARANTINED", "BLOCKED"].includes(current.status)) {
      await cancelOriginalDownloadIfNeeded();
      stopPolling();
    } else if (["ERROR", "RELEASED", "DELETED"].includes(current.status)) {
      stopPolling();
    }
  } catch (error) {
    console.error("VeriFYD Lens poll exception:", error);
  }
}

async function render(startPolling = true) {
  const stored = await loadStoredResult();
  if (stored) current = stored;

  if (!current) {
    document.getElementById("summary").textContent = "NO SCAN FOUND";
    return;
  }

  document.getElementById("security").textContent = formatScore(current.security_score);
  document.getElementById("auth").textContent = formatScore(current.authenticity_score);
  document.getElementById("trust").textContent = formatScore(current.trust_score);

  const sec = securityPresentation(
    current.security_score,
    current.defender_status,
    current.findings
  );
  const auth = authenticityPresentation(
    current.authenticity_score,
    current.authenticity_label
  );
  const trust = trustPresentation(current.trust_score);

  setTone(document.getElementById("securityCard"), sec.tone);
  setTone(document.getElementById("authCard"), auth.tone);
  setTone(document.getElementById("trustCard"), trust.tone);

  document.getElementById("securityState").textContent = sec.state;
  document.getElementById("authState").textContent = auth.state;
  document.getElementById("trustState").textContent = trust.state;

  const overall = overallPresentation(sec, auth, trust);
  const overallEl = document.getElementById("overall");
  setTone(overallEl, overall.tone);
  document.getElementById("summary").textContent =
    ["QUEUED","DOWNLOADING","SECURITY_SCANNING","AUTHENTICITY_SCANNING"].includes(current.status)
      ? (current.summary || current.status || "SCANNING…")
      : overall.title;
  document.getElementById("overallNote").textContent =
    ["QUEUED","DOWNLOADING","SECURITY_SCANNING","AUTHENTICITY_SCANNING"].includes(current.status)
      ? "Analysis is in progress. The original download remains protected while VeriFYD completes its checks."
      : overall.note;

  const defender = current.defender_status || "Pending";
  const method = current.defender_method ? ` via ${current.defender_method}` : "";
  document.getElementById("secstatus").textContent = `${defender}${method}`;

  document.getElementById("verdict").textContent =
    current.authenticity_label || "Not analyzed";

  document.getElementById("reason").textContent =
    current.authenticity_reasoning || "No VeriFYD reasoning returned yet.";

  const findings = document.getElementById("findings");
  findings.textContent = "";
  const items = Array.isArray(current.findings) && current.findings.length
    ? current.findings
    : ["No findings yet."];

  for (const item of items) {
    const row = document.createElement("div");
    row.className = "finding";
    row.textContent = item;
    findings.appendChild(row);
  }

  const fileBits = [];
  if (current.filename) fileBits.push(`Name: ${current.filename}`);
  if (Number.isFinite(current.size_bytes)) fileBits.push(`Size: ${current.size_bytes.toLocaleString()} bytes`);
  if (current.sha256) fileBits.push(`SHA-256: ${current.sha256}`);
  if (current.status) fileBits.push(`Status: ${current.status}`);
  if (current.media_type) fileBits.push(`Media type: ${current.media_type}`);
  if (current.protectionMode === "automatic") fileBits.push("Protection: AUTOMATIC");
  if (current.released_path) fileBits.push(`Released to: ${current.released_path}`);

  document.getElementById("file").textContent =
    fileBits.join("\n") || "Waiting for scan…";

  const isQuarantined = current.status === "QUARANTINED" && !!current.scan_id;
  document.getElementById("release").disabled = !isQuarantined;
  document.getElementById("delete").disabled = !isQuarantined;

  const canResume =
    current.protectionMode === "automatic" &&
    Number.isInteger(current.browserDownloadId) &&
    current.status === "ERROR";
  document.getElementById("resume").hidden = !canResume;

  const terminal = ["QUARANTINED","ERROR","BLOCKED","RELEASED","DELETED"];
  if (startPolling && current.scan_id && !terminal.includes(current.status) && !pollTimer) {
    pollTimer = setInterval(pollScan, 2000);
    await pollScan();
  }
}

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName === "local" && changes.lastResult) render();
});

document.getElementById("release").addEventListener("click", async () => {
  if (!current || !current.scan_id) return;

  document.getElementById("msg").textContent =
    "Choose where to save the protected copy…";

  try {
    await cancelOriginalDownloadIfNeeded();

    const releaseUrl =
      `${AGENT}/release-file/${current.scan_id}?ts=${Date.now()}`;

    const downloadId = await chrome.downloads.download({
      url: releaseUrl,
      filename: current.filename || undefined,
      saveAs: true,
      conflictAction: "uniquify"
    });

    if (!Number.isInteger(downloadId)) {
      throw new Error("Chrome did not start the protected save.");
    }

    const onChanged = async (delta) => {
      if (delta.id !== downloadId) return;

      if (delta.state && delta.state.current === "complete") {
        chrome.downloads.onChanged.removeListener(onChanged);

        let savedPath = "";
        try {
          const items = await chrome.downloads.search({ id: downloadId });
          if (items && items[0] && items[0].filename) {
            savedPath = items[0].filename;
          }
        } catch (_) {}

        const confirmResponse = await fetch(
          `${AGENT}/release-confirm/${current.scan_id}`,
          { method: "POST" }
        );

        const confirmData = await confirmResponse.json().catch(() => ({}));

        if (!confirmResponse.ok) {
          throw new Error(confirmData.detail || "Release confirmation failed");
        }

        current.status = "RELEASED";
        current.released_path = savedPath || "User-selected location";
        await saveStoredResult(current);

        document.getElementById("msg").textContent =
          savedPath
            ? `Saved protected copy to: ${savedPath}`
            : "Protected copy saved successfully.";

        await render(false);
      }

      if ((delta.state && delta.state.current === "interrupted") || delta.error) {
        chrome.downloads.onChanged.removeListener(onChanged);
        document.getElementById("msg").textContent =
          "Save was canceled or interrupted. The protected copy remains quarantined.";
      }
    };

    chrome.downloads.onChanged.addListener(onChanged);

  } catch (error) {
    document.getElementById("msg").textContent =
      String(error.message || error);
  }
});

document.getElementById("delete").addEventListener("click", async () => {
  if (!current || !current.scan_id) return;

  document.getElementById("msg").textContent = "Deleting…";

  try {
    await cancelOriginalDownloadIfNeeded();

    const response = await fetch(`${AGENT}/delete/${current.scan_id}`, {
      method: "POST"
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Delete failed");
    }

    current.status = "DELETED";
    await saveStoredResult(current);

    document.getElementById("msg").textContent =
      "Quarantined file deleted.";

    await render(false);
  } catch (error) {
    document.getElementById("msg").textContent =
      String(error.message || error);
  }
});

document.getElementById("resume").addEventListener("click", async () => {
  if (!current || !Number.isInteger(current.browserDownloadId)) return;

  try {
    await chrome.downloads.resume(current.browserDownloadId);
    document.getElementById("msg").textContent =
      "Original Chrome download resumed without VeriFYD analysis.";
    document.getElementById("resume").hidden = true;
  } catch (error) {
    document.getElementById("msg").textContent =
      `Could not resume original download: ${String(error.message || error)}`;
  }
});

render();
