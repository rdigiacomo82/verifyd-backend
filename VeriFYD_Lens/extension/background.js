const AGENT = "http://127.0.0.1:8765";
const REQUIRED_AGENT_VERSION = "0.4.2";

async function settings() {
  return await chrome.storage.local.get({ automaticProtection: true });
}

async function health() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 1500);
  try {
    const response = await fetch(`${AGENT}/health`, { signal: controller.signal });
    return response.ok ? await response.json() : null;
  } catch (error) {
    console.warn("VeriFYD Lens health check failed:", error);
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

async function start(url, browserDownloadId = null, mode = "automatic") {
  const base = {
    status: "QUEUED",
    summary: mode === "automatic" ? "DOWNLOAD PAUSED" : "QUEUED",
    source: url,
    browserDownloadId,
    protectionMode: mode,
    findings: [
      mode === "automatic"
        ? "Chrome download paused automatically while VeriFYD Lens analyzes a quarantined copy."
        : "VeriFYD Lens accepted the link for protected analysis."
    ]
  };

  await chrome.storage.local.set({ lastResult: base });

  const response = await fetch(`${AGENT}/scan/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url })
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `Lens agent HTTP ${response.status}`);

  base.scan_id = data.scan_id;
  base.status = data.status || "QUEUED";
  await chrome.storage.local.set({ lastResult: base });

  await chrome.tabs.create({ url: chrome.runtime.getURL("result.html") });
}

async function menu() {
  try { await chrome.contextMenus.removeAll(); } catch (_) {}
  chrome.contextMenus.create({
    id: "verifyd-quarantine-scan",
    title: "Quarantine & analyze with VeriFYD Lens",
    contexts: ["link"]
  });
}

chrome.runtime.onInstalled.addListener(async () => {
  await menu();
  const stored = await chrome.storage.local.get("automaticProtection");
  if (typeof stored.automaticProtection === "undefined") {
    await chrome.storage.local.set({ automaticProtection: true });
  }
});

chrome.runtime.onStartup.addListener(menu);

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== "verifyd-quarantine-scan" || !info.linkUrl) return;

  try {
    await start(info.linkUrl, null, "manual");
  } catch (error) {
    await chrome.storage.local.set({
      lastResult: {
        status: "ERROR",
        summary: "SCAN FAILED",
        source: info.linkUrl,
        findings: [String(error.message || error)]
      }
    });
    await chrome.tabs.create({ url: chrome.runtime.getURL("result.html") });
  }
});

chrome.downloads.onCreated.addListener((item) => {
  void (async () => {
    if (!item) return;

    const url = item.finalUrl || item.url || "";

    if (
      url.startsWith("http://127.0.0.1:8765/release-file/") ||
      url.startsWith("http://localhost:8765/release-file/")
    ) return;

    if (!/^https?:\/\//i.test(url)) return;

    try {
      await chrome.downloads.pause(item.id);
      console.log("VeriFYD Lens paused download:", item.id, url);
    } catch (error) {
      console.warn("VeriFYD Lens could not pause download:", item.id, error);
      return;
    }

    const stored = await settings();

    if (!stored.automaticProtection) {
      try { await chrome.downloads.resume(item.id); } catch (error) {
        console.warn("VeriFYD Lens could not resume download:", item.id, error);
      }
      return;
    }

    const agentHealth = await health();

    if (!agentHealth || !agentHealth.ok || agentHealth.version !== REQUIRED_AGENT_VERSION) {
      console.warn("VeriFYD Lens agent unavailable or version mismatch:", agentHealth);
      try { await chrome.downloads.resume(item.id); } catch (error) {
        console.warn("VeriFYD Lens could not resume after agent check:", item.id, error);
      }
      return;
    }

    try {
      await start(url, item.id, "automatic");
    } catch (error) {
      console.error("VeriFYD Lens automatic scan failed:", error);
      await chrome.storage.local.set({
        lastResult: {
          status: "ERROR",
          summary: "LENS COULD NOT START",
          source: url,
          browserDownloadId: item.id,
          protectionMode: "automatic",
          findings: [
            String(error.message || error),
            "The original Chrome download remains paused."
          ]
        }
      });
      await chrome.tabs.create({ url: chrome.runtime.getURL("result.html") });
    }
  })();
});
