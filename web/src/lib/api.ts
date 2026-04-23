import type { AnalysisResult, BridgeStatus, HistoryItem, WsEvent } from "./types";

// In dev, Vite proxies /api and /captures to 127.0.0.1:8000.
// In a production build, we assume the frontend is served from the
// same origin as the FastAPI app.

export async function uploadVideo(file: File): Promise<{ job_id: string; filename: string }> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Upload failed (${res.status}): ${txt}`);
  }
  return res.json();
}

export function openProgressSocket(
  jobId: string,
  onEvent: (ev: WsEvent) => void,
  onClose: () => void,
): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/api/ws/${jobId}`);
  ws.addEventListener("message", (m) => {
    try {
      const ev = JSON.parse(m.data) as WsEvent;
      onEvent(ev);
    } catch {
      // swallow; server should always emit JSON
    }
  });
  ws.addEventListener("close", onClose);
  ws.addEventListener("error", onClose);
  return ws;
}

export async function getBridgeStatus(): Promise<BridgeStatus> {
  const res = await fetch("/api/bridge/status");
  if (!res.ok) throw new Error(`Bridge status failed (${res.status})`);
  return res.json();
}

export async function toggleBridge(): Promise<BridgeStatus> {
  const res = await fetch("/api/bridge/toggle", { method: "POST" });
  if (!res.ok) throw new Error(`Bridge toggle failed (${res.status})`);
  return res.json();
}

export async function listHistory(): Promise<{ items: HistoryItem[] }> {
  const res = await fetch("/api/history");
  if (!res.ok) throw new Error(`History fetch failed (${res.status})`);
  return res.json();
}

export async function getAnalysis(jobId: string): Promise<AnalysisResult> {
  const res = await fetch(`/api/analysis/${jobId}`);
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Analysis fetch failed (${res.status}): ${txt}`);
  }
  return res.json();
}

export async function deleteAnalysis(jobId: string): Promise<void> {
  const res = await fetch(`/api/analysis/${jobId}`, { method: "DELETE" });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`Delete failed (${res.status}): ${txt}`);
  }
}
