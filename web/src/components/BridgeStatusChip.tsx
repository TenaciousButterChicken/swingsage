import { useCallback, useEffect, useState } from "react";
import type { BridgeStatus } from "../lib/types";
import { getBridgeStatus, toggleBridge } from "../lib/api";

// Poll cadence: 5 s is frequent enough that the chip feels live after a
// toggle or a backend crash, slow enough to not hammer the server.
const POLL_MS = 5000;

export default function BridgeStatusChip() {
  const [status, setStatus] = useState<BridgeStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await getBridgeStatus();
      setStatus(s);
      setFetchError(null);
    } catch (e) {
      setFetchError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  async function onClick() {
    if (busy) return;
    setBusy(true);
    try {
      const s = await toggleBridge();
      setStatus(s);
      setFetchError(null);
    } catch (e) {
      setFetchError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  // Loading / disconnected-from-server state
  if (status === null) {
    return (
      <span className="rounded-full hairline bg-ink-800/60 px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-ink-300">
        VTrack …
      </span>
    );
  }

  const listening = status.listening;
  const hasError = !!status.error || !!fetchError;
  const dotColor = hasError
    ? "bg-ember-500"
    : listening
      ? "bg-fairway-500"
      : "bg-ink-400";
  const labelColor = hasError
    ? "text-ember-400"
    : listening
      ? "text-ink-100"
      : "text-ink-300";
  const title = hasError
    ? `Error: ${status.error ?? fetchError}`
    : listening
      ? `Capturing on ${status.host}:${status.port} → GSPro :${status.gspro_port} — click to stop`
      : `Bridge off — click to start capture on ${status.host}:${status.port}`;

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      title={title}
      aria-label={title}
      className={[
        "flex items-center gap-2 rounded-full hairline bg-ink-800/60 px-3 py-1",
        "font-mono text-[10px] uppercase tracking-widest transition-colors",
        "hover:bg-ink-800 disabled:cursor-wait disabled:opacity-60",
        labelColor,
      ].join(" ")}
    >
      <span className={`relative grid h-2 w-2 place-items-center`}>
        <span className={`h-2 w-2 rounded-full ${dotColor}`} />
        {listening && !hasError && (
          <span className="absolute h-2 w-2 rounded-full bg-fairway-500 opacity-60 animate-ping" />
        )}
      </span>
      <span>{busy ? "…" : listening ? "VTrack on" : "VTrack off"}</span>
    </button>
  );
}
