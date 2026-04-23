import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { deleteAnalysis, getAnalysis, listHistory } from "../lib/api";
import type { AnalysisResult, HistoryItem } from "../lib/types";

interface HistoryViewProps {
  onOpen: (jobId: string, result: AnalysisResult) => void;
  onBack: () => void;
}

export default function HistoryView({ onOpen, onBack }: HistoryViewProps) {
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    listHistory()
      .then((r) => {
        if (alive) setItems(r.items);
      })
      .catch((e) => {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, []);

  async function open(item: HistoryItem) {
    setLoadingId(item.job_id);
    try {
      const result = await getAnalysis(item.job_id);
      onOpen(item.job_id, result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoadingId(null);
    }
  }

  async function remove(item: HistoryItem) {
    const ok = window.confirm(
      "Delete this swing from history? The video + analysis files will be removed from disk."
    );
    if (!ok) return;
    try {
      await deleteAnalysis(item.job_id);
      setItems((prev) => (prev ?? []).filter((it) => it.job_id !== item.job_id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-6">
        <div>
          <p className="label-eyebrow">Swing history</p>
          <h2 className="mt-1 font-display text-display-lg tracking-tight text-ink-100">
            Every swing you've analysed.
          </h2>
          <p className="mt-3 max-w-xl text-sm text-ink-200">
            Click any entry to re-open the full analysis — metrics, video, ball
            flight, and the coaching Qwen gave you at the time.
          </p>
        </div>
        <button className="btn-ghost" onClick={onBack}>
          Analyse a new swing
        </button>
      </div>

      {error && (
        <div className="card-glass p-5">
          <p className="label-eyebrow text-ember-400">Couldn't load history</p>
          <p className="mt-2 font-mono text-xs text-ink-200">{error}</p>
        </div>
      )}

      {items === null && !error && (
        <div className="card-glass p-10 text-center text-sm text-ink-300">
          Loading…
        </div>
      )}

      {items !== null && items.length === 0 && (
        <div className="card-glass p-10 text-center">
          <p className="font-display text-lg text-ink-100">No swings yet.</p>
          <p className="mt-2 text-sm text-ink-300">
            Upload a swing clip and SwingSage will save every analysis here.
          </p>
        </div>
      )}

      {items !== null && items.length > 0 && (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {items.map((it, i) => (
            <HistoryCard
              key={it.job_id}
              item={it}
              onClick={() => open(it)}
              onDelete={() => remove(it)}
              loading={loadingId === it.job_id}
              delay={i * 0.03}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function HistoryCard({
  item,
  onClick,
  onDelete,
  loading,
  delay,
}: {
  item: HistoryItem;
  onClick: () => void;
  onDelete: () => void;
  loading: boolean;
  delay: number;
}) {
  const when = formatWhen(item.created_at);
  return (
    <motion.div
      className="card-glass group relative overflow-hidden transition-shadow hover:shadow-glow"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay }}
    >
      {/* Delete button — absolutely positioned, stopPropagation so it
          doesn't also trigger the card's open action. */}
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        aria-label="Delete this swing from history"
        title="Delete this swing"
        className="absolute right-2 top-2 z-10 grid h-7 w-7 place-items-center rounded-full bg-ink-950/70 text-ink-300 opacity-0 transition-all hover:bg-ember-500/20 hover:text-ember-400 focus:opacity-100 group-hover:opacity-100"
      >
        <TrashIcon />
      </button>
      <button
        type="button"
        onClick={onClick}
        disabled={loading}
        className="block w-full text-left disabled:opacity-60"
      >
        <div className="relative aspect-video bg-ink-900">
          {item.thumbnail ? (
            <img
              src={item.thumbnail}
              alt=""
              className="h-full w-full object-cover"
            />
          ) : (
            <div className="grid h-full w-full place-items-center text-xs text-ink-300">
              No thumbnail
            </div>
          )}
          {item.has_ball_flight && (
            <span className="absolute left-2 top-2 rounded-full bg-fairway-500/20 px-2 py-0.5 font-mono text-[9px] uppercase tracking-widest text-fairway-400 hairline">
              VTrack paired
            </span>
          )}
        </div>
        <div className="p-4">
          <p className="label-eyebrow">{when}</p>
          <div className="mt-2 flex items-baseline gap-4 font-mono text-xs text-ink-200">
            <Summary label="Shoulder" value={item.shoulder_turn_deg} />
            <Summary label="X-factor" value={item.x_factor_deg} />
          </div>
          {item.faults.length > 0 && (
            <ul className="mt-3 space-y-1">
              {item.faults.map((f) => (
                <li
                  key={f}
                  className="truncate text-xs text-ember-400"
                  title={f}
                >
                  • {f}
                </li>
              ))}
            </ul>
          )}
          {loading && (
            <p className="mt-3 text-xs text-champagne-300">Loading analysis…</p>
          )}
        </div>
      </button>
    </motion.div>
  );
}

function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <path
        d="M3 4.5 H13 M5.5 4.5 V3 A1 1 0 0 1 6.5 2 H9.5 A1 1 0 0 1 10.5 3 V4.5 M4.5 4.5 V13 A1 1 0 0 0 5.5 14 H10.5 A1 1 0 0 0 11.5 13 V4.5 M7 7 V11.5 M9 7 V11.5"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function Summary({ label, value }: { label: string; value: number | null }) {
  return (
    <span>
      <span className="text-ink-300">{label}</span>{" "}
      <span className="text-ink-100">
        {value === null ? "—" : `${value.toFixed(1)}°`}
      </span>
    </span>
  );
}

function formatWhen(isoUtc: string): string {
  try {
    // The server stores UTC in "YYYY-MM-DD HH:MM:SS" format (no timezone).
    // Treat it as UTC explicitly so the browser renders local time correctly.
    const normalized = isoUtc.includes("T") ? isoUtc : isoUtc.replace(" ", "T") + "Z";
    const d = new Date(normalized);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    if (diffMin < 1) return "just now";
    if (diffMin < 60) return `${diffMin} min ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr} h ago`;
    return d.toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    return isoUtc;
  }
}
