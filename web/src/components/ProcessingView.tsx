import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { openProgressSocket } from "../lib/api";
import type { AnalysisResult, StageName, WsEvent } from "../lib/types";

interface ProcessingViewProps {
  jobId: string;
  filename: string;
  onDone: (jobId: string, result: AnalysisResult) => void;
  onError: (message: string) => void;
}

// Ordered list of stages with display copy. Keep in sync with the
// pipeline's stage() emissions in server/main.py.
const STAGES: { name: StageName; label: string; hint: string }[] = [
  { name: "decode", label: "Decode",     hint: "Reading every frame of your video" },
  { name: "pose",   label: "3D pose",    hint: "NLF recovers 24 joint positions per frame" },
  { name: "trim",   label: "Auto-trim",  hint: "Locking onto the swing via wrist-velocity peak" },
  { name: "swingnet", label: "Events",   hint: "SwingNet locates address → impact → finish (diagnostic)" },
  { name: "metrics", label: "Metrics",   hint: "Extrema over the backswing → all the numbers" },
  { name: "coaching", label: "Coach",    hint: "Qwen 3 14B translates numbers into feedback" },
];

export default function ProcessingView({ jobId, filename, onDone, onError }: ProcessingViewProps) {
  const [currentStage, setCurrentStage] = useState<StageName | null>(null);
  const [completedStages, setCompletedStages] = useState<Set<StageName>>(new Set());
  const [logs, setLogs] = useState<string[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const startedAt = useRef<number>(performance.now());

  // Tick a clock so the user sees something happening during the ~40s NLF pass
  useEffect(() => {
    const id = window.setInterval(() => {
      setElapsed((performance.now() - startedAt.current) / 1000);
    }, 100);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    const ws = openProgressSocket(
      jobId,
      (ev: WsEvent) => {
        if (ev.type === "stage") {
          setCompletedStages((prev) => {
            if (!currentStage) return prev;
            const next = new Set(prev);
            next.add(currentStage);
            return next;
          });
          setCurrentStage(ev.name);
        } else if (ev.type === "log") {
          setLogs((prev) => [...prev, ev.message]);
        } else if (ev.type === "done") {
          onDone(jobId, ev.result);
        } else if (ev.type === "error") {
          onError(ev.message);
        }
      },
      () => {
        // WS closed — if we haven't resolved, surface the issue.
      },
    );
    return () => {
      try {
        ws.close();
      } catch {
        /* ignore */
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  const currentStageMeta = STAGES.find((s) => s.name === currentStage) ?? null;
  const currentIdx = currentStage ? STAGES.findIndex((s) => s.name === currentStage) : -1;
  const progressFraction = currentIdx < 0 ? 0 : (currentIdx + 0.5) / STAGES.length;

  return (
    <div className="mx-auto max-w-3xl">
      <div className="mb-8 flex items-baseline justify-between">
        <div>
          <p className="label-eyebrow">Processing</p>
          <h2 className="mt-1 font-display text-3xl tracking-tight text-ink-100">
            {filename}
          </h2>
        </div>
        <div className="text-right">
          <p className="label-eyebrow">Elapsed</p>
          <p className="mt-1 font-mono text-2xl text-ink-100">{elapsed.toFixed(1)}s</p>
        </div>
      </div>

      <div className="card-glass overflow-hidden p-8">
        {/* Thin progress rail */}
        <div className="relative mb-8 h-1 w-full overflow-hidden rounded-full bg-ink-800">
          <motion.div
            className="absolute inset-y-0 left-0 rounded-full bg-champagne-300"
            animate={{ width: `${progressFraction * 100}%` }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          />
        </div>

        <ul className="space-y-4">
          {STAGES.map((s) => {
            const isCurrent = s.name === currentStage;
            const isDone = completedStages.has(s.name);
            return (
              <li key={s.name} className="flex items-center gap-5">
                <StageMarker state={isDone ? "done" : isCurrent ? "active" : "pending"} />
                <div className="flex-1">
                  <div
                    className={[
                      "font-display text-lg tracking-tight transition-colors",
                      isDone || isCurrent ? "text-ink-100" : "text-ink-300",
                    ].join(" ")}
                  >
                    {s.label}
                  </div>
                  <div
                    className={[
                      "text-sm transition-colors",
                      isCurrent ? "text-champagne-300" : "text-ink-300",
                    ].join(" ")}
                  >
                    {s.hint}
                  </div>
                </div>
              </li>
            );
          })}
        </ul>

        {currentStageMeta && (
          <motion.div
            key={currentStageMeta.name}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-8 rounded-xl bg-ink-800/50 px-5 py-4 font-mono text-xs text-ink-200"
          >
            <p className="label-eyebrow mb-2">Latest log</p>
            {logs.length === 0 ? (
              <p className="text-ink-300">…</p>
            ) : (
              <p className="break-words">{logs[logs.length - 1]}</p>
            )}
          </motion.div>
        )}
      </div>

      <p className="mt-6 text-center text-sm text-ink-300">
        The pose pass is the slow one —{" "}
        <span className="text-ink-200">~150 ms per frame</span> on your RTX 5080.
        Hang tight.
      </p>
    </div>
  );
}

function StageMarker({ state }: { state: "pending" | "active" | "done" }) {
  if (state === "done") {
    return (
      <div className="grid h-10 w-10 place-items-center rounded-xl bg-fairway-500/15 hairline">
        <svg viewBox="0 0 16 16" width="16" height="16" fill="none">
          <path
            d="M3 8 L7 12 L13 5"
            stroke="#6EE7A1"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    );
  }
  if (state === "active") {
    return (
      <div className="relative grid h-10 w-10 place-items-center rounded-xl bg-champagne-300/10 hairline">
        <motion.span
          className="absolute inset-0 rounded-xl"
          style={{ boxShadow: "0 0 0 1px rgba(230,196,136,0.4)" }}
          animate={{ opacity: [0.3, 0.9, 0.3] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="h-2 w-2 rounded-full bg-champagne-300"
          animate={{ scale: [1, 1.6, 1], opacity: [1, 0.5, 1] }}
          transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>
    );
  }
  return (
    <div className="grid h-10 w-10 place-items-center rounded-xl bg-ink-900/50 hairline">
      <span className="h-1.5 w-1.5 rounded-full bg-ink-400" />
    </div>
  );
}
