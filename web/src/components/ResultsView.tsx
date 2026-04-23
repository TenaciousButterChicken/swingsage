import { useState } from "react";
import { motion } from "framer-motion";
import type { AnalysisResult } from "../lib/types";

interface ResultsViewProps {
  result: AnalysisResult;
  onReset: () => void;
}

// Reference ranges for the metric rating — same numbers the LLM prompt uses.
const RANGES: Record<string, { low: number; high: number; unit: string; label: string; better?: "high" | "middle" }> = {
  shoulder_turn_deg:   { low: 80,  high: 100, unit: "°", label: "Shoulder turn",   better: "middle" },
  hip_turn_deg:        { low: 40,  high: 55,  unit: "°", label: "Hip turn",        better: "middle" },
  x_factor_deg:        { low: 35,  high: 50,  unit: "°", label: "X-factor",        better: "middle" },
  lead_arm_abduction_deg: { low: 80, high: 110, unit: "°", label: "Lead arm abduction", better: "middle" },
  lead_arm_flex_deg:   { low: 160, high: 180, unit: "°", label: "Lead arm flex",   better: "high" },
  spine_tilt_forward_deg: { low: 25, high: 40, unit: "°", label: "Spine tilt (fwd)", better: "middle" },
  spine_tilt_side_deg: { low: -10, high: 10,  unit: "°", label: "Spine side bend", better: "middle" },
};

type Rating = "in-range" | "near" | "out";

function rate(value: number | null, key: string): Rating | null {
  if (value === null || value === undefined) return null;
  const r = RANGES[key];
  if (!r) return null;
  if (value >= r.low && value <= r.high) return "in-range";
  const margin = Math.abs(r.high - r.low) * 0.2;
  if (value >= r.low - margin && value <= r.high + margin) return "near";
  return "out";
}

export default function ResultsView({ result, onReset }: ResultsViewProps) {
  return (
    <div className="space-y-10">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-6">
        <div>
          <p className="label-eyebrow">Analysis</p>
          <h2 className="mt-1 font-display text-display-lg tracking-tight text-ink-100">
            Your swing, read by the numbers.
          </h2>
          <p className="mt-3 max-w-xl text-sm text-ink-200">
            Total pipeline time{" "}
            <span className="font-mono text-ink-100">{result.elapsed_seconds}s</span>
            {" · "}
            {result.total_frames} frames at {result.fps.toFixed(1)} fps
            {" · "}
            impact on frame{" "}
            <span className="font-mono text-champagne-300">
              {result.metrics.phases.impact_frame}
            </span>
          </p>
        </div>
        <button className="btn-ghost" onClick={onReset}>
          Analyse another swing
        </button>
      </div>

      {/* Media row */}
      <div className="grid gap-6 lg:grid-cols-3">
        <motion.div
          className="card-glass overflow-hidden lg:col-span-2"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="flex items-center justify-between border-b border-ink-800/70 px-6 py-4">
            <div>
              <p className="label-eyebrow">Trimmed swing</p>
              <p className="mt-1 font-display text-lg text-ink-100">
                Auto-windowed ~3.5 s around impact
              </p>
            </div>
          </div>
          <video
            className="block w-full bg-black"
            src={result.artifacts.trimmed_mp4}
            controls
            playsInline
          />
        </motion.div>

        <motion.div
          className="card-glass flex flex-col"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1], delay: 0.05 }}
        >
          <div className="border-b border-ink-800/70 px-6 py-4">
            <p className="label-eyebrow">Key frames</p>
            <p className="mt-1 font-display text-lg text-ink-100">What the metrics saw</p>
          </div>
          <KeyframeGallery result={result} />
        </motion.div>
      </div>

      {/* Metrics dashboard */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1], delay: 0.1 }}
      >
        <div className="mb-5 flex items-end justify-between">
          <div>
            <p className="label-eyebrow">Biomechanics</p>
            <h3 className="mt-1 font-display text-2xl text-ink-100">Metric dashboard</h3>
          </div>
          <KinematicSequenceBadge result={result} />
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            metricKey="shoulder_turn_deg"
            value={result.metrics.at_top.shoulder_turn_deg}
            at="at Top"
          />
          <MetricCard
            metricKey="hip_turn_deg"
            value={result.metrics.at_top.hip_turn_deg}
            at="at Top"
          />
          <MetricCard
            metricKey="x_factor_deg"
            value={result.metrics.at_top.x_factor_deg}
            at="at Top"
          />
          <MetricCard
            metricKey="lead_arm_flex_deg"
            value={result.metrics.at_top.lead_arm_flex_deg}
            at="peak during backswing"
          />
          <MetricCard
            metricKey="lead_arm_abduction_deg"
            value={result.metrics.at_top.lead_arm_abduction_deg}
            at="at Top"
          />
          <MetricCard
            metricKey="spine_tilt_forward_deg"
            value={result.metrics.at_address.spine_tilt_forward_deg}
            at="at Address (median)"
          />
          <MetricCard
            metricKey="spine_tilt_side_deg"
            value={result.metrics.at_address.spine_tilt_side_deg}
            at="at Address (median)"
          />
        </div>
      </motion.div>

      {/* Coaching */}
      {result.coaching ? (
        <CoachingPanel coaching={result.coaching} />
      ) : (
        <div className="card-glass p-6">
          <p className="label-eyebrow text-ember-400">Coaching unavailable</p>
          <p className="mt-2 text-sm text-ink-200">
            LM Studio wasn't reachable. Start the server and load Qwen 3 14B, then try again.
          </p>
        </div>
      )}

      {/* SwingNet diagnostic */}
      <SwingNetDiagnostic result={result} />
    </div>
  );
}

function MetricCard({
  metricKey,
  value,
  at,
}: {
  metricKey: string;
  value: number | null;
  at: string;
}) {
  const r = RANGES[metricKey];
  const rating = rate(value, metricKey);
  const color =
    rating === "in-range"
      ? "text-fairway-400"
      : rating === "near"
        ? "text-champagne-300"
        : rating === "out"
          ? "text-ember-400"
          : "text-ink-300";

  return (
    <div className="card-glass p-5">
      <p className="label-eyebrow">{r?.label ?? metricKey}</p>
      <div className="mt-3 flex items-baseline gap-2">
        <span className={`font-display text-4xl tracking-tight ${color}`}>
          {value === null ? "—" : value.toFixed(1)}
        </span>
        {value !== null && r && <span className={`text-lg ${color}`}>{r.unit}</span>}
      </div>
      <p className="mt-1 text-xs text-ink-300">{at}</p>
      {r && (
        <div className="mt-4">
          <RangeBar value={value} low={r.low} high={r.high} rating={rating} />
          <p className="mt-2 font-mono text-[10px] text-ink-300">
            target {r.low}{r.unit} – {r.high}{r.unit}
          </p>
        </div>
      )}
    </div>
  );
}

function RangeBar({
  value,
  low,
  high,
  rating,
}: {
  value: number | null;
  low: number;
  high: number;
  rating: Rating | null;
}) {
  // Map the range onto a 0-100 scale with a ~40% in-range middle band.
  const span = Math.max(high - low, 0.01);
  const absMin = low - span * 0.8;
  const absMax = high + span * 0.8;
  const pct = value === null ? null : ((value - absMin) / (absMax - absMin)) * 100;
  const inLow = ((low - absMin) / (absMax - absMin)) * 100;
  const inHigh = ((high - absMin) / (absMax - absMin)) * 100;

  const color =
    rating === "in-range"
      ? "bg-fairway-500"
      : rating === "near"
        ? "bg-champagne-300"
        : rating === "out"
          ? "bg-ember-500"
          : "bg-ink-400";

  return (
    <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-ink-800">
      {/* In-range band */}
      <div
        className="absolute inset-y-0 rounded-full bg-fairway-500/15"
        style={{ left: `${inLow}%`, right: `${100 - inHigh}%` }}
      />
      {pct !== null && (
        <motion.div
          className={`absolute top-0 h-full w-1.5 ${color} rounded-full`}
          style={{ left: `calc(${Math.max(0, Math.min(100, pct))}% - 3px)` }}
          initial={{ scaleY: 0.4, opacity: 0 }}
          animate={{ scaleY: 1, opacity: 1 }}
          transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        />
      )}
    </div>
  );
}

function KeyframeGallery({ result }: { result: AnalysisResult }) {
  const items = [
    { label: "Window start", src: result.artifacts.keyframes.window_start },
    { label: "Top of backswing", src: result.artifacts.keyframes.peak_shoulder_top },
    { label: "Impact", src: result.artifacts.keyframes.impact },
    { label: "Window end", src: result.artifacts.keyframes.window_end },
  ];
  const [active, setActive] = useState(1);
  return (
    <div className="flex flex-1 flex-col">
      <div className="relative flex-1 bg-black">
        <motion.img
          key={items[active].src}
          src={items[active].src}
          alt={items[active].label}
          className="h-full w-full object-contain"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.35 }}
        />
        <div className="absolute bottom-3 left-3 rounded-full bg-ink-950/80 px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-champagne-300 hairline">
          {items[active].label}
        </div>
      </div>
      <div className="grid grid-cols-4 gap-2 border-t border-ink-800/70 p-3">
        {items.map((it, i) => (
          <button
            key={it.label}
            onClick={() => setActive(i)}
            className={[
              "overflow-hidden rounded-lg transition-all duration-200",
              i === active ? "ring-2 ring-champagne-300" : "opacity-70 hover:opacity-100",
            ].join(" ")}
            aria-label={it.label}
          >
            <img src={it.src} alt={it.label} className="h-16 w-full object-cover" />
          </button>
        ))}
      </div>
    </div>
  );
}

function KinematicSequenceBadge({ result }: { result: AnalysisResult }) {
  const { order_correct, peak_velocity_frames, ideal_order } = result.metrics.kinematic_sequence;
  return (
    <div className="card-glass flex items-center gap-4 px-5 py-3">
      <div className={order_correct ? "text-fairway-400" : "text-champagne-300"}>
        {order_correct ? <IconCheck /> : <IconAlert />}
      </div>
      <div>
        <p className="label-eyebrow">Kinematic sequence</p>
        <p className="mt-0.5 font-display text-base text-ink-100">
          {order_correct ? "Order is correct" : "Order is off"}
        </p>
        <p className="mt-1 font-mono text-[11px] text-ink-300">
          {ideal_order
            .map((k) => `${k}=${peak_velocity_frames[k] ?? "—"}`)
            .join("  ·  ")}
        </p>
      </div>
    </div>
  );
}

function CoachingPanel({ coaching }: { coaching: NonNullable<AnalysisResult["coaching"]> }) {
  return (
    <motion.div
      className="card-glass overflow-hidden"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1], delay: 0.15 }}
    >
      <div className="flex items-center justify-between border-b border-ink-800/70 px-6 py-4">
        <div>
          <p className="label-eyebrow">Coaching</p>
          <p className="mt-1 font-display text-lg text-ink-100">
            Qwen 3 14B read your numbers
          </p>
        </div>
        <ConfidenceBadge level={coaching.confidence} />
      </div>

      <div className="grid gap-8 p-6 lg:grid-cols-[1fr_1.2fr]">
        <div>
          <p className="label-eyebrow mb-3">Faults</p>
          {coaching.faults.length === 0 ? (
            <p className="text-sm text-ink-200">No major faults detected.</p>
          ) : (
            <ul className="space-y-2">
              {coaching.faults.map((f) => (
                <li
                  key={f}
                  className="rounded-xl bg-ember-500/10 px-4 py-2.5 text-sm text-ember-400 hairline"
                >
                  {f}
                </li>
              ))}
            </ul>
          )}

          <p className="label-eyebrow mb-3 mt-7">Diagnosis</p>
          <p className="text-sm leading-relaxed text-ink-100">{coaching.diagnosis}</p>
        </div>

        <div>
          <p className="label-eyebrow mb-3">Drills</p>
          <ul className="space-y-3">
            {coaching.drills.map((d, i) => (
              <li key={d.name} className="rounded-xl bg-ink-800/50 p-4 hairline">
                <div className="flex items-start gap-3">
                  <span className="font-mono text-xs text-champagne-300">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <div className="flex-1">
                    <p className="font-display text-base text-ink-100">{d.name}</p>
                    <p className="mt-1 text-sm text-ink-200">{d.why}</p>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </motion.div>
  );
}

function ConfidenceBadge({ level }: { level: "low" | "medium" | "high" }) {
  const color =
    level === "high"
      ? "text-fairway-400"
      : level === "medium"
        ? "text-champagne-300"
        : "text-ember-400";
  return (
    <div className="flex items-center gap-2">
      <span className="label-eyebrow">Confidence</span>
      <span className={`font-mono text-xs uppercase ${color}`}>{level}</span>
    </div>
  );
}

function SwingNetDiagnostic({ result }: { result: AnalysisResult }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="card-glass">
      <button
        className="flex w-full items-center justify-between px-6 py-4 text-left"
        onClick={() => setOpen((v) => !v)}
      >
        <div>
          <p className="label-eyebrow">Diagnostic</p>
          <p className="mt-1 font-display text-base text-ink-100">
            SwingNet events (not used for metrics)
          </p>
        </div>
        <span className="font-mono text-xs text-ink-300">{open ? "hide" : "show"}</span>
      </button>
      {open && (
        <div className="border-t border-ink-800/70 p-6">
          <table className="w-full text-left font-mono text-xs">
            <thead className="text-ink-300">
              <tr>
                <th className="pb-2">Event</th>
                <th className="pb-2">Frame</th>
                <th className="pb-2">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {result.swingnet_events.map((e) => (
                <tr key={e.name} className="border-t border-ink-800/50">
                  <td className="py-2 text-ink-100">{e.name}</td>
                  <td className="py-2 text-ink-200">{e.frame}</td>
                  <td
                    className={[
                      "py-2",
                      e.confidence >= 0.5 ? "text-fairway-400" : e.confidence >= 0.1 ? "text-champagne-300" : "text-ink-300",
                    ].join(" ")}
                  >
                    {(e.confidence * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function IconCheck() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
      <circle cx="11" cy="11" r="10" stroke="currentColor" strokeWidth="1.2" opacity="0.4" />
      <path d="M6 11 L9.5 14.5 L16 7.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}

function IconAlert() {
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
      <circle cx="11" cy="11" r="10" stroke="currentColor" strokeWidth="1.2" opacity="0.4" />
      <path d="M11 6 V12 M11 16 V16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}
