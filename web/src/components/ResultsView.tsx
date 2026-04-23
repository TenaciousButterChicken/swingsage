import { useState } from "react";
import { motion } from "framer-motion";
import type { AnalysisResult, BallFlight, TrimInfo } from "../lib/types";

interface ResultsViewProps {
  result: AnalysisResult;
  jobId: string;
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

export default function ResultsView({ result, jobId, onReset }: ResultsViewProps) {
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
          <VideoPanel artifacts={result.artifacts} trim={result.trim} />
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

      {/* Ball flight — only rendered when a VTrack shot was paired with this video */}
      {result.ball_flight && <BallFlightPanel ballFlight={result.ball_flight} />}

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

      {/* Chat — ask Qwen follow-up questions about your numbers */}
      {result.coaching && <ChatPanel jobId={jobId} />}

      {/* SwingNet diagnostic */}
      <SwingNetDiagnostic result={result} />
    </div>
  );
}

function VideoPanel({
  artifacts,
  trim,
}: {
  artifacts: AnalysisResult["artifacts"];
  trim: TrimInfo;
}) {
  const [mode, setMode] = useState<"raw" | "pose">("raw");
  const src = mode === "raw" ? artifacts.trimmed_mp4 : artifacts.pose_overlay_mp4;
  const windowLabel = trim.used_fallback
    ? `Full clip (${trim.window_seconds.toFixed(1)} s) — auto-trim confidence ${trim.confidence.toFixed(1)}× too low`
    : `Auto-windowed ${trim.window_seconds.toFixed(1)} s around impact (confidence ${trim.confidence.toFixed(1)}×)`;

  return (
    <>
      <div className="flex items-center justify-between border-b border-ink-800/70 px-6 py-4">
        <div>
          <p className="label-eyebrow">
            {trim.used_fallback ? "Untrimmed swing" : "Trimmed swing"}
          </p>
          <p className="mt-1 font-display text-lg text-ink-100">
            {mode === "raw"
              ? windowLabel
              : "NLF skeleton overlay — 24 joints per frame"}
          </p>
        </div>
        <div className="flex items-center rounded-full hairline bg-ink-800/50 p-1">
          <ToggleBtn active={mode === "raw"} onClick={() => setMode("raw")}>
            Raw
          </ToggleBtn>
          <ToggleBtn active={mode === "pose"} onClick={() => setMode("pose")}>
            Pose overlay
          </ToggleBtn>
        </div>
      </div>
      <video
        key={src}
        className="block w-full bg-black"
        src={src}
        controls
        playsInline
      />
    </>
  );
}

function ToggleBtn({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={[
        "rounded-full px-3.5 py-1.5 font-mono text-[11px] uppercase tracking-widest transition-all duration-200",
        active
          ? "bg-champagne-300 text-ink-950"
          : "text-ink-300 hover:text-ink-100",
      ].join(" ")}
    >
      {children}
    </button>
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
  // When auto-trim falls back, window_start/window_end are just the first
  // and last frames of the source clip — not meaningful swing boundaries.
  // Relabel them so Neil doesn't read "WINDOW END" and think the pipeline
  // picked his finish pose.
  const fell = result.trim.used_fallback;
  const items = [
    {
      label: fell ? "Clip start" : "Window start",
      src: result.artifacts.keyframes.window_start,
    },
    { label: "Top of backswing", src: result.artifacts.keyframes.peak_shoulder_top },
    { label: "Impact", src: result.artifacts.keyframes.impact },
    {
      label: fell ? "Clip end" : "Window end",
      src: result.artifacts.keyframes.window_end,
    },
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

function BallFlightPanel({ ballFlight }: { ballFlight: BallFlight }) {
  // Convert the SI-stored values back to what golfers actually think in.
  const mps_to_mph = 2.2369362920544;
  const m_to_yds = 1.0936132983377;

  const mph = (v: number | null) => (v === null ? null : v * mps_to_mph);
  const yds = (v: number | null) => (v === null ? null : v * m_to_yds);

  // Pretty-print the capture timestamp as local HH:MM:SS so the user can
  // cross-check against when they hit the shot.
  const capturedLocal = (() => {
    try {
      const d = new Date(ballFlight.captured_at);
      return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch {
      return ballFlight.captured_at;
    }
  })();

  const shape = classifyShotShape(
    ballFlight.face_angle_deg,
    ballFlight.spin_axis_deg,
  );

  const ballFields: StatSpec[] = [
    { label: "Ball speed", value: mph(ballFlight.ball_speed_mps), unit: "mph", digits: 1 },
    { label: "Carry", value: yds(ballFlight.carry_distance_m), unit: "yds", digits: 1 },
    { label: "Launch", value: ballFlight.launch_angle_deg, unit: "°", digits: 1 },
    { label: "Spin axis", value: ballFlight.spin_axis_deg, unit: "°", digits: 1 },
    { label: "Back spin", value: ballFlight.back_spin_rpm, unit: "rpm", digits: 0 },
    { label: "Side spin", value: ballFlight.side_spin_rpm, unit: "rpm", digits: 0 },
  ];
  const clubFields: StatSpec[] = [
    { label: "Club speed", value: mph(ballFlight.club_speed_mps), unit: "mph", digits: 1 },
    { label: "Smash", value: ballFlight.smash_factor, unit: "", digits: 2 },
    { label: "Club path", value: ballFlight.club_path_deg, unit: "°", digits: 1 },
    { label: "Face to target", value: ballFlight.face_angle_deg, unit: "°", digits: 1 },
    { label: "Attack angle", value: ballFlight.attack_angle_deg, unit: "°", digits: 1 },
  ];

  const visibleBall = ballFields.filter((f) => f.value !== null);
  const visibleClub = clubFields.filter((f) => f.value !== null);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1], delay: 0.12 }}
    >
      <div className="mb-5 flex items-end justify-between gap-4">
        <div>
          <p className="label-eyebrow">Ball flight</p>
          <h3 className="mt-1 font-display text-2xl text-ink-100">Paired VTrack shot</h3>
          <p className="mt-1 font-mono text-[11px] text-ink-300">captured at {capturedLocal}</p>
        </div>
        {shape && <ShotShapeBadge shape={shape} />}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="card-glass p-6">
          <p className="label-eyebrow mb-4">Ball</p>
          <div className="grid gap-4 sm:grid-cols-3">
            {visibleBall.map((f) => (
              <StatTile key={f.label} {...f} />
            ))}
          </div>
        </div>
        {visibleClub.length > 0 && (
          <div className="card-glass p-6">
            <p className="label-eyebrow mb-4">Club</p>
            <div className="grid gap-4 sm:grid-cols-3">
              {visibleClub.map((f) => (
                <StatTile key={f.label} {...f} />
              ))}
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}

// ─── Shot shape classifier ────────────────────────────────────────────
// Right-hander convention (we don't store handedness per-shot yet):
//   face_angle_deg > 0  → face OPEN to target (right)
//   face_angle_deg < 0  → face CLOSED to target (left)
//   spin_axis_deg > 0   → right-tilting axis → ball curves right (fade/slice)
//   spin_axis_deg < 0   → left-tilting axis → ball curves left (draw/hook)
//
// Start direction is dominated by face angle (~85% face, 15% path in the
// new ball flight laws). Curve magnitude is read off the spin axis.
type Curve = "hook" | "draw" | "straight" | "fade" | "slice";
type Start = "pull" | "straight" | "push";

export interface ShotShape {
  label: string;
  start: Start;
  curve: Curve;
  severity: "clean" | "off-line" | "wild";
}

function classifyShotShape(
  faceAngleDeg: number | null,
  spinAxisDeg: number | null,
): ShotShape | null {
  if (faceAngleDeg === null || spinAxisDeg === null) return null;

  // Start direction thresholds
  const face = faceAngleDeg;
  const start: Start =
    face < -2 ? "pull" : face > 2 ? "push" : "straight";

  // Curve thresholds on spin axis
  const axis = spinAxisDeg;
  const abs = Math.abs(axis);
  let curve: Curve;
  if (abs <= 3) curve = "straight";
  else if (abs <= 12) curve = axis > 0 ? "fade" : "draw";
  else curve = axis > 0 ? "slice" : "hook";

  // Compose a human label. Pure cases get their own word; mixed cases get
  // a hyphenated composite like "Pull-fade" or "Push-draw" that golfers
  // actually use.
  let label: string;
  if (start === "straight" && curve === "straight") {
    label = "Straight";
  } else if (start === "straight") {
    label = capitalize(curve);
  } else if (curve === "straight") {
    label = capitalize(start);
  } else {
    label = `${capitalize(start)}-${curve}`;
  }

  // Severity: informs color. Straight / near-straight is clean;
  // moderate off-line curves are "off-line"; slice/hook are "wild".
  const severity: ShotShape["severity"] =
    curve === "straight" && start === "straight"
      ? "clean"
      : curve === "slice" || curve === "hook"
        ? "wild"
        : "off-line";

  return { label, start, curve, severity };
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function ShotShapeBadge({ shape }: { shape: ShotShape }) {
  const color =
    shape.severity === "clean"
      ? "text-fairway-400 border-fairway-500/40 bg-fairway-500/10"
      : shape.severity === "off-line"
        ? "text-champagne-300 border-champagne-300/40 bg-champagne-300/10"
        : "text-ember-400 border-ember-500/40 bg-ember-500/10";

  return (
    <div
      className={[
        "flex items-center gap-3 rounded-2xl border px-4 py-2.5",
        color,
      ].join(" ")}
    >
      <ShapeArrow shape={shape} />
      <div>
        <p className="label-eyebrow">Shape</p>
        <p className="font-display text-lg leading-tight">{shape.label}</p>
      </div>
    </div>
  );
}

function ShapeArrow({ shape }: { shape: ShotShape }) {
  // Tiny SVG: a 32×32 sketch showing the flight path. The ball starts at
  // the bottom and travels up; x-offset at the bottom encodes the start
  // direction, the curve direction encodes draw vs fade.
  const startX =
    shape.start === "pull" ? 6 : shape.start === "push" ? 26 : 16;
  const endX =
    shape.curve === "hook" ? 4 :
    shape.curve === "draw" ? 11 :
    shape.curve === "fade" ? 21 :
    shape.curve === "slice" ? 28 : startX;
  // Control-point for the quadratic curve — bulge to one side based on curve.
  const controlX =
    shape.curve === "hook" || shape.curve === "draw"
      ? Math.min(startX, endX) - 4
      : shape.curve === "fade" || shape.curve === "slice"
        ? Math.max(startX, endX) + 4
        : (startX + endX) / 2;

  return (
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
      {/* Baseline target line (dashed) */}
      <line x1="16" y1="4" x2="16" y2="28" stroke="currentColor" strokeWidth="0.8" strokeDasharray="1 2" opacity="0.35" />
      {/* Flight path */}
      <path
        d={`M ${startX} 28 Q ${controlX} 16 ${endX} 4`}
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        fill="none"
      />
      {/* Ball at the landing */}
      <circle cx={endX} cy="4" r="1.8" fill="currentColor" />
    </svg>
  );
}

interface StatSpec {
  label: string;
  value: number | null;
  unit: string;
  digits: number;
}

function StatTile({ label, value, unit, digits }: StatSpec) {
  return (
    <div>
      <p className="label-eyebrow">{label}</p>
      <div className="mt-2 flex items-baseline gap-1.5">
        <span className="font-display text-2xl tracking-tight text-ink-100">
          {value === null ? "—" : value.toFixed(digits)}
        </span>
        {value !== null && unit && (
          <span className="text-sm text-ink-300">{unit}</span>
        )}
      </div>
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

type ChatRole = "user" | "assistant";
interface ChatMsg {
  role: ChatRole;
  content: string;
}

const STARTER_QUESTIONS = [
  "What is X-factor and why does mine matter?",
  "Why is my shoulder turn too big — how do I fix it?",
  "What drill should I do first this week?",
  "Explain my kinematic sequence in plain language.",
];

function ChatPanel({ jobId }: { jobId: string }) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send(text: string) {
    const trimmed = text.trim();
    if (!trimmed || streaming) return;
    setError(null);
    const next: ChatMsg[] = [...messages, { role: "user", content: trimmed }];
    setMessages(next);
    setInput("");
    setStreaming(true);

    // Start an empty assistant bubble we'll fill in as tokens arrive.
    setMessages((m) => [...m, { role: "assistant", content: "" }]);

    try {
      const resp = await fetch(`/api/chat/${jobId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: next }),
      });
      if (!resp.ok || !resp.body) {
        const detail = await resp.text().catch(() => resp.statusText);
        throw new Error(detail || `HTTP ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let acc = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        acc += decoder.decode(value, { stream: true });
        setMessages((m) => {
          const copy = m.slice();
          copy[copy.length - 1] = { role: "assistant", content: acc };
          return copy;
        });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      // Roll back the empty assistant bubble we optimistically pushed.
      setMessages((m) => (m[m.length - 1]?.content ? m : m.slice(0, -1)));
    } finally {
      setStreaming(false);
    }
  }

  return (
    <motion.div
      className="card-glass overflow-hidden"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1], delay: 0.2 }}
    >
      <div className="border-b border-ink-800/70 px-6 py-4">
        <p className="label-eyebrow">Ask the coach</p>
        <p className="mt-1 font-display text-lg text-ink-100">
          Follow-up questions about your numbers
        </p>
        <p className="mt-1 text-xs text-ink-300">
          Qwen 3 14B has your biomechanics and its initial diagnosis in context.
          Ask what a metric means, why yours is off, or what to work on first.
        </p>
      </div>

      {messages.length === 0 && (
        <div className="flex flex-wrap gap-2 border-b border-ink-800/70 px-6 py-4">
          {STARTER_QUESTIONS.map((q) => (
            <button
              key={q}
              onClick={() => send(q)}
              className="rounded-full hairline bg-ink-800/40 px-3.5 py-1.5 text-xs text-ink-200 transition-colors hover:bg-ink-800/80 hover:text-ink-100"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {messages.length > 0 && (
        <div className="flex flex-col gap-4 px-6 py-5">
          {messages.map((m, i) => (
            <div
              key={i}
              className={
                m.role === "user"
                  ? "self-end max-w-[80%] rounded-2xl bg-champagne-300 px-4 py-2.5 text-sm text-ink-950"
                  : "self-start max-w-[85%] rounded-2xl bg-ink-800/60 px-4 py-2.5 text-sm leading-relaxed text-ink-100 hairline whitespace-pre-wrap"
              }
            >
              {m.content || (m.role === "assistant" && streaming && i === messages.length - 1 ? (
                <span className="text-ink-300">Thinking…</span>
              ) : null)}
            </div>
          ))}
        </div>
      )}

      {error && (
        <p className="border-t border-ink-800/70 px-6 py-3 text-xs text-ember-400">
          {error}
        </p>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex items-center gap-3 border-t border-ink-800/70 px-6 py-4"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={streaming ? "Waiting for reply…" : "Ask about your swing…"}
          disabled={streaming}
          className="flex-1 bg-transparent font-sans text-sm text-ink-100 placeholder:text-ink-300 focus:outline-none"
        />
        <button
          type="submit"
          disabled={streaming || !input.trim()}
          className="rounded-full bg-champagne-300 px-4 py-1.5 font-mono text-[11px] uppercase tracking-widest text-ink-950 transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
        >
          {streaming ? "…" : "Send"}
        </button>
      </form>
    </motion.div>
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
