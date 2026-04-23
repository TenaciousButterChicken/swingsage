// Types mirror the shapes the FastAPI server emits. Keep in sync with
// server/main.py's _run_pipeline events and job.result payload.

export type StageName =
  | "decode"
  | "pose"
  | "trim"
  | "swingnet"
  | "metrics"
  | "coaching";

export interface StageEvent {
  type: "stage";
  name: StageName;
  label: string;
  total_frames?: number;
}

export interface LogEvent {
  type: "log";
  message: string;
}

export interface DoneEvent {
  type: "done";
  result: AnalysisResult;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type WsEvent = StageEvent | LogEvent | DoneEvent | ErrorEvent;

export interface SwingNetEvent {
  name: string;
  frame: number;
  confidence: number;
}

export interface Drill {
  name: string;
  why: string;
}

export interface Coaching {
  model: string;
  confidence: "low" | "medium" | "high";
  faults: string[];
  diagnosis: string;
  drills: Drill[];
}

export interface Metrics {
  phases: {
    impact_frame: number;
    peak_shoulder_frame: number;
    setup_end_frame: number;
  };
  at_address: {
    spine_tilt_forward_deg: number | null;
    spine_tilt_side_deg: number | null;
  };
  at_top: {
    shoulder_turn_deg: number | null;
    hip_turn_deg: number | null;
    x_factor_deg: number | null;
    lead_arm_abduction_deg: number | null;
    lead_arm_flex_deg: number | null;
  };
  kinematic_sequence: {
    peak_velocity_frames: Record<string, number>;
    order_correct: boolean;
    ideal_order: string[];
  };
}

export interface Artifacts {
  trimmed_mp4: string;
  pose_overlay_mp4: string;
  keyframes: {
    window_start: string;
    peak_shoulder_top: string;
    impact: string;
    window_end: string;
  };
}

export interface TrimInfo {
  used_fallback: boolean;
  confidence: number;
  window_start: number;
  window_end: number;
  window_seconds: number;
}

export interface AnalysisResult {
  elapsed_seconds: number;
  fps: number;
  total_frames: number;
  trim: TrimInfo;
  artifacts: Artifacts;
  swingnet_events: SwingNetEvent[];
  metrics: Metrics;
  coaching: Coaching | null;
}
