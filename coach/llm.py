"""Coaching LLM client — talks to LM Studio's OpenAI-compatible endpoint.

Takes a biomechanical metrics dict (from analytics.joint_angles.metrics_to_coach_dict)
plus optional VTrack ball/club data, and returns structured coaching feedback
(faults, diagnosis, drills) as JSON.

Designed around Qwen 3 14B via LM Studio at http://127.0.0.1:1234/v1 by default.
The OpenAI SDK speaks the same wire format so we can just point it at a local
base URL with a dummy API key.

Note: LM Studio's response_format accepts only "text" or "json_schema" — NOT
the vanilla "json_object" that upstream OpenAI uses. We pass a schema so Qwen
returns structured output we can parse without defensive fallbacks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

import openai

from capture.config import load_config


# The prompt is deliberately compact and schema-strict. Qwen 3's IFEval / MMLU-Pro
# scores mean it reliably produces JSON when asked — but prompt discipline still
# matters. Every metric field is passed explicitly so the model never guesses.
_SYSTEM_PROMPT = """You are a seasoned PGA-level golf coach reviewing a single swing. You see \
only the numbers below — you do NOT see the video. Your job is to give concise, \
actionable feedback grounded strictly in these numbers. Do not invent metrics.

Return a JSON object with exactly these keys:
  "faults": array of 0-3 short strings naming the biggest issues (e.g. \
"early extension", "over-the-top", "lead arm bent at top"). Empty array \
means no major faults detected.
  "diagnosis": 1-2 sentence plain-English explanation of what the numbers tell us.
  "drills": array of 1-3 drill suggestions, each a dict with "name" (string) \
and "why" (string, 1 sentence linking the drill to a specific metric).
  "confidence": "low" | "medium" | "high" — how much the data supports the \
diagnosis (low if any key metric is null or the kinematic window is too short).

All "at_top" and "at_address" metrics are computed as extrema/medians across
the appropriate swing phase (e.g., peak |shoulder rotation| over the backswing,
median spine tilt over the pre-takeaway setup). They are NOT sampled at a single
event frame, so they are insensitive to small event-localization errors and
you can trust them even when a specific event frame would be off by a few frames.

All rotations in "at_top" are reported as unsigned magnitudes.

Reference ranges (amateur → pro):
  shoulder_turn_deg at top: 80-100 (magnitude of rotation from address)
  hip_turn_deg at top: 40-55
  x_factor_deg at top: 35-50 (bigger = more power if controlled)
  spine_tilt_forward_deg at address: 25-40 (positive = bent toward ball)
  lead_arm_abduction_deg at top: 80-110 (shoulder elevation; ~90 = arm horizontal; NOT straightness)
  lead_arm_flex_deg at top: 160-180 (elbow extension; 180 = fully straight arm)
  Kinematic sequence: pelvis -> chest -> lead_arm -> lead_wrist (ground up)

Confidence guidance: set "low" if any required field is null; "medium" if
kinematic_sequence.order_correct is false; otherwise "high".
"""


# Separate, conversational system prompt for the follow-up chat. The initial
# analysis runs in strict-JSON mode; the chat is free-form explanation grounded
# in the same numbers. Giving Qwen the numbers inline means every answer can
# reference Neil's actual values instead of generic textbook ranges.
_CHAT_SYSTEM_PROMPT = """You are a seasoned PGA-level golf coach having a \
conversation with a player who just got their swing analyzed. You have access \
to their biomechanical numbers from this one swing, and the initial faults / \
diagnosis / drills you already gave them.

Your role in this chat:
- Answer follow-up questions grounded in THEIR numbers (quote specific values).
- Explain golf terminology in plain language; the player is not a tour pro.
- When they ask "what is X?" (e.g. "what is X-factor?"), define the concept \
first in one sentence, then tie it to their specific value and what it means.
- When they ask "why is my X off?", explain the likely causes and name a \
concrete drill or check they can do.
- Keep answers tight: 2-4 short paragraphs max. No preamble, no sign-offs.
- Don't invent data. If they ask about something you weren't given (ball flight, \
club path, tempo), say so and suggest what data would answer it.

Reference ranges (amateur target -> pro):
  shoulder_turn_deg at top: 80-100 (magnitude of upper body rotation)
  hip_turn_deg at top: 40-55
  x_factor_deg at top: 35-50 (shoulder minus hip; bigger = more potential power if controlled)
  spine_tilt_forward_deg at address: 25-40 (bent toward ball)
  lead_arm_abduction_deg at top: 80-110 (shoulder elevation; ~90 = arm horizontal)
  lead_arm_flex_deg at top: 160-180 (elbow extension; 180 = fully straight arm)
  Kinematic sequence ideal: pelvis -> chest -> lead_arm -> lead_wrist (ground up)

Lengths and signs:
- All "at_top" rotations are unsigned magnitudes.
- spine_tilt_side positive = tilted away from target (right for right-handers).
- All metrics are extrema/medians over the relevant swing phase, so they are \
robust to small event-frame detection errors.
"""


def _chat_context_block(
    metrics: dict[str, Any],
    coaching: dict[str, Any] | None,
) -> str:
    """Format the swing's numbers + initial coaching as a JSON blob for the
    chat model's system message. Keeps every field explicit so the model can
    quote exact values back to the user."""
    ctx: dict[str, Any] = {"biomechanics": metrics}
    if coaching:
        ctx["initial_coaching"] = {
            "faults": coaching.get("faults", []),
            "diagnosis": coaching.get("diagnosis", ""),
            "drills": coaching.get("drills", []),
            "confidence": coaching.get("confidence", ""),
        }
    return json.dumps(ctx, indent=2)


_COACHING_SCHEMA: dict[str, Any] = {
    "name": "coaching_feedback",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "faults": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
                "description": "0-3 short strings naming the biggest issues",
            },
            "diagnosis": {
                "type": "string",
                "description": "1-2 sentence plain-English explanation grounded in the metrics",
            },
            "drills": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "why": {"type": "string", "description": "1 sentence linking drill to metric"},
                    },
                    "required": ["name", "why"],
                    "additionalProperties": False,
                },
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
        },
        "required": ["faults", "diagnosis", "drills", "confidence"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class CoachingFeedback:
    faults: list[str]
    diagnosis: str
    drills: list[dict[str, str]]
    confidence: str
    raw_json: str


class CoachClient:
    """Thin wrapper around an OpenAI-compatible chat endpoint (LM Studio)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str = "lm-studio",  # LM Studio ignores the key but the SDK requires one
        timeout: float = 120.0,
    ) -> None:
        cfg = load_config()
        self.base_url = base_url or cfg.llm_api_base
        self.model = model or cfg.llm_model
        self.client = openai.OpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout)

    def is_alive(self) -> bool:
        """Quick ping — returns True if the server is reachable and has at least one model."""
        try:
            models = self.client.models.list()
            return bool(list(models))
        except Exception:
            return False

    def coach(
        self,
        metrics: dict[str, Any],
        ball_data: dict[str, Any] | None = None,
        temperature: float = 0.2,
    ) -> CoachingFeedback:
        """Run one coaching call. Raises openai.APIError on transport/server failure."""
        user_payload: dict[str, Any] = {"biomechanics": metrics}
        if ball_data:
            user_payload["ball_flight"] = ball_data

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, indent=2)},
            ],
            response_format={"type": "json_schema", "json_schema": _COACHING_SCHEMA},
            temperature=temperature,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        return CoachingFeedback(
            faults=list(data.get("faults", []) or []),
            diagnosis=str(data.get("diagnosis", "")),
            drills=list(data.get("drills", []) or []),
            confidence=str(data.get("confidence", "low")),
            raw_json=raw,
        )

    def stream_chat(
        self,
        metrics: dict[str, Any],
        coaching: dict[str, Any] | None,
        messages: list[dict[str, str]],
        temperature: float = 0.4,
    ) -> Iterator[str]:
        """Yield assistant response tokens for a follow-up conversation about
        the just-analyzed swing. ``messages`` is the user/assistant history
        (no system role); this method injects the system prompt + numbers.

        Slightly higher temperature than the initial coaching call because
        explanations benefit from some variety, while the JSON analysis
        benefited from determinism."""
        system_messages: list[dict[str, str]] = [
            {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "Swing data for this conversation (JSON):\n"
                    + _chat_context_block(metrics, coaching)
                ),
            },
        ]
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=system_messages + list(messages),
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                yield text


def coach_once(
    metrics: dict[str, Any],
    ball_data: dict[str, Any] | None = None,
    **client_kwargs,
) -> CoachingFeedback:
    """Convenience one-shot — build a CoachClient and run a single request."""
    return CoachClient(**client_kwargs).coach(metrics, ball_data=ball_data)


if __name__ == "__main__":
    import sys

    # Windows console defaults to cp1252; force UTF-8 so Qwen's arrows / dashes
    # don't crash print(). errors="replace" keeps stray characters from killing
    # the smoke test.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    # Synthetic metrics for a smoke test (matches metrics_to_coach_dict format).
    demo_metrics = {
        "events": {"address_frame": 74, "top_frame": 114, "impact_frame": 143},
        "at_address": {"spine_tilt_forward_deg": 32.0, "spine_tilt_side_deg": -3.0},
        "at_top": {
            "shoulder_turn_deg": 92.0,
            "hip_turn_deg": 48.0,
            "x_factor_deg": 44.0,
            "lead_arm_abduction_deg": 172.0,
        },
        "kinematic_sequence": {
            "peak_velocity_frames": {
                "pelvis": 118, "chest": 125, "lead_arm": 132, "lead_wrist": 141,
            },
            "order_correct": True,
            "ideal_order": ["pelvis", "chest", "lead_arm", "lead_wrist"],
        },
    }

    client = CoachClient()
    if not client.is_alive():
        print(f"LM Studio not reachable at {client.base_url}. Start it with:", file=sys.stderr)
        print("  lms server start --port 1234", file=sys.stderr)
        print("  lms load bartowski/Qwen_Qwen3-14B-GGUF@Q5_K_M", file=sys.stderr)
        sys.exit(1)

    fb = client.coach(demo_metrics)
    print(f"Model: {client.model}")
    print(f"Confidence: {fb.confidence}")
    print(f"Faults: {fb.faults}")
    print(f"Diagnosis: {fb.diagnosis}")
    print("Drills:")
    for d in fb.drills:
        print(f"  - {d.get('name', '?')}: {d.get('why', '')}")
