"""
Trajectory: the turn-by-turn record of an agent episode.

Signals tracked per turn (6 total — chosen to target LiC failure modes):
  goal_drift          — cos(response, goal): drift from the original question
  shard_coverage      — min cos(response, shard_i): forgotten sub-goals
  contradiction       — max P_NLI(contradiction | response, past_j): long-range
  response_stability  — cos(response_t, response_{t-1}): abrupt reversals
  token_entropy       — mean per-token Shannon entropy (logprob path only)
  repetition          — windowed, length-normalised n-gram overlap: stuck loops

Each signal has level, velocity (Δ), and acceleration (Δ²).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

SIGNAL_NAMES = [
    "goal_drift",
    "shard_coverage",
    "contradiction",
    "response_stability",
    "token_entropy",
    "repetition",
]


@dataclass
class TurnState:
    """Snapshot of a single turn in a multi-turn episode."""

    turn_number: int
    response: str = ""
    answer: str = ""
    retrieved_context: str = ""
    cumulative_context: str = ""
    tool_called: str = ""
    reasoning_text: str = ""
    token_count: int = 0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Signals ──
    goal_drift: float | None = None
    goal_drift_v: float | None = None
    goal_drift_a: float | None = None

    shard_coverage: float | None = None
    shard_coverage_v: float | None = None
    shard_coverage_a: float | None = None

    contradiction: float | None = None
    contradiction_v: float | None = None
    contradiction_a: float | None = None

    response_stability: float | None = None
    response_stability_v: float | None = None
    response_stability_a: float | None = None

    token_entropy: float | None = None
    token_entropy_v: float | None = None
    token_entropy_a: float | None = None

    repetition: float | None = None
    repetition_v: float | None = None
    repetition_a: float | None = None

    def get_signal(self, name: str) -> float | None:
        return getattr(self, name, None)

    def get_velocity(self, name: str) -> float | None:
        return getattr(self, f"{name}_v", None)

    def get_acceleration(self, name: str) -> float | None:
        return getattr(self, f"{name}_a", None)

    def signals_dict(self) -> dict[str, float | None]:
        return {s: self.get_signal(s) for s in SIGNAL_NAMES}

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "turn": self.turn_number,
            "response": self.response[:200],
        }
        for s in SIGNAL_NAMES:
            d[s] = self.get_signal(s)
            d[f"{s}_v"] = self.get_velocity(s)
            d[f"{s}_a"] = self.get_acceleration(s)
        return d


class Trajectory:
    """Ordered record of all turns in an episode."""

    def __init__(self, goal: str = "", episode_id: str = "", shards: list[str] | None = None):
        self.goal = goal
        self.episode_id = episode_id
        self.shards = shards or []
        self.turns: list[TurnState] = []
        self._start_time = time.time()

    def add_turn(self, state: TurnState) -> None:
        self.turns.append(state)

    @property
    def current_turn(self) -> int:
        return len(self.turns)

    @property
    def latest(self) -> TurnState | None:
        return self.turns[-1] if self.turns else None

    @property
    def latest_response(self) -> str:
        return self.turns[-1].response if self.turns else ""

    def responses(self) -> list[str]:
        return [t.response for t in self.turns]

    def tools_used(self) -> list[str]:
        return [t.tool_called for t in self.turns if t.tool_called]

    def total_tokens(self) -> int:
        return sum(t.token_count for t in self.turns)

    def total_latency_ms(self) -> float:
        return sum(t.latency_ms for t in self.turns)

    def signal_series(self, signal_name: str) -> list[float]:
        return [
            getattr(t, signal_name)
            for t in self.turns
            if getattr(t, signal_name, None) is not None
        ]

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "episode_id": self.episode_id,
            "num_turns": len(self.turns),
            "turns": [t.to_dict() for t in self.turns],
        }
