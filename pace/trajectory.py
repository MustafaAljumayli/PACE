"""
Trajectory: the turn-by-turn record of an agent episode.

Signals tracked per turn (5 total):
  S1: answer_similarity     — cosine sim(answer_t, answer_{t-1})
  S2: info_gain             — cosine dist(new_context, cumulative)
  S3: token_entropy         — sequence entropy from token logprobs (fallback: text entropy)
  S4: tool_entropy          — Shannon entropy of tool distribution
  S5: agent_agreement       — cross-agent NLI similarity (set externally)

Each signal has level, velocity (Δ), and acceleration (Δ²).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

SIGNAL_NAMES = [
    "answer_similarity",
    "info_gain",
    "token_entropy",
    "tool_entropy",
    "agent_agreement",
]


@dataclass
class TurnState:
    """Snapshot of a single turn in a multi-turn episode."""

    turn_number: int
    answer: str
    retrieved_context: str = ""
    cumulative_context: str = ""
    tool_called: str = ""
    reasoning_text: str = ""
    token_count: int = 0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # S1
    answer_similarity: float | None = None
    answer_similarity_v: float | None = None
    answer_similarity_a: float | None = None
    # S2
    info_gain: float | None = None
    info_gain_v: float | None = None
    info_gain_a: float | None = None
    # S3
    token_entropy: float | None = None
    token_entropy_v: float | None = None
    token_entropy_a: float | None = None
    # S4
    tool_entropy: float | None = None
    tool_entropy_v: float | None = None
    tool_entropy_a: float | None = None
    # S5 — set by DualAgentRunner, not by SignalComputer
    agent_agreement: float | None = None
    agent_agreement_v: float | None = None
    agent_agreement_a: float | None = None

    # Backward compat
    @property
    def answer_sim_velocity(self) -> float | None:
        return self.answer_similarity_v

    @property
    def info_gain_velocity(self) -> float | None:
        return self.info_gain_v

    def get_signal(self, name: str) -> float | None:
        return getattr(self, name, None)

    def get_velocity(self, name: str) -> float | None:
        return getattr(self, f"{name}_v", None)

    def get_acceleration(self, name: str) -> float | None:
        return getattr(self, f"{name}_a", None)


class Trajectory:
    """Ordered record of all turns in an episode."""

    def __init__(self, query: str, episode_id: str = ""):
        self.query = query
        self.episode_id = episode_id
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
    def latest_answer(self) -> str:
        return self.turns[-1].answer if self.turns else ""

    def answers(self) -> list[str]:
        return [t.answer for t in self.turns]

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
            "query": self.query,
            "episode_id": self.episode_id,
            "num_turns": len(self.turns),
            "total_tokens": self.total_tokens(),
            "total_latency_ms": self.total_latency_ms(),
            "turns": [
                {
                    "turn": t.turn_number,
                    "answer": t.answer[:200],
                    "tool": t.tool_called,
                    **{s: t.get_signal(s) for s in SIGNAL_NAMES},
                    **{f"{s}_v": t.get_velocity(s) for s in SIGNAL_NAMES},
                    "token_count": t.token_count,
                    "latency_ms": t.latency_ms,
                }
                for t in self.turns
            ],
        }
