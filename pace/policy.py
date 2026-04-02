"""
PACEPolicy: the decision engine.

Decisions: CONTINUE, STOP, REWIND, SCALE.

Key changes from v1:
  - Relative thresholds: convergence is measured against recent variance,
    not absolute epsilon values. Addresses the "how did you pick 0.95?" question.
  - Signal-mask-aware: only checks signals that are active in the current
    ablation condition. If signal_mask={"answer_similarity"}, the policy
    only uses S1 for its decisions.
  - All 5 signals contribute to convergence/scale decisions when active.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from pace.trajectory import Trajectory, SIGNAL_NAMES


class Decision(Enum):
    CONTINUE = auto()
    STOP = auto()
    REWIND = auto()
    SCALE = auto()


@dataclass
class PolicyResult:
    decision: Decision
    reason: str
    rewind_to_turn: int | None = None
    confidence: float = 0.0
    signals_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyConfig:
    """
    Threshold configuration for PACE.

    Relative thresholds:
      - similarity_threshold: still absolute (0-1 cosine range is already normalized)
      - info_gain_floor: relative to recent variance when relative_thresholds=True
      - token_entropy_floor: stop if token entropy drops below this (confident)
      - agreement_threshold: stop if cross-agent agreement exceeds this
    """
    # ── Convergence ──
    similarity_threshold: float = 0.95
    convergence_window: int = 2
    info_gain_floor: float = 0.05
    token_entropy_ceiling: float = 1.5   # Low sequence entropy (bits) = confident; calibrate per model
    agreement_threshold: float = 0.85     # High agreement = agents concur

    # ── Relative thresholds ──
    relative_thresholds: bool = True
    # When True, info_gain_floor is interpreted as: "gain < mean(recent) - k*std(recent)"
    relative_k: float = 1.0  # Number of std devs below mean to trigger "floor"
    relative_window: int = 4  # Window for computing recent mean/std

    # ── Rewind ──
    degradation_threshold: float = 0.85

    # ── Scale ──
    scale_info_gain_threshold: float = 0.3
    scale_similarity_ceiling: float = 0.7

    # ── Safety ──
    min_turns: int = 2
    max_turns: int = 15

    # ── Ablation ──
    signal_mask: set[str] | None = None  # None = use all available signals


class PACEPolicy:
    """
    Threshold-based stopping/rewind policy.

    Signal-mask-aware: only uses signals in config.signal_mask for decisions.
    This is how ablation works — each condition tests a different mask.
    """

    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    def _active(self, signal_name: str) -> bool:
        if self.config.signal_mask is None:
            return True
        return signal_name in self.config.signal_mask

    def decide(self, trajectory: Trajectory) -> PolicyResult:
        c = self.config
        turns = trajectory.turns
        t = len(turns)

        if t < c.min_turns:
            return PolicyResult(
                decision=Decision.CONTINUE,
                reason=f"Below minimum turns ({t} < {c.min_turns})",
            )

        if t >= c.max_turns:
            rewind = self._check_rewind(trajectory)
            if rewind is not None:
                return rewind
            return PolicyResult(decision=Decision.STOP, reason=f"Max turns ({c.max_turns})")

        current = turns[-1]
        snap = {s: current.get_signal(s) for s in SIGNAL_NAMES}
        snap["turn"] = t

        # === CHECK 1: Multi-signal convergence ===
        if self._check_convergence(trajectory):
            return PolicyResult(
                decision=Decision.STOP,
                reason=self._convergence_reason(trajectory),
                confidence=min((current.answer_similarity or 0.0), 1.0),
                signals_snapshot=snap,
            )

        # === CHECK 2: Rewind ===
        rewind = self._check_rewind(trajectory)
        if rewind is not None:
            return rewind

        # === CHECK 3: Scale ===
        if self._check_should_scale(trajectory):
            return PolicyResult(
                decision=Decision.SCALE,
                reason="Productive progress — extending budget",
                confidence=0.5,
                signals_snapshot=snap,
            )

        return PolicyResult(
            decision=Decision.CONTINUE,
            reason="Signals suggest continued progress",
            signals_snapshot=snap,
        )

    def _check_convergence(self, trajectory: Trajectory) -> bool:
        """
        Multi-signal convergence check.

        Stop when ALL active convergence signals agree the agent has plateaued.
        Each signal has its own convergence criterion. The policy requires
        unanimity across active signals over the convergence window.
        """
        c = self.config
        turns = trajectory.turns
        window = turns[-c.convergence_window:]
        if len(window) < c.convergence_window:
            return False

        votes = []

        # S1: Answer similarity high
        if self._active("answer_similarity"):
            votes.append(all(
                (t.answer_similarity or 0.0) >= c.similarity_threshold
                for t in window
            ))

        # S2: Info gain low (relative or absolute)
        if self._active("info_gain"):
            if c.relative_thresholds:
                votes.append(self._relative_floor_check(trajectory, "info_gain"))
            else:
                votes.append(all(
                    (t.info_gain or 1.0) <= c.info_gain_floor
                    for t in window
                ))

        # S3: Token entropy low (model is confident)
        if self._active("token_entropy"):
            votes.append(all(
                (t.token_entropy or 1.0) <= c.token_entropy_ceiling
                for t in window
            ))

        # S4: Tool entropy — not directly a convergence signal, skip here.

        # S5: Agent agreement high
        if self._active("agent_agreement"):
            votes.append(all(
                (t.agent_agreement or 0.0) >= c.agreement_threshold
                for t in window
            ))

        # Need at least one vote and all must agree
        return len(votes) > 0 and all(votes)

    def _relative_floor_check(self, trajectory: Trajectory, signal_name: str) -> bool:
        """
        Relative threshold: signal is "at floor" if recent values are
        below mean(history) - k * std(history).

        This makes the threshold adaptive to the signal's natural range
        for this particular episode, avoiding arbitrary absolute epsilons.
        """
        c = self.config
        series = trajectory.signal_series(signal_name)
        if len(series) < c.relative_window + c.convergence_window:
            return False

        # History window (before the convergence window)
        history = series[-(c.relative_window + c.convergence_window):-c.convergence_window]
        recent = series[-c.convergence_window:]

        if not history:
            return False

        mean_h = sum(history) / len(history)
        var_h = sum((x - mean_h) ** 2 for x in history) / len(history)
        std_h = math.sqrt(var_h) if var_h > 0 else 0.01  # Avoid division by zero

        floor = mean_h - c.relative_k * std_h
        return all(v <= max(floor, c.info_gain_floor) for v in recent)

    def _convergence_reason(self, trajectory: Trajectory) -> str:
        """Build a human-readable reason string listing which signals converged."""
        parts = []
        current = trajectory.turns[-1]
        if self._active("answer_similarity") and current.answer_similarity is not None:
            parts.append(f"similarity={current.answer_similarity:.3f}")
        if self._active("info_gain") and current.info_gain is not None:
            parts.append(f"info_gain={current.info_gain:.3f}")
        if self._active("token_entropy") and current.token_entropy is not None:
            parts.append(f"token_entropy={current.token_entropy:.3f}")
        if self._active("agent_agreement") and current.agent_agreement is not None:
            parts.append(f"agreement={current.agent_agreement:.3f}")
        return f"Converged: {', '.join(parts)}"

    def _check_rewind(self, trajectory: Trajectory) -> PolicyResult | None:
        c = self.config
        turns = trajectory.turns
        if len(turns) < 3:
            return None

        # Only check rewind if answer_similarity is active
        if not self._active("answer_similarity"):
            return None

        peak_idx = max(
            range(1, len(turns)),
            key=lambda i: turns[i].answer_similarity or 0.0,
        )
        if peak_idx == len(turns) - 1:
            return None

        peak_sim = turns[peak_idx].answer_similarity or 0.0
        latest_sim = turns[-1].answer_similarity or 0.0

        if latest_sim < c.degradation_threshold and peak_sim > c.similarity_threshold:
            # `peak_idx` is a list index (0-based). Convert to the user-facing
            # turn number from the stored TurnState.
            peak_turn = turns[peak_idx].turn_number
            return PolicyResult(
                decision=Decision.REWIND,
                reason=(
                    f"Degradation: peak at turn {peak_turn} "
                    f"(sim={peak_sim:.3f}), latest={latest_sim:.3f}"
                ),
                rewind_to_turn=peak_turn,
                confidence=peak_sim - latest_sim,
            )
        return None

    def _check_should_scale(self, trajectory: Trajectory) -> bool:
        c = self.config
        current = trajectory.turns[-1]
        info_ok = (current.info_gain or 0.0) >= c.scale_info_gain_threshold if self._active("info_gain") else False
        sim_ok = (current.answer_similarity or 1.0) <= c.scale_similarity_ceiling if self._active("answer_similarity") else False

        # Scale if agent is still making progress (high info gain, answer still changing)
        # OR if agents disagree and confidence is high (confident conflict)
        confident_conflict = (
            self._active("agent_agreement")
            and (current.agent_agreement or 1.0) < 0.5
            and (current.token_entropy or 1.0) < c.token_entropy_ceiling
        )
        return (info_ok and sim_ok) or confident_conflict
