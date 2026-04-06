"""
InterventionPolicy: decides when and how to intervene during multi-turn conversations.

Intervention types:
  NONE    — continue normally
  RECAP   — inject a recap message summarising revealed shards
  CONTEXT_EVICTION — evict conversation history, re-prompt with all shards in one shot

The policy checks each active signal against its threshold.
If any signal triggers, the corresponding intervention fires.
Signal activation is controlled by ``enabled_signals`` on InterventionConfig,
allowing the ablation study to test arbitrary subsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from pace.trajectory import Trajectory, SIGNAL_NAMES


class InterventionType(Enum):
    NONE = auto()
    RECAP = auto()
    CONTEXT_EVICTION = auto()


@dataclass
class InterventionConfig:
    """
    Threshold configuration for PACE interventions.

    Each signal has a threshold. When the signal crosses its threshold,
    the policy recommends an intervention. The ``enabled_signals`` set
    controls which signals are active (None = all).
    """
    # ── Signal thresholds ──
    drift_threshold: float = 0.45
    coverage_threshold: float = 0.35
    contradiction_threshold: float = 0.65
    stability_threshold: float = 0.55
    entropy_threshold: float = 1.5
    repetition_threshold: float = 0.55

    # ── Intervention behaviour ──
    intervention_type: InterventionType = InterventionType.RECAP
    min_turns_before_intervention: int = 2
    max_interventions: int = 3
    cooldown_turns: int = 2

    # ── Ablation control ──
    enabled_signals: set[str] | None = None  # None = all 6 signals active

    def is_enabled(self, signal_name: str) -> bool:
        if self.enabled_signals is None:
            return True
        return signal_name in self.enabled_signals


@dataclass
class InterventionResult:
    intervention: InterventionType
    reason: str
    triggered_signals: list[str] = field(default_factory=list)
    signals_snapshot: dict[str, Any] = field(default_factory=dict)


class InterventionPolicy:
    """
    Threshold-based intervention policy.

    Checks each enabled signal against its threshold for the latest turn.
    If any signal fires, returns the configured intervention type.
    """

    def __init__(self, config: InterventionConfig | None = None):
        self.config = config or InterventionConfig()
        self._intervention_count = 0
        self._last_intervention_turn = -100

    def reset(self) -> None:
        self._intervention_count = 0
        self._last_intervention_turn = -100

    def evaluate(self, trajectory: Trajectory) -> InterventionResult:
        """Evaluate the latest turn and decide whether to intervene."""
        c = self.config
        turns = trajectory.turns
        t = len(turns)

        if t < c.min_turns_before_intervention:
            return InterventionResult(
                intervention=InterventionType.NONE,
                reason=f"Below minimum turns ({t} < {c.min_turns_before_intervention})",
            )

        if self._intervention_count >= c.max_interventions:
            return InterventionResult(
                intervention=InterventionType.NONE,
                reason=f"Max interventions reached ({c.max_interventions})",
            )

        if (t - self._last_intervention_turn) < c.cooldown_turns:
            return InterventionResult(
                intervention=InterventionType.NONE,
                reason=f"Cooldown ({c.cooldown_turns} turns)",
            )

        current = turns[-1]
        triggered: list[str] = []
        snap = current.signals_dict()
        snap["turn"] = t

        # goal_drift: LOW value means drifted far from original goal
        if c.is_enabled("goal_drift") and current.goal_drift is not None:
            if current.goal_drift < c.drift_threshold:
                triggered.append(f"goal_drift={current.goal_drift:.3f} < {c.drift_threshold}")

        # shard_coverage: LOW value means a shard was forgotten
        if c.is_enabled("shard_coverage") and current.shard_coverage is not None:
            if current.shard_coverage < c.coverage_threshold:
                triggered.append(f"shard_coverage={current.shard_coverage:.3f} < {c.coverage_threshold}")

        # contradiction: HIGH value means the model contradicted itself
        if c.is_enabled("contradiction") and current.contradiction is not None:
            if current.contradiction > c.contradiction_threshold:
                triggered.append(f"contradiction={current.contradiction:.3f} > {c.contradiction_threshold}")

        # response_stability: LOW value means abrupt reversal
        if c.is_enabled("response_stability") and current.response_stability is not None:
            if current.response_stability < c.stability_threshold:
                triggered.append(f"response_stability={current.response_stability:.3f} < {c.stability_threshold}")

        # token_entropy: HIGH value means model is uncertain
        if c.is_enabled("token_entropy") and current.token_entropy is not None:
            if current.token_entropy > c.entropy_threshold:
                triggered.append(f"token_entropy={current.token_entropy:.3f} > {c.entropy_threshold}")

        # repetition: HIGH value means stuck in a loop
        if c.is_enabled("repetition") and current.repetition is not None:
            if current.repetition > c.repetition_threshold:
                triggered.append(f"repetition={current.repetition:.3f} > {c.repetition_threshold}")

        if triggered:
            self._intervention_count += 1
            self._last_intervention_turn = t
            return InterventionResult(
                intervention=c.intervention_type,
                reason="; ".join(triggered),
                triggered_signals=triggered,
                signals_snapshot=snap,
            )

        return InterventionResult(
            intervention=InterventionType.NONE,
            reason="All signals within thresholds",
            signals_snapshot=snap,
        )
