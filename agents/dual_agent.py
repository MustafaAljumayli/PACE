"""
DualAgentRunner: runs two agents (e.g., GPT-4o + Claude Sonnet) on the same
query in parallel and computes S5 (agent_agreement) at each turn.

This is the "social cue" contribution — the hypothesis is that when two
independent agents converge on the same answer, the system can stop with
higher confidence than when monitoring a single agent's self-consistency.

Architecture:
  - Agent A and Agent B each run their own turn loop independently
  - After each turn t, we extract both agents' current answers
  - NLIScorer computes semantic similarity between the two answers
  - The agreement score is injected into BOTH trajectories as S5
  - PACE policy sees all 5 signals and can use S5 for convergence

The two agents can be:
  - Different models (GPT-4o + Claude Sonnet) — tests cross-model agreement
  - Same model, different temperatures — controls for model differences
  - Single + Multi agent — tests architecture agreement

For the paper: GPT-4o + Claude Sonnet is the primary configuration.
"""

from __future__ import annotations

import time
from typing import Any

from pace.embeddings import NLIScorer
from pace.signals import SignalComputer
from pace.trajectory import Trajectory, TurnState
from pace.policy import PACEPolicy, PolicyResult, Decision
from agents import BaseAgent


class DualAgentRunner:
    """
    Run two agents on the same query, computing inter-agent agreement.

    The runner alternates: Agent A does turn t, Agent B does turn t,
    then we compute agreement and check the PACE policy.

    Args:
        agent_a: First agent (e.g., GPT-4o ReAct)
        agent_b: Second agent (e.g., Claude Sonnet ReAct)
        nli_scorer: NLI model for semantic agreement scoring
        signal_computer: Computes S1-S4 for each agent's trajectory
        pace_policy: Makes stop/continue decisions using all 5 signals
        primary: Which agent's answer to return ("a", "b", or "consensus")
    """

    def __init__(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        nli_scorer: NLIScorer | None = None,
        signal_computer: SignalComputer | None = None,
        pace_policy: PACEPolicy | None = None,
        primary: str = "a",
        max_turns: int = 10,
    ):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.nli = nli_scorer or NLIScorer()
        self.sc = signal_computer or SignalComputer()
        self.policy = pace_policy
        self.primary = primary
        self.max_turns = max_turns

    def run(
        self,
        query: str,
        episode_id: str = "",
    ) -> DualAgentResult:
        """
        Run both agents and return the result.

        Returns a DualAgentResult containing both trajectories,
        the agreement series, and the final answer.
        """
        traj_a = Trajectory(query=query, episode_id=f"{episode_id}_A")
        traj_b = Trajectory(query=query, episode_id=f"{episode_id}_B")
        decisions: list[PolicyResult] = []
        agreement_series: list[float] = []
        cumulative_a = ""
        cumulative_b = ""

        for turn_num in range(1, self.max_turns + 1):
            # ── Agent A executes turn ──
            start_a = time.time()
            state_a = self.agent_a._execute_turn(query, traj_a, turn_num)
            state_a.latency_ms = (time.time() - start_a) * 1000
            if state_a.retrieved_context:
                cumulative_a += "\n---\n" + state_a.retrieved_context
            state_a.cumulative_context = cumulative_a
            traj_a.add_turn(state_a)
            self.sc.compute(traj_a)

            # ── Agent B executes turn ──
            start_b = time.time()
            state_b = self.agent_b._execute_turn(query, traj_b, turn_num)
            state_b.latency_ms = (time.time() - start_b) * 1000
            if state_b.retrieved_context:
                cumulative_b += "\n---\n" + state_b.retrieved_context
            state_b.cumulative_context = cumulative_b
            traj_b.add_turn(state_b)
            self.sc.compute(traj_b)

            # ── S5: Inter-agent agreement ──
            agreement = self.nli.similarity(state_a.answer, state_b.answer)
            agreement_series.append(agreement)

            # Inject S5 into both trajectories
            state_a.agent_agreement = agreement
            state_b.agent_agreement = agreement

            # Compute S5 derivatives (velocity/acceleration)
            self._compute_agreement_derivatives(traj_a, turn_num - 1)
            self._compute_agreement_derivatives(traj_b, turn_num - 1)

            # ── PACE decision (uses primary agent's trajectory) ──
            primary_traj = traj_a if self.primary == "a" else traj_b
            if self.policy is not None:
                result = self.policy.decide(primary_traj)
                decisions.append(result)

                if result.decision == Decision.STOP:
                    return self._build_result(
                        traj_a, traj_b, agreement_series, decisions,
                        final_turn=turn_num, rewind_turn=None,
                    )
                elif result.decision == Decision.REWIND:
                    return self._build_result(
                        traj_a, traj_b, agreement_series, decisions,
                        final_turn=turn_num, rewind_turn=result.rewind_to_turn,
                    )
                elif result.decision == Decision.SCALE:
                    self.max_turns = min(int(self.max_turns * 1.5), 20)

        # Exhausted turns
        return self._build_result(
            traj_a, traj_b, agreement_series, decisions,
            final_turn=self.max_turns, rewind_turn=None,
        )

    def _compute_agreement_derivatives(self, traj: Trajectory, t: int) -> None:
        """Compute velocity and acceleration for the agreement signal."""
        current = traj.turns[t]
        if t >= 1:
            prev = traj.turns[t - 1]
            prev_ag = prev.agent_agreement
            cur_ag = current.agent_agreement
            if prev_ag is not None and cur_ag is not None:
                current.agent_agreement_v = cur_ag - prev_ag
            else:
                current.agent_agreement_v = 0.0
        else:
            current.agent_agreement_v = 0.0

        if t >= 2:
            prev = traj.turns[t - 1]
            if current.agent_agreement_v is not None and prev.agent_agreement_v is not None:
                current.agent_agreement_a = current.agent_agreement_v - prev.agent_agreement_v
            else:
                current.agent_agreement_a = 0.0
        else:
            current.agent_agreement_a = 0.0

    def _build_result(
        self,
        traj_a: Trajectory,
        traj_b: Trajectory,
        agreement_series: list[float],
        decisions: list[PolicyResult],
        final_turn: int,
        rewind_turn: int | None,
    ) -> "DualAgentResult":
        # Select final answer
        if rewind_turn is not None:
            if self.primary == "a":
                answer = traj_a.turns[rewind_turn].answer
            else:
                answer = traj_b.turns[rewind_turn].answer
        else:
            if self.primary == "consensus":
                # If agents agree, use A's answer; if they disagree, use the
                # one with higher within-agent convergence
                if agreement_series and agreement_series[-1] > 0.8:
                    answer = traj_a.latest_answer
                else:
                    sim_a = traj_a.turns[-1].answer_similarity or 0.0
                    sim_b = traj_b.turns[-1].answer_similarity or 0.0
                    answer = traj_a.latest_answer if sim_a >= sim_b else traj_b.latest_answer
            elif self.primary == "a":
                answer = traj_a.latest_answer
            else:
                answer = traj_b.latest_answer

        return DualAgentResult(
            answer=answer,
            trajectory_a=traj_a,
            trajectory_b=traj_b,
            agreement_series=agreement_series,
            decisions=decisions,
            final_turn=final_turn,
            rewind_turn=rewind_turn,
        )


class DualAgentResult:
    """Result from a dual-agent run."""

    def __init__(
        self,
        answer: str,
        trajectory_a: Trajectory,
        trajectory_b: Trajectory,
        agreement_series: list[float],
        decisions: list[PolicyResult],
        final_turn: int,
        rewind_turn: int | None = None,
    ):
        self.answer = answer
        self.trajectory_a = trajectory_a
        self.trajectory_b = trajectory_b
        self.agreement_series = agreement_series
        self.decisions = decisions
        self.final_turn = final_turn
        self.rewind_turn = rewind_turn

    @property
    def total_tokens(self) -> int:
        return self.trajectory_a.total_tokens() + self.trajectory_b.total_tokens()

    @property
    def total_latency_ms(self) -> float:
        # Agents run sequentially, so sum
        return self.trajectory_a.total_latency_ms() + self.trajectory_b.total_latency_ms()

    @property
    def rewind_used(self) -> bool:
        return self.rewind_turn is not None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer[:200],
            "final_turn": self.final_turn,
            "rewind_turn": self.rewind_turn,
            "agreement_series": self.agreement_series,
            "total_tokens": self.total_tokens,
            "trajectory_a": self.trajectory_a.to_dict(),
            "trajectory_b": self.trajectory_b.to_dict(),
            "decisions": [
                {"decision": d.decision.name, "reason": d.reason}
                for d in self.decisions
            ],
        }
