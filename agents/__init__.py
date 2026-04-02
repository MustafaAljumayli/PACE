"""
Base agent abstraction.

Both single-agent and multi-agent implementations follow the same turn protocol:
  1. Agent receives query + current trajectory
  2. Agent produces a TurnState (answer, context, tool used, reasoning)
  3. Signals are computed on the TurnState
  4. Policy decides: continue / stop / rewind / scale

This protocol is what makes PACE agent-agnostic.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from pace.trajectory import Trajectory, TurnState
from pace.signals import SignalComputer
from pace.policy import PACEPolicy, PolicyResult, Decision


class BaseAgent(ABC):
    """
    Abstract base for any agent that participates in a PACE-monitored loop.

    Subclasses implement `_execute_turn()` which does the actual LLM call + tool use.
    The `run()` method handles the outer loop with PACE monitoring.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_turns: int = 10,
        tools: dict[str, Any] | None = None,
    ):
        self.model = model
        self.max_turns = max_turns
        self.tools = tools or {}

    @abstractmethod
    def _execute_turn(
        self,
        query: str,
        trajectory: Trajectory,
        turn_number: int,
    ) -> TurnState:
        """
        Execute one turn of the agent loop.

        Must return a TurnState with at minimum:
          - answer: the agent's current best answer
          - tool_called: which tool was used (empty string if none)
          - retrieved_context: new information obtained this turn
          - reasoning_text: the agent's reasoning trace
          - token_count: approximate tokens used

        The cumulative_context field will be set by the run loop.
        """
        ...

    def run(
        self,
        query: str,
        pace_policy: PACEPolicy | None = None,
        signal_computer: SignalComputer | None = None,
        episode_id: str = "",
    ) -> tuple[str, Trajectory, list[PolicyResult]]:
        """
        Run the full multi-turn loop with PACE monitoring.

        Returns:
            (final_answer, trajectory, policy_decisions)

        If pace_policy is None, runs to max_turns (fixed budget baseline).
        """
        sc = signal_computer or SignalComputer()
        trajectory = Trajectory(query=query, episode_id=episode_id)
        decisions: list[PolicyResult] = []
        cumulative_ctx = ""

        for turn_num in range(1, self.max_turns + 1):
            # Execute the turn
            start = time.time()
            state = self._execute_turn(query, trajectory, turn_num)
            state.latency_ms = (time.time() - start) * 1000

            # Track cumulative context
            if state.retrieved_context:
                cumulative_ctx += "\n---\n" + state.retrieved_context
            state.cumulative_context = cumulative_ctx

            # Record turn
            trajectory.add_turn(state)

            # Compute signals
            sc.compute(trajectory)

            # Policy decision
            if pace_policy is not None:
                result = pace_policy.decide(trajectory)
                decisions.append(result)

                if result.decision == Decision.STOP:
                    return trajectory.latest_answer, trajectory, decisions

                elif result.decision == Decision.REWIND:
                    rewind_turn_number = result.rewind_to_turn or 0
                    # `rewind_to_turn` is the user-facing `TurnState.turn_number`
                    # (1-based), not a list index.
                    rewind_idx = next(
                        (i for i, t in enumerate(trajectory.turns) if t.turn_number == rewind_turn_number),
                        0,
                    )
                    rewind_answer = trajectory.turns[rewind_idx].answer
                    return rewind_answer, trajectory, decisions

                elif result.decision == Decision.SCALE:
                    # Increase budget by 50%, capped at 20
                    self.max_turns = min(int(self.max_turns * 1.5), 20)

                # CONTINUE: keep going

        # Exhausted turns — return latest answer
        return trajectory.latest_answer, trajectory, decisions


class FixedBudgetRunner:
    """
    Runs an agent to a fixed turn count with NO early stopping.
    Records all turns so we can retrospectively analyze what PACE would have done.
    """

    def __init__(self, agent: BaseAgent, num_turns: int = 10):
        self.agent = agent
        self.num_turns = num_turns

    def run(
        self,
        query: str,
        signal_computer: SignalComputer | None = None,
        episode_id: str = "",
    ) -> tuple[str, Trajectory]:
        """Run to exactly num_turns. Return final answer + full trajectory."""
        sc = signal_computer or SignalComputer()
        trajectory = Trajectory(query=query, episode_id=episode_id)
        cumulative_ctx = ""

        for turn_num in range(1, self.num_turns + 1):
            state = self.agent._execute_turn(query, trajectory, turn_num)
            if state.retrieved_context:
                cumulative_ctx += "\n---\n" + state.retrieved_context
            state.cumulative_context = cumulative_ctx
            trajectory.add_turn(state)
            sc.compute(trajectory)

        return trajectory.latest_answer, trajectory
