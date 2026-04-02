"""
τ²-Bench Wrapper.

τ²-Bench is a dual-control benchmark where both agent and user have tools.
The official implementation runs via CLI. This wrapper:
  1. Intercepts the agent's turn loop to inject PACE monitoring
  2. Records trajectory signals at each agent turn
  3. Compares PACE-stopped results vs. fixed-budget results

Setup: pip install tau2-bench (from sierra-research/tau2-bench)
       Then: tau2 run --domain airline --agent-llm gpt-4.1

For PACE integration, we wrap the agent's response function to
capture signals between turns.

NOTE: τ²-Bench has its own agent loop. PACE integrates by wrapping
the agent LLM call, not by replacing the loop. This means PACE
observes the agent's outputs and can signal "this agent should stop
deliberating" but the actual stopping mechanism depends on whether
we modify the tau2 harness or do post-hoc analysis.

Two modes:
  A) POST-HOC: Run tau2 normally, record all turns, analyze retroactively
     what PACE would have decided. This is simpler and sufficient for the paper.
  B) LIVE: Modify the tau2 agent to check PACE at each turn and potentially
     short-circuit. This requires forking tau2-bench.

We implement Mode A (post-hoc) for the paper.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pace.trajectory import Trajectory, TurnState
from pace.signals import SignalComputer
from pace.policy import PACEPolicy, Decision


@dataclass
class Tau2Result:
    """Result of one τ²-Bench task."""
    task_id: str
    domain: str
    success: bool
    num_agent_turns: int
    total_tokens: int
    trajectory: Trajectory | None = None
    pace_decisions: list[dict] = field(default_factory=list)
    pace_would_stop_at: int | None = None
    pace_would_rewind_to: int | None = None
    raw_simulation: dict = field(default_factory=dict)


class Tau2Benchmark:
    """
    Run τ²-Bench and analyze results with PACE.

    Usage:
        bench = Tau2Benchmark(domain="airline", agent_llm="gpt-4.1")
        bench.run_simulations(num_tasks=20)
        results = bench.analyze_with_pace(policy=PACEPolicy())
    """

    def __init__(
        self,
        domain: str = "airline",
        agent_llm: str = "gpt-4.1",
        user_llm: str = "gpt-4.1",
        output_dir: str = "results/tau2",
    ):
        self.domain = domain
        self.agent_llm = agent_llm
        self.user_llm = user_llm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_simulations(self, num_tasks: int = 20, num_trials: int = 1) -> Path:
        """
        Run τ²-Bench via CLI. Returns path to simulation results.

        Requires tau2-bench to be installed.
        """
        cmd = [
            "tau2", "run",
            "--domain", self.domain,
            "--agent-llm", self.agent_llm,
            "--user-llm", self.user_llm,
            "--num-trials", str(num_trials),
            "--num-tasks", str(num_tasks),
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        # tau2 saves to data/simulations/ by default
        return Path("data/simulations/")

    def load_simulations(self, sim_dir: Path | None = None) -> list[dict]:
        """Load simulation results from tau2's output directory."""
        sim_dir = sim_dir or Path("data/simulations/")
        results = []
        for f in sorted(sim_dir.glob("*.json")):
            with open(f) as fh:
                results.append(json.load(fh))
        return results

    def extract_agent_turns(self, simulation: dict) -> list[dict]:
        """
        Extract agent turns from a τ²-Bench simulation.

        Each turn in the simulation has messages between user and agent.
        We extract the agent's responses as individual turns.
        """
        agent_turns = []
        messages = simulation.get("messages", simulation.get("conversation", []))

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            if role == "assistant":
                # This is an agent turn
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])
                tool_name = ""
                tool_output = ""

                if tool_calls:
                    tool_name = tool_calls[0].get("function", {}).get("name", "")

                # Look for tool result in next message
                if i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                    tool_output = messages[i + 1].get("content", "")

                agent_turns.append({
                    "content": content,
                    "tool_name": tool_name,
                    "tool_output": tool_output,
                    "message_index": i,
                })

        return agent_turns

    def analyze_with_pace(
        self,
        simulations: list[dict],
        policy: PACEPolicy | None = None,
        signal_computer: SignalComputer | None = None,
    ) -> list[Tau2Result]:
        """
        Post-hoc analysis: for each completed simulation, replay the agent
        turns through PACE and determine where it would have stopped.
        """
        policy = policy or PACEPolicy()
        sc = signal_computer or SignalComputer()
        results = []

        for sim in simulations:
            task_id = sim.get("task_id", sim.get("id", "unknown"))
            domain = sim.get("domain", self.domain)
            success = sim.get("success", sim.get("score", 0) > 0)

            agent_turns = self.extract_agent_turns(sim)
            trajectory = Trajectory(
                query=sim.get("task", sim.get("instruction", "")),
                episode_id=task_id,
            )

            pace_decisions = []
            pace_stop_turn = None
            pace_rewind_turn = None
            cumulative_ctx = ""

            for i, at in enumerate(agent_turns):
                # Build TurnState from tau2 data
                if at["tool_output"]:
                    cumulative_ctx += f"\n---\n{at['tool_output']}"

                state = TurnState(
                    turn_number=i + 1,
                    answer=at["content"],
                    retrieved_context=at["tool_output"],
                    cumulative_context=cumulative_ctx,
                    tool_called=at["tool_name"],
                    reasoning_text=at["content"],
                )
                trajectory.add_turn(state)
                sc.compute(trajectory)

                # What would PACE decide?
                decision = policy.decide(trajectory)
                pace_decisions.append({
                    "turn": i + 1,
                    "decision": decision.decision.name,
                    "reason": decision.reason,
                    "rewind_to": decision.rewind_to_turn,
                })

                if pace_stop_turn is None and decision.decision == Decision.STOP:
                    pace_stop_turn = i + 1
                if pace_rewind_turn is None and decision.decision == Decision.REWIND:
                    pace_rewind_turn = decision.rewind_to_turn

            results.append(Tau2Result(
                task_id=task_id,
                domain=domain,
                success=success,
                num_agent_turns=len(agent_turns),
                total_tokens=0,  # tau2 doesn't easily expose this
                trajectory=trajectory,
                pace_decisions=pace_decisions,
                pace_would_stop_at=pace_stop_turn,
                pace_would_rewind_to=pace_rewind_turn,
                raw_simulation=sim,
            ))

        return results
