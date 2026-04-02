"""
τ²-Bench Experiment Runner.

Mode A (post-hoc analysis):
  1. Run tau2 normally at full budget
  2. Replay agent turns through PACE
  3. Measure where PACE would have stopped / rewound
  4. Compare: did stopping early hurt or help accuracy?

Usage:
    # Step 1: Run simulations (requires tau2-bench installed)
    python experiments/run_tau2.py --step run --domain airline --num-tasks 20

    # Step 2: Analyze with PACE
    python experiments/run_tau2.py --step analyze --sim-dir data/simulations/

    # Or both:
    python experiments/run_tau2.py --step all --domain airline --num-tasks 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from pace import SignalComputer
from pace.policy import PACEPolicy, PolicyConfig
from benchmarks.tau2 import Tau2Benchmark, Tau2Result


def run_and_analyze(
    domain: str = "airline",
    agent_llm: str = "gpt-4.1",
    user_llm: str = "gpt-4.1",
    num_tasks: int = 20,
    sim_dir: str | None = None,
    output_dir: str = "results/tau2",
) -> None:
    bench = Tau2Benchmark(
        domain=domain,
        agent_llm=agent_llm,
        user_llm=user_llm,
        output_dir=output_dir,
    )

    # Run simulations if no pre-existing results
    if sim_dir:
        sim_path = Path(sim_dir)
    else:
        print(f"Running τ²-Bench: domain={domain}, agent={agent_llm}, tasks={num_tasks}")
        sim_path = bench.run_simulations(num_tasks=num_tasks)

    # Load results
    simulations = bench.load_simulations(sim_path)
    print(f"Loaded {len(simulations)} simulation(s)")

    # Analyze with PACE
    policy = PACEPolicy(PolicyConfig(
        similarity_threshold=0.95,
        convergence_window=2,
        info_gain_floor=0.05,
        degradation_threshold=0.85,
        min_turns=2,
        max_turns=15,
    ))
    sc = SignalComputer()

    results = bench.analyze_with_pace(simulations, policy=policy, signal_computer=sc)

    # Print analysis
    print(f"\n{'='*60}")
    print(f"τ²-Bench PACE Analysis: {domain}")
    print(f"{'='*60}")

    total = len(results)
    successes = sum(1 for r in results if r.success)
    pace_would_stop_early = sum(1 for r in results if r.pace_would_stop_at is not None)
    pace_would_rewind = sum(1 for r in results if r.pace_would_rewind_to is not None)

    avg_agent_turns = sum(r.num_agent_turns for r in results) / max(total, 1)
    avg_pace_turns = sum(
        r.pace_would_stop_at or r.num_agent_turns for r in results
    ) / max(total, 1)
    turn_savings = (1 - avg_pace_turns / avg_agent_turns) * 100 if avg_agent_turns > 0 else 0

    print(f"  Total tasks: {total}")
    print(f"  Baseline success rate: {successes}/{total} ({successes/max(total,1)*100:.1f}%)")
    print(f"  Avg agent turns (baseline): {avg_agent_turns:.1f}")
    print(f"  Avg turns if PACE stopped: {avg_pace_turns:.1f}")
    print(f"  Turn savings: {turn_savings:.1f}%")
    print(f"  PACE would stop early: {pace_would_stop_early}/{total}")
    print(f"  PACE would rewind: {pace_would_rewind}/{total}")
    print()

    # Per-task breakdown
    print(f"{'Task':<15} {'Success':>8} {'Turns':>6} {'PACE Stop':>10} {'PACE Rewind':>12}")
    print("-" * 55)
    for r in results[:30]:  # Cap display
        print(
            f"{r.task_id:<15} "
            f"{'✓' if r.success else '✗':>8} "
            f"{r.num_agent_turns:>6} "
            f"{r.pace_would_stop_at or '-':>10} "
            f"{r.pace_would_rewind_to or '-':>12}"
        )

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"tau2_{domain}_{agent_llm}_pace_analysis.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "domain": domain,
                "agent_llm": agent_llm,
                "total_tasks": total,
                "baseline_success_rate": successes / max(total, 1),
                "avg_turns_baseline": avg_agent_turns,
                "avg_turns_pace": avg_pace_turns,
                "turn_savings_pct": turn_savings,
                "pace_early_stops": pace_would_stop_early,
                "pace_rewinds": pace_would_rewind,
                "per_task": [
                    {
                        "task_id": r.task_id,
                        "success": r.success,
                        "num_turns": r.num_agent_turns,
                        "pace_stop_at": r.pace_would_stop_at,
                        "pace_rewind_to": r.pace_would_rewind_to,
                        "decisions": r.pace_decisions,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="τ²-Bench PACE Experiment")
    parser.add_argument("--step", choices=["run", "analyze", "all"], default="all")
    parser.add_argument("--domain", default="airline")
    parser.add_argument("--agent-llm", default="gpt-4.1")
    parser.add_argument("--user-llm", default="gpt-4.1")
    parser.add_argument("--num-tasks", type=int, default=20)
    parser.add_argument("--sim-dir", default=None, help="Path to pre-run simulations")
    parser.add_argument("--output-dir", default="results/tau2")
    args = parser.parse_args()

    run_and_analyze(
        domain=args.domain,
        agent_llm=args.agent_llm,
        user_llm=args.user_llm,
        num_tasks=args.num_tasks,
        sim_dir=args.sim_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
