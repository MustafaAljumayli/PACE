"""
FRAMES Experiment Runner.

Runs the full experiment matrix on FRAMES:
  1. Single Agent + Fixed Budget (5 turns, 10 turns)
  2. Single Agent + PACE
  3. Multi Agent + Fixed Budget (5 turns, 10 turns)
  4. Multi Agent + PACE

Outputs results to results/frames/ as JSON for analysis.

Usage:
    python experiments/run_frames.py --mode all --num-questions 50
    python experiments/run_frames.py --mode single-pace --num-questions 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pace import SignalComputer, PACEPolicy, Trajectory
from pace.policy import PolicyConfig, Decision
from agents import FixedBudgetRunner
from agents.single_react import SingleReActAgent
from agents.multi_agentflow import MultiAgentTeam
from benchmarks.frames import FRAMESBenchmark, FRAMESResult
from tools import get_frames_tools


def run_condition(
    agent_type: str,
    budget_type: str,
    benchmark: FRAMESBenchmark,
    model: str = "gpt-4o",
    fixed_turns: int = 10,
    output_dir: Path = Path("results/frames"),
) -> list[FRAMESResult]:
    """
    Run one experimental condition.

    agent_type: "single" or "multi"
    budget_type: "fixed-5", "fixed-10", "pace"
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tools = get_frames_tools()
    sc = SignalComputer()

    # Setup agent
    if agent_type == "single":
        agent = SingleReActAgent(model=model, max_turns=15, tools=tools)
    else:
        agent = MultiAgentTeam(model=model, max_turns=15, tools=tools)

    # Setup policy
    if budget_type.startswith("fixed"):
        turns = int(budget_type.split("-")[1])
        pace_policy = None
        runner = FixedBudgetRunner(agent, num_turns=turns)
    else:
        pace_policy = PACEPolicy(PolicyConfig(
            similarity_threshold=0.95,
            convergence_window=2,
            info_gain_floor=0.05,
            degradation_threshold=0.85,
            min_turns=2,
            max_turns=15,
        ))

    questions = benchmark.load()
    results: list[FRAMESResult] = []
    condition_name = f"{agent_type}_{budget_type}"

    print(f"\n{'='*60}")
    print(f"CONDITION: {condition_name} | {len(questions)} questions | model={model}")
    print(f"{'='*60}\n")

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q.question[:80]}...")
        start = time.time()

        try:
            if pace_policy:
                answer, trajectory, decisions = agent.run(
                    query=q.question,
                    pace_policy=pace_policy,
                    signal_computer=sc,
                    episode_id=q.question_id,
                )
                decision_dicts = [
                    {
                        "turn": d.signals_snapshot.get("turn", j + 1),
                        "decision": d.decision.name,
                        "reason": d.reason,
                        "rewind_to": d.rewind_to_turn,
                    }
                    for j, d in enumerate(decisions)
                ]
                rewind_used = any(d.decision == Decision.REWIND for d in decisions)
                rewind_turn = next(
                    (d.rewind_to_turn for d in decisions if d.decision == Decision.REWIND),
                    None,
                )
            else:
                answer, trajectory = runner.run(
                    query=q.question,
                    signal_computer=sc,
                    episode_id=q.question_id,
                )
                decision_dicts = None
                rewind_used = False
                rewind_turn = None

            evaluation = benchmark.evaluate(answer, q.answer)

            result = FRAMESResult(
                question_id=q.question_id,
                question=q.question,
                gold_answer=q.answer,
                predicted_answer=answer,
                evaluation=evaluation,
                num_turns_used=trajectory.current_turn,
                total_tokens=trajectory.total_tokens(),
                total_latency_ms=trajectory.total_latency_ms(),
                pace_decisions=decision_dicts,
                trajectory_data=trajectory.to_dict(),
                rewind_used=rewind_used,
                rewind_turn=rewind_turn,
            )
            results.append(result)

            elapsed = time.time() - start
            print(
                f"    → score={evaluation['score']:.2f} | "
                f"turns={trajectory.current_turn} | "
                f"tokens={trajectory.total_tokens()} | "
                f"time={elapsed:.1f}s"
                f"{' | REWIND→t' + str(rewind_turn) if rewind_used else ''}"
            )

        except Exception as e:
            print(f"    → ERROR: {e}")
            results.append(FRAMESResult(
                question_id=q.question_id,
                question=q.question,
                gold_answer=q.answer,
                predicted_answer=f"ERROR: {e}",
                evaluation={"score": 0.0, "exact_match": False},
                num_turns_used=0,
                total_tokens=0,
                total_latency_ms=0,
            ))

    # Save results
    out_file = output_dir / f"{condition_name}_{model}_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "condition": condition_name,
                "model": model,
                "num_questions": len(questions),
                "results": [
                    {
                        "question_id": r.question_id,
                        "question": r.question,
                        "gold": r.gold_answer,
                        "predicted": r.predicted_answer,
                        "evaluation": r.evaluation,
                        "turns": r.num_turns_used,
                        "tokens": r.total_tokens,
                        "latency_ms": r.total_latency_ms,
                        "rewind_used": r.rewind_used,
                        "rewind_turn": r.rewind_turn,
                        "pace_decisions": r.pace_decisions,
                    }
                    for r in results
                ],
                "summary": _summarize(results),
            },
            f,
            indent=2,
        )
    print(f"\n  Saved: {out_file}")
    _print_summary(condition_name, results)
    return results


def _summarize(results: list[FRAMESResult]) -> dict:
    """Compute summary statistics."""
    if not results:
        return {}
    scores = [r.evaluation.get("score", 0.0) for r in results]
    turns = [r.num_turns_used for r in results if r.num_turns_used > 0]
    tokens = [r.total_tokens for r in results if r.total_tokens > 0]
    rewinds = sum(1 for r in results if r.rewind_used)

    return {
        "accuracy_mean": sum(scores) / len(scores),
        "accuracy_exact_match": sum(1 for r in results if r.evaluation.get("exact_match")) / len(results),
        "turns_mean": sum(turns) / max(len(turns), 1),
        "turns_median": sorted(turns)[len(turns) // 2] if turns else 0,
        "tokens_mean": sum(tokens) / max(len(tokens), 1),
        "rewind_count": rewinds,
        "rewind_pct": rewinds / len(results) * 100,
        "n": len(results),
    }


def _print_summary(condition: str, results: list[FRAMESResult]) -> None:
    s = _summarize(results)
    print(f"\n  ── Summary: {condition} ──")
    print(f"  Accuracy (mean score): {s.get('accuracy_mean', 0):.3f}")
    print(f"  Accuracy (exact match): {s.get('accuracy_exact_match', 0):.3f}")
    print(f"  Avg turns: {s.get('turns_mean', 0):.1f}")
    print(f"  Avg tokens: {s.get('tokens_mean', 0):.0f}")
    print(f"  Rewinds: {s.get('rewind_count', 0)} ({s.get('rewind_pct', 0):.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run FRAMES experiments")
    parser.add_argument(
        "--mode",
        choices=["all", "single-fixed", "single-pace", "multi-fixed", "multi-pace"],
        default="all",
    )
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--output-dir", default="results/frames")
    args = parser.parse_args()

    benchmark = FRAMESBenchmark(max_questions=args.num_questions)
    output_dir = Path(args.output_dir)

    conditions = []
    if args.mode in ("all", "single-fixed"):
        conditions.append(("single", "fixed-5"))
        conditions.append(("single", "fixed-10"))
    if args.mode in ("all", "single-pace"):
        conditions.append(("single", "pace"))
    if args.mode in ("all", "multi-fixed"):
        conditions.append(("multi", "fixed-5"))
        conditions.append(("multi", "fixed-10"))
    if args.mode in ("all", "multi-pace"):
        conditions.append(("multi", "pace"))

    all_results = {}
    for agent_type, budget_type in conditions:
        results = run_condition(
            agent_type=agent_type,
            budget_type=budget_type,
            benchmark=benchmark,
            model=args.model,
            output_dir=output_dir,
        )
        all_results[f"{agent_type}_{budget_type}"] = _summarize(results)

    # Print comparison table
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)
    print(f"{'Condition':<25} {'Accuracy':>10} {'Avg Turns':>10} {'Avg Tokens':>12} {'Rewinds':>10}")
    print("-" * 70)
    for cond, s in all_results.items():
        print(
            f"{cond:<25} "
            f"{s.get('accuracy_mean', 0):>10.3f} "
            f"{s.get('turns_mean', 0):>10.1f} "
            f"{s.get('tokens_mean', 0):>12.0f} "
            f"{s.get('rewind_count', 0):>10}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
