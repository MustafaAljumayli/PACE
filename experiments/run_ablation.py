"""
Signal Ablation Study.

Tests every signal individually, every pair, and all combined to measure
which signals contribute most to PACE's stopping decisions.

Full combinatorial for 5 signals:
  - 5 solo conditions       (S1, S2, S3, S4, S5)
  - 10 pair conditions      (S1+S2, S1+S3, ..., S4+S5)
  - 10 triple conditions    (S1+S2+S3, ...)
  - 5 quad conditions       (all-but-one)
  - 1 full condition        (all 5)
  = 31 total conditions

For each condition:
  - Set signal_mask on both SignalComputer and PolicyConfig
  - Run PACE over pre-recorded trajectories (post-hoc replay)
  - Measure: where PACE would stop, accuracy at that point, turns saved

Usage:
    # First: run fixed-budget experiments to generate full trajectories
    python experiments/run_frames.py --mode single-fixed --num-questions 50

    # Then: run ablation over those trajectories
    python experiments/run_ablation.py --trajectories results/frames/
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pace.trajectory import Trajectory, TurnState, SIGNAL_NAMES
from pace.signals import SignalComputer
from pace.policy import PACEPolicy, PolicyConfig, Decision
from pace.embeddings import Embedder


def generate_ablation_conditions() -> list[tuple[str, set[str]]]:
    """
    Generate all 31 signal subset conditions.

    Returns list of (condition_name, signal_mask) tuples.
    """
    conditions = []

    # Short names for readability
    short = {
        "answer_similarity": "S1",
        "info_gain": "S2",
        "token_entropy": "S3",
        "tool_entropy": "S4",
        "agent_agreement": "S5",
    }

    # All non-empty subsets of size 1..5
    for r in range(1, len(SIGNAL_NAMES) + 1):
        for combo in itertools.combinations(SIGNAL_NAMES, r):
            name = "+".join(short[s] for s in combo)
            conditions.append((name, set(combo)))

    return conditions


def load_trajectories(results_dir: Path) -> list[dict]:
    """Load fixed-budget trajectory data from experiment results."""
    trajectories = []
    for f in results_dir.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        for result in data.get("results", []):
            traj_data = result.get("trajectory_data")
            if traj_data and traj_data.get("turns"):
                trajectories.append({
                    "question_id": result.get("question_id", ""),
                    "gold": result.get("gold", ""),
                    "trajectory": traj_data,
                    # Store the answer at each turn for accuracy evaluation
                    "turn_answers": {
                        t["turn"]: t.get("answer", "")
                        for t in traj_data["turns"]
                    },
                    "final_evaluation": result.get("evaluation", {}),
                })
    return trajectories


def replay_trajectory(
    traj_data: dict,
    policy: PACEPolicy,
    sc: SignalComputer,
) -> dict:
    """Replay a recorded trajectory through PACE with a specific signal mask."""
    turns = traj_data.get("trajectory", {}).get("turns", [])
    traj = Trajectory(query=traj_data.get("trajectory", {}).get("query", ""))

    stop_turn = None
    rewind_turn = None

    for turn_data in turns:
        state = TurnState(
            turn_number=turn_data["turn"],
            answer=turn_data.get("answer", ""),
            tool_called=turn_data.get("tool", ""),
            token_count=turn_data.get("token_count", 0),
        )

        # Restore pre-computed signal values if available
        for sig in SIGNAL_NAMES:
            val = turn_data.get(sig)
            if val is not None:
                setattr(state, sig, val)

        traj.add_turn(state)

        # Recompute signals that are in the mask but weren't pre-recorded
        # (This handles the case where S5 wasn't in the original run)
        sc.compute(traj)

        result = policy.decide(traj)

        if stop_turn is None:
            if result.decision == Decision.STOP:
                stop_turn = turn_data["turn"]
            elif result.decision == Decision.REWIND:
                stop_turn = turn_data["turn"]
                rewind_turn = result.rewind_to_turn

    total_turns = len(turns)
    pace_turns = stop_turn or total_turns
    answer_turn = rewind_turn if rewind_turn is not None else (stop_turn or total_turns)

    return {
        "total_turns": total_turns,
        "pace_stop_turn": stop_turn,
        "pace_rewind_turn": rewind_turn,
        "pace_answer_turn": answer_turn,
        "turns_saved": total_turns - pace_turns,
        "savings_pct": (total_turns - pace_turns) / max(total_turns, 1) * 100,
    }


def run_ablation(
    trajectories: list[dict],
    conditions: list[tuple[str, set[str]]],
) -> list[dict]:
    """Run all ablation conditions over all trajectories."""
    results = []

    for cond_name, mask in conditions:
        sc = SignalComputer(signal_mask=mask)
        policy = PACEPolicy(PolicyConfig(signal_mask=mask))

        total_saved = 0
        total_baseline = 0
        rewind_count = 0
        early_stop_count = 0

        for traj_data in trajectories:
            replay = replay_trajectory(traj_data, policy, sc)
            total_saved += replay["turns_saved"]
            total_baseline += replay["total_turns"]
            if replay["pace_rewind_turn"] is not None:
                rewind_count += 1
            if replay["pace_stop_turn"] is not None:
                early_stop_count += 1

        n = len(trajectories)
        avg_savings = total_saved / max(total_baseline, 1) * 100
        avg_turns_baseline = total_baseline / max(n, 1)
        avg_turns_pace = (total_baseline - total_saved) / max(n, 1)

        results.append({
            "condition": cond_name,
            "signals": sorted(mask),
            "num_signals": len(mask),
            "avg_turns_baseline": round(avg_turns_baseline, 2),
            "avg_turns_pace": round(avg_turns_pace, 2),
            "avg_savings_pct": round(avg_savings, 2),
            "early_stops": early_stop_count,
            "early_stop_pct": round(early_stop_count / max(n, 1) * 100, 1),
            "rewinds": rewind_count,
            "n": n,
        })

    return results


def print_ablation_table(results: list[dict]) -> None:
    """Print a formatted ablation table sorted by group size then savings."""
    # Group by number of signals
    for num_sigs in range(1, 6):
        group = [r for r in results if r["num_signals"] == num_sigs]
        if not group:
            continue

        group.sort(key=lambda r: r["avg_savings_pct"], reverse=True)

        if num_sigs == 1:
            header = "SOLO SIGNALS"
        elif num_sigs == 2:
            header = "SIGNAL PAIRS"
        elif num_sigs == 3:
            header = "SIGNAL TRIPLES"
        elif num_sigs == 4:
            header = "SIGNAL QUADS (ALL-BUT-ONE)"
        else:
            header = "ALL SIGNALS"

        print(f"\n{'─'*60}")
        print(f"  {header}")
        print(f"{'─'*60}")
        print(f"  {'Condition':<20} {'Savings%':>10} {'Avg Turns':>10} {'Stops%':>10} {'Rewinds':>8}")
        print(f"  {'─'*58}")
        for r in group:
            print(
                f"  {r['condition']:<20} "
                f"{r['avg_savings_pct']:>9.1f}% "
                f"{r['avg_turns_pace']:>10.1f} "
                f"{r['early_stop_pct']:>9.1f}% "
                f"{r['rewinds']:>8}"
            )


def main():
    parser = argparse.ArgumentParser(description="PACE Signal Ablation Study")
    parser.add_argument("--trajectories", default="results/frames/")
    parser.add_argument("--output", default="results/ablation_results.json")
    args = parser.parse_args()

    trajectories = load_trajectories(Path(args.trajectories))
    if not trajectories:
        print(f"No trajectories found in {args.trajectories}")
        print("Run fixed-budget experiments first:")
        print("  python experiments/run_frames.py --mode single-fixed --num-questions 50")
        return

    print(f"Loaded {len(trajectories)} trajectories")

    conditions = generate_ablation_conditions()
    print(f"Running {len(conditions)} ablation conditions...\n")

    results = run_ablation(trajectories, conditions)

    # Print table
    print_ablation_table(results)

    # Highlight key findings
    solos = [r for r in results if r["num_signals"] == 1]
    best_solo = max(solos, key=lambda r: r["avg_savings_pct"])
    full = [r for r in results if r["num_signals"] == 5][0]

    print(f"\n{'='*60}")
    print(f"KEY FINDINGS")
    print(f"{'='*60}")
    print(f"  Best single signal: {best_solo['condition']} ({best_solo['avg_savings_pct']:.1f}% savings)")
    print(f"  All signals combined: {full['avg_savings_pct']:.1f}% savings")
    print(f"  Marginal gain of adding all signals vs best solo: "
          f"{full['avg_savings_pct'] - best_solo['avg_savings_pct']:+.1f}%")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
