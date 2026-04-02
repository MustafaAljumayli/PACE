"""
Threshold Sweep: Sensitivity Analysis for PACE Parameters.

Runs PACE over pre-recorded trajectories with different threshold
combinations to find optimal values and produce sensitivity plots.

This is a POST-HOC analysis — it doesn't re-run agents, just replays
existing trajectories through PACE with different configs.

Usage:
    # First: run experiments to generate trajectories
    python experiments/run_frames.py --mode single-fixed --num-questions 50

    # Then: sweep thresholds over those trajectories
    python experiments/sweep_thresholds.py --trajectories results/frames/
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from pace.trajectory import Trajectory, TurnState
from pace.signals import SignalComputer
from pace.policy import PACEPolicy, PolicyConfig, Decision


def load_trajectories(results_dir: Path) -> list[dict]:
    """Load trajectories from fixed-budget experiment results."""
    trajectories = []
    for f in results_dir.glob("*fixed*.json"):
        with open(f) as fh:
            data = json.load(fh)
        for result in data.get("results", []):
            if result.get("trajectory_data"):
                trajectories.append(result)
    return trajectories


def replay_with_policy(
    trajectory_data: dict,
    policy: PACEPolicy,
    sc: SignalComputer,
) -> dict:
    """
    Replay a recorded trajectory through a PACE policy.
    Returns what PACE would have decided at each turn.
    """
    turns = trajectory_data.get("turns", [])
    traj = Trajectory(query=trajectory_data.get("query", ""))

    decisions = []
    stop_turn = None
    rewind_turn = None

    for turn_data in turns:
        state = TurnState(
            turn_number=turn_data["turn"],
            answer=turn_data.get("answer", ""),
            tool_called=turn_data.get("tool", ""),
            token_count=turn_data.get("token_count", 0),
        )
        # Re-populate signals if they exist
        if turn_data.get("answer_similarity") is not None:
            state.answer_similarity = turn_data["answer_similarity"]
        if turn_data.get("info_gain") is not None:
            state.info_gain = turn_data["info_gain"]

        traj.add_turn(state)

        # Only compute signals if they weren't pre-recorded
        if state.answer_similarity is None:
            sc.compute(traj)

        result = policy.decide(traj)
        decisions.append({
            "turn": turn_data["turn"],
            "decision": result.decision.name,
        })

        if stop_turn is None and result.decision in (Decision.STOP, Decision.REWIND):
            stop_turn = turn_data["turn"]
            if result.decision == Decision.REWIND:
                rewind_turn = result.rewind_to_turn

    return {
        "total_turns": len(turns),
        "pace_stop_turn": stop_turn,
        "pace_rewind_turn": rewind_turn,
        "turns_saved": (len(turns) - stop_turn) if stop_turn else 0,
        "decisions": decisions,
    }


def run_sweep(
    trajectories: list[dict],
    sweep_config: dict,
) -> list[dict]:
    """Run threshold sweep across all parameter combinations."""
    sc = SignalComputer()

    # Generate all combinations
    param_names = list(sweep_config.keys())
    param_values = list(sweep_config.values())
    combos = list(itertools.product(*param_values))

    print(f"Sweeping {len(combos)} configurations over {len(trajectories)} trajectories")

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(param_names, combo))

        config = PolicyConfig(**params)
        policy = PACEPolicy(config)

        total_turns_saved = 0
        total_turns_baseline = 0
        rewind_count = 0

        for traj_data in trajectories:
            td = traj_data.get("trajectory_data", traj_data)
            replay = replay_with_policy(td, policy, sc)
            total_turns_saved += replay["turns_saved"]
            total_turns_baseline += replay["total_turns"]
            if replay["pace_rewind_turn"] is not None:
                rewind_count += 1

        avg_savings_pct = (
            total_turns_saved / max(total_turns_baseline, 1) * 100
        )

        results.append({
            **params,
            "avg_turn_savings_pct": round(avg_savings_pct, 2),
            "total_turns_saved": total_turns_saved,
            "total_turns_baseline": total_turns_baseline,
            "rewind_count": rewind_count,
        })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(combos)} configs evaluated...")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default="results/frames/")
    parser.add_argument("--config", default="configs/pace_thresholds.yaml")
    parser.add_argument("--output", default="results/sweep_results.json")
    args = parser.parse_args()

    # Load sweep config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    sweep_config = config.get("sweep", {})

    if not sweep_config:
        print("No sweep config found. Add a 'sweep' section to pace_thresholds.yaml")
        return

    # Load trajectories
    trajectories = load_trajectories(Path(args.trajectories))
    if not trajectories:
        print(f"No trajectories found in {args.trajectories}. Run experiments first.")
        return

    print(f"Loaded {len(trajectories)} trajectories")

    # Run sweep
    results = run_sweep(trajectories, sweep_config)

    # Sort by savings
    results.sort(key=lambda r: r["avg_turn_savings_pct"], reverse=True)

    # Print top 10
    print(f"\n{'='*70}")
    print("TOP 10 CONFIGURATIONS BY TURN SAVINGS")
    print(f"{'='*70}")
    for r in results[:10]:
        params = {k: v for k, v in r.items() if k in sweep_config}
        print(f"  Savings: {r['avg_turn_savings_pct']:.1f}% | Rewinds: {r['rewind_count']} | {params}")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
