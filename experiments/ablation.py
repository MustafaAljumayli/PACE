"""
Combinatorial Signal Ablation + Threshold Sweep.

Tests every non-empty subset of the 6 PACE signals × a threshold grid.

Signal subsets (2^6 - 1 = 63 conditions):
  6  solo      (each signal alone)
  15 pairs     (every pair)
  20 triples
  15 quads
  6  quints    (all-but-one)
  1  full      (all 6)

For each condition × threshold combo:
  - Replay pre-recorded trajectories with that signal mask + thresholds
  - Measure: intervention count, accuracy at each point, turns used

Two-stage mode (--coarse then --refine-top-k):
  Stage 1: All 63 subsets with default thresholds (fast)
  Stage 2: Top-K subsets with threshold sweep (thorough)

Usage:
    python experiments/ablation.py --dry-run
    python experiments/ablation.py --trajectories logs/math/sharded-pace/ --coarse
    python experiments/ablation.py --trajectories logs/math/sharded-pace/ --refine-top-k 5
    python experiments/ablation.py --num-samples 10  # live run (costs money)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIC_DIR = PROJECT_ROOT / "lost_in_conversation"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LIC_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)

from pace.trajectory import Trajectory, TurnState, SIGNAL_NAMES
from pace.signals import SignalComputer, HFContradictionScorer
from pace.policy import InterventionPolicy, InterventionConfig, InterventionType
from pace.embeddings import Embedder
from pace.lic import ConversationAnalyzer


SHORT_NAMES = {
    "goal_drift": "GD",
    "shard_coverage": "SC",
    "contradiction": "CT",
    "response_stability": "RS",
    "token_entropy": "TE",
    "repetition": "RP",
}


DEFAULT_THRESHOLD_GRID = {
    "drift_threshold":         [0.35, 0.45, 0.55],
    "coverage_threshold":      [0.25, 0.35, 0.45],
    "contradiction_threshold": [0.50, 0.65, 0.80],
    "stability_threshold":     [0.45, 0.55, 0.65],
    "entropy_threshold":       [1.0, 1.5, 2.0],
    "repetition_threshold":    [0.40, 0.55, 0.70],
}

THRESHOLD_TO_SIGNAL = {
    "drift_threshold": "goal_drift",
    "coverage_threshold": "shard_coverage",
    "contradiction_threshold": "contradiction",
    "stability_threshold": "response_stability",
    "entropy_threshold": "token_entropy",
    "repetition_threshold": "repetition",
}


def generate_signal_subsets(pool: list[str] | None = None) -> list[tuple[str, set[str]]]:
    pool = pool or list(SIGNAL_NAMES)
    conditions = []
    for r in range(1, len(pool) + 1):
        for combo in itertools.combinations(pool, r):
            label = "+".join(SHORT_NAMES.get(s, s) for s in combo)
            conditions.append((label, set(combo)))
    return conditions


def generate_threshold_configs(active_signals: set[str], grid: dict | None = None) -> list[dict]:
    grid = grid or DEFAULT_THRESHOLD_GRID
    relevant: dict[str, list] = {}
    for param, values in grid.items():
        sig = THRESHOLD_TO_SIGNAL.get(param)
        if sig is not None and sig in active_signals:
            relevant[param] = values
    if not relevant:
        return [{}]
    keys = sorted(relevant.keys())
    combos = list(itertools.product(*(relevant[k] for k in keys)))
    return [dict(zip(keys, vals)) for vals in combos]


def count_configurations(pool: list[str] | None = None, include_thresholds: bool = False) -> int:
    subsets = generate_signal_subsets(pool)
    if not include_thresholds:
        return len(subsets)
    return sum(len(generate_threshold_configs(mask)) for _, mask in subsets)


def load_pace_trajectories(log_dir: Path) -> list[dict]:
    """Load PACE-format JSONL logs (with pace_signals and trace)."""
    records = []
    for f in sorted(log_dir.rglob("*.jsonl")):
        content = f.read_text().strip()
        if not content:
            continue
        buf = ""
        depth = 0
        for char in content:
            buf += char
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        records.append(json.loads(buf.strip()))
                    except json.JSONDecodeError:
                        pass
                    buf = ""
    return records


def replay_with_policy(
    record: dict,
    config: InterventionConfig,
    signal_computer: SignalComputer,
) -> dict:
    """Replay a recorded conversation through a specific policy config."""
    trace = record.get("trace", [])
    goal = ""
    shards: list[str] = []

    for msg in trace:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                if not goal:
                    goal = content
                shards.append(content)

    trajectory = Trajectory(goal=goal, episode_id=record.get("task_id", ""), shards=shards)
    policy = InterventionPolicy(config)

    interventions = 0
    intervention_turns = []

    for msg in trace:
        if msg.get("role") != "assistant":
            continue
        response = msg.get("content", "")
        token_top_logprobs = msg.get("token_top_logprobs", [])

        state = TurnState(
            turn_number=len(trajectory.turns),
            response=response,
            answer=response,
            metadata={"token_top_logprobs": token_top_logprobs} if token_top_logprobs else {},
        )
        trajectory.add_turn(state)
        signal_computer.compute(trajectory)

        result = policy.evaluate(trajectory)
        if result.intervention != InterventionType.NONE:
            interventions += 1
            intervention_turns.append(len(trajectory.turns))

    return {
        "task_id": record.get("task_id", ""),
        "is_correct": record.get("is_correct"),
        "score": record.get("score"),
        "total_turns": len(trajectory.turns),
        "interventions": interventions,
        "intervention_turns": intervention_turns,
    }


def run_ablation_sweep(
    records: list[dict],
    conditions: list[tuple[str, set[str]]],
    threshold_sweep: bool = False,
    embedder: Embedder | None = None,
    contradiction_scorer=None,
    verbose: bool = False,
) -> list[dict]:
    """Run ablation across all conditions and optionally threshold configs."""
    results = []
    total_configs = 0

    for cond_name, mask in conditions:
        th_configs = generate_threshold_configs(mask) if threshold_sweep else [{}]

        for th_cfg in th_configs:
            total_configs += 1
            config = InterventionConfig(enabled_signals=mask, **th_cfg)
            sc = SignalComputer(
                embedder=embedder,
                contradiction_scorer=contradiction_scorer if "contradiction" in mask else None,
                signal_mask=mask,
            )

            n_interventions = 0
            n_correct = 0

            for rec in records:
                replay = replay_with_policy(rec, config, sc)
                n_interventions += replay["interventions"]
                if replay.get("is_correct"):
                    n_correct += 1

            n = len(records)
            entry = {
                "condition": cond_name,
                "signals": sorted(mask),
                "num_signals": len(mask),
                "thresholds": th_cfg,
                "n": n,
                "n_correct": n_correct,
                "accuracy": n_correct / max(n, 1),
                "total_interventions": n_interventions,
                "avg_interventions": n_interventions / max(n, 1),
            }
            results.append(entry)

            if verbose:
                print(
                    f"  [{total_configs:>4}] {cond_name:<30} "
                    f"acc={entry['accuracy']:.1%}  iv={entry['avg_interventions']:.1f}"
                )

    return results


def print_ablation_table(results: list[dict]) -> None:
    labels = {1: "SOLO", 2: "PAIRS", 3: "TRIPLES", 4: "QUADS", 5: "QUINTS", 6: "ALL"}
    for num_sigs in range(1, 7):
        group = [r for r in results if r["num_signals"] == num_sigs]
        if not group:
            continue
        group.sort(key=lambda r: r["accuracy"], reverse=True)
        print(f"\n{'─' * 70}")
        print(f"  {labels.get(num_sigs, str(num_sigs))} SIGNALS")
        print(f"{'─' * 70}")
        print(f"  {'Condition':<30} {'Accuracy':>10} {'Avg IV':>10} {'N':>6}")
        print(f"  {'─' * 66}")
        for r in group[:10]:
            print(
                f"  {r['condition']:<30} "
                f"{r['accuracy']:>9.1%} "
                f"{r['avg_interventions']:>10.2f} "
                f"{r['n']:>6}"
            )


def main():
    parser = argparse.ArgumentParser(description="PACE Signal Ablation Study")
    parser.add_argument("--trajectories", default=None,
                        help="Directory with PACE JSONL logs for post-hoc replay")
    parser.add_argument("--output", default="logs/ablation/",
                        help="Output directory")
    parser.add_argument("--coarse", action="store_true",
                        help="Only default thresholds (fast)")
    parser.add_argument("--refine-top-k", type=int, default=0,
                        help="Refine top-K from coarse with threshold sweep")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="Full threshold sweep (expensive)")
    parser.add_argument("--exclude-signals", nargs="*", default=[])
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    # Live run options (generates new data — costs money)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Live run: number of samples to generate")
    parser.add_argument("--task", default="math")
    parser.add_argument("--model", default="gpt-4o-mini")

    args = parser.parse_args()

    pool = [s for s in SIGNAL_NAMES if s not in args.exclude_signals]
    conditions = generate_signal_subsets(pool)

    if args.dry_run:
        print(f"Signal pool: {pool}")
        print(f"Signal subsets: {len(conditions)}")
        total_coarse = len(conditions)
        total_full = count_configurations(pool, include_thresholds=True)
        print(f"Coarse configs: {total_coarse}")
        print(f"Full sweep configs: {total_full}")
        return

    # Load trajectories for post-hoc replay
    if args.trajectories:
        traj_dir = Path(args.trajectories)
        records = load_pace_trajectories(traj_dir)
    else:
        # Look in default locations
        log_dir = PROJECT_ROOT / "logs"
        records = []
        for sub in ["math/sharded-pace", "math/sharded"]:
            d = log_dir / sub
            if d.exists():
                records.extend(load_pace_trajectories(d))

    if not records:
        print("No trajectory data found.")
        print("Run experiments first: python run_pace_experiment.py run --task math --num-samples 10")
        return

    print(f"Loaded {len(records)} conversations")

    embedder = Embedder()
    contradiction_scorer = HFContradictionScorer() if not args.no_nli else None

    sweep = args.threshold_sweep or args.refine_top_k > 0
    results = run_ablation_sweep(
        records, conditions,
        threshold_sweep=sweep and args.refine_top_k == 0,
        embedder=embedder,
        contradiction_scorer=contradiction_scorer,
        verbose=args.verbose,
    )

    print_ablation_table(results)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
