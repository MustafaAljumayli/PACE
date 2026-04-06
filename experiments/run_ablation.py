"""
Combinatorial Signal Ablation Study — Multi-Model.

DEPRECATED: Use experiments/ablation.py instead, which uses the updated
6-signal framework (goal_drift, shard_coverage, contradiction,
response_stability, token_entropy, repetition).

This module is kept for backward compatibility with the trajectory loading
and multi-model iteration logic. It now delegates to the new ablation module.

Usage:
    python experiments/ablation.py --dry-run               # use the new module
    python experiments/run_ablation.py --dry-run            # this wrapper still works
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pace.trajectory import SIGNAL_NAMES
from experiments.ablation import (
    generate_signal_subsets,
    generate_threshold_configs,
    count_configurations,
    load_pace_trajectories,
    run_ablation_sweep,
    print_ablation_table,
    SHORT_NAMES,
    DEFAULT_THRESHOLD_GRID,
)
from pace.signals import SignalComputer, HFContradictionScorer
from pace.embeddings import Embedder


def main():
    parser = argparse.ArgumentParser(
        description="PACE Combinatorial Signal Ablation Study (wrapper → experiments/ablation.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trajectories", default="logs/",
                        help="Directory containing trajectory logs")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output", default="logs/ablation/",
                        help="Output directory for results")
    parser.add_argument("--coarse", action="store_true")
    parser.add_argument("--refine-top-k", type=int, default=0)
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--exclude-signals", nargs="*", default=[])
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pool = [s for s in SIGNAL_NAMES if s not in args.exclude_signals]
    conditions = generate_signal_subsets(pool)

    if args.dry_run:
        print(f"Signal pool ({len(pool)}): {pool}")
        print(f"Signal subsets: {len(conditions)}")
        print(f"Coarse configs: {count_configurations(pool)}")
        print(f"Full sweep configs: {count_configurations(pool, include_thresholds=True)}")
        return

    traj_dir = Path(args.trajectories)
    if not traj_dir.exists():
        print(f"Trajectory directory not found: {traj_dir}")
        return

    records = load_pace_trajectories(traj_dir)
    if not records:
        print("No trajectories found.")
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
