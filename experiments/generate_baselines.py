"""
Batch baseline generation across all domains × models with logprobs.

Runs LiC simulations for each (model, task, conv_type) triple, capturing
token logprobs for PACE's token_entropy signal.  Outputs go to
PACE/logs/baselines/{task}/{conv_type}/ in LiC JSONL format.

Phase 1: Generate raw conversation data (API cost)
Phase 2: Compute PACE signals post-hoc (embedding cost only)

Usage:
    # Generate math baselines for a single model
    python experiments/generate_baselines.py \\
        --models gpt-4o-mini \\
        --tasks math \\
        --conv-types full sharded

    # All domains, all available models
    python experiments/generate_baselines.py --all-models --all-tasks

    # Dry run to see what will be generated
    python experiments/generate_baselines.py --dry-run --all-models --all-tasks

    # Only models that support logprobs
    python experiments/generate_baselines.py --logprobs-only --all-tasks

    # Control parallelism
    python experiments/generate_baselines.py --workers 4 --models gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Ensure imports work regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIC_DIR = PROJECT_ROOT / "lost_in_conversation"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LIC_DIR))

# Load .env before any API clients initialize
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)

from pace.models import (
    MODEL_REGISTRY, MODEL_REGISTRY_ALL, available_models, get_model, ModelConfig
)


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

ALL_TASKS = ["math", "code", "database", "actions", "data2text", "summary"]
CONV_TYPES = ["full", "sharded", "concat"]
DEFAULT_DATASET = "data/sharded_instructions_600.json"
LOG_ROOT = PROJECT_ROOT / "logs" / "baselines"

# System/user models are always gpt-4o-mini (cheap, reliable)
SYSTEM_MODEL = "gpt-4o-mini"
USER_MODEL = "gpt-4o-mini"


# ──────────────────────────────────────────────
#  Task loading
# ──────────────────────────────────────────────

def load_samples(
    dataset_file: str | Path,
    tasks: list[str] | None = None,
) -> list[dict]:
    """Load samples from LiC dataset JSON, optionally filtered by task."""
    path = LIC_DIR / dataset_file if not Path(dataset_file).is_absolute() else Path(dataset_file)
    with open(path) as f:
        samples = json.load(f)
    if tasks:
        samples = [s for s in samples if s["task"] in tasks]
    return samples


def count_existing_runs(
    task: str, conv_type: str, model_id: str, dataset_fn: str,
) -> Counter:
    """Count already-completed runs to skip duplicates."""
    log_dir = LOG_ROOT / task / conv_type
    if not log_dir.exists():
        return Counter()

    sanitized = model_id
    for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        sanitized = sanitized.replace(ch, '_')

    counts: Counter = Counter()
    for fn in log_dir.iterdir():
        if not fn.name.endswith(".jsonl"):
            continue
        if sanitized not in fn.name:
            continue
        try:
            with open(fn) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("assistant_model") == model_id:
                            counts[rec.get("task_id", "")] += 1
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue
    return counts


# ──────────────────────────────────────────────
#  Simulation runner
# ──────────────────────────────────────────────

def run_single(todo: dict, verbose: bool = False) -> dict:
    """Run a single simulation (full or sharded) with logprobs."""
    os.chdir(str(LIC_DIR))

    from simulator_sharded import ConversationSimulatorSharded
    from simulator_full import ConversationSimulatorFull

    sample = todo["sample"]
    model_id = todo["model_id"]
    conv_type = todo["conv_type"]
    log_folder = str(todo["log_folder"])
    model_cfg: ModelConfig = todo["model_cfg"]

    try:
        if conv_type == "full":
            sim = ConversationSimulatorFull(
                sample,
                assistant_model=model_id,
                system_model=SYSTEM_MODEL,
                dataset_fn=todo["dataset_fn"],
                log_folder=log_folder,
            )
        elif conv_type == "concat":
            sim = ConversationSimulatorFull(
                sample,
                assistant_model=model_id,
                system_model=SYSTEM_MODEL,
                run_concat=True,
                dataset_fn=todo["dataset_fn"],
                log_folder=log_folder,
            )
        elif conv_type == "sharded":
            sim = ConversationSimulatorSharded(
                sample,
                assistant_model=model_id,
                system_model=SYSTEM_MODEL,
                user_model=USER_MODEL,
                dataset_fn=todo["dataset_fn"],
                log_folder=log_folder,
            )
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        is_correct, score = sim.run(verbose=verbose, save_log=True)
        return {
            "task_id": sample["task_id"],
            "task": sample["task"],
            "model": model_id,
            "conv_type": conv_type,
            "is_correct": is_correct,
            "score": score,
            "status": "ok",
        }
    except Exception as e:
        import traceback
        return {
            "task_id": sample.get("task_id", "?"),
            "task": sample.get("task", "?"),
            "model": model_id,
            "conv_type": conv_type,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate LiC baselines across models × domains with logprobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model short names (from pace.models registry)")
    parser.add_argument("--all-models", action="store_true",
                        help="Use all models with available API keys")
    parser.add_argument("--logprobs-only", action="store_true",
                        help="Only models that support logprobs")
    parser.add_argument("--tasks", nargs="*", default=["math"],
                        help="Tasks to generate (default: math)")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Run all 6 LiC domains")
    parser.add_argument("--conv-types", nargs="*", default=["full", "sharded"],
                        help="Conversation types: full, sharded, concat")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Dataset file relative to LiC directory")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Number of runs per sample (for variance estimation)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (>1 for concurrent API calls)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without running")
    parser.add_argument("--shuffle", action="store_true",
                        help="Randomize execution order")
    args = parser.parse_args()

    # Determine tasks
    tasks = ALL_TASKS if args.all_tasks else (args.tasks or ["math"])

    # Determine models
    if args.all_models:
        model_names = available_models()
    elif args.models:
        model_names = args.models
    else:
        model_names = ["gpt-4o-mini"]

    if args.logprobs_only:
        model_names = [m for m in model_names if get_model(m).supports_logprobs]

    # Validate models
    for m in model_names:
        cfg = get_model(m)
        key = os.environ.get(cfg.env_var, "")
        if not key:
            print(f"WARNING: {m} requires {cfg.env_var} but it's not set. Skipping.")
            model_names = [n for n in model_names if n != m]

    # Load samples
    samples = load_samples(args.dataset, tasks)
    task_counts = Counter(s["task"] for s in samples)
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {len(samples)} ({dict(task_counts)})")
    print(f"Models:  {model_names}")
    print(f"Types:   {args.conv_types}")
    print(f"Runs:    {args.n_runs} per sample")

    # Build work queue
    todos = []
    skipped = 0

    for model_name in model_names:
        cfg = get_model(model_name)
        # Use the API ID as the model string passed to generate()
        # The model_openai.py will route to the right provider
        model_id = model_name  # Use canonical name; model_openai routes via registry

        for conv_type in args.conv_types:
            for task in tasks:
                task_samples = [s for s in samples if s["task"] == task]
                existing = count_existing_runs(task, conv_type, model_id, args.dataset)
                log_folder = str(LOG_ROOT / task / conv_type)

                for sample in task_samples:
                    n_done = existing.get(sample["task_id"], 0)
                    n_needed = max(0, args.n_runs - n_done)
                    if n_needed == 0:
                        skipped += 1
                        continue

                    for _ in range(n_needed):
                        todos.append({
                            "sample": sample,
                            "model_id": model_id,
                            "model_cfg": cfg,
                            "conv_type": conv_type,
                            "dataset_fn": args.dataset,
                            "log_folder": log_folder,
                        })

    print(f"\nTotal: {len(todos)} runs to execute ({skipped} already done)")

    if args.dry_run:
        by_model = Counter(t["model_id"] for t in todos)
        by_task = Counter(t["sample"]["task"] for t in todos)
        by_type = Counter(t["conv_type"] for t in todos)
        print(f"\nBy model: {dict(by_model)}")
        print(f"By task:  {dict(by_task)}")
        print(f"By type:  {dict(by_type)}")

        total_cost_est = 0
        for model_name, count in by_model.items():
            tier = get_model(model_name).cost_tier
            est = {"cheap": 0.002, "mid": 0.01, "expensive": 0.05}.get(tier, 0.01)
            model_cost = count * est
            total_cost_est += model_cost
            print(f"  {model_name}: ~${model_cost:.2f} ({tier})")
        print(f"\nEstimated total cost: ~${total_cost_est:.2f}")
        return

    if not todos:
        print("Nothing to run — all baselines already generated.")
        return

    if args.shuffle:
        random.shuffle(todos)

    # Execute
    print(f"\nStarting generation with {args.workers} worker(s)...")

    try:
        import tqdm
        progress = tqdm.tqdm(total=len(todos), desc="Generating")
    except ImportError:
        progress = None

    results = []
    errors = []

    def _run_and_track(todo):
        result = run_single(todo, verbose=args.verbose)
        if result["status"] == "error":
            errors.append(result)
        results.append(result)
        if progress:
            progress.update(1)
        return result

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(_run_and_track, todos))
    else:
        for todo in todos:
            _run_and_track(todo)

    if progress:
        progress.close()

    # Summary
    ok_count = sum(1 for r in results if r["status"] == "ok")
    err_count = len(errors)
    print(f"\nDone: {ok_count} succeeded, {err_count} failed")

    if errors:
        print("\nErrors:")
        for e in errors[:5]:
            print(f"  {e['task_id']} ({e['model']}, {e['conv_type']}): {e['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    # Save run summary
    summary_path = LOG_ROOT / "generation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": model_names,
            "tasks": tasks,
            "conv_types": args.conv_types,
            "total_runs": len(results),
            "succeeded": ok_count,
            "failed": err_count,
            "results_summary": [
                {k: v for k, v in r.items() if k != "traceback"}
                for r in results
            ],
        }, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
