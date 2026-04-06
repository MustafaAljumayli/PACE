#!/usr/bin/env python3
"""
PACE Experiment Runner — CLI for running, analysing, and reporting experiments.

Subcommands:
    run       Run PACE-instrumented LiC simulations (sharded with signals + interventions)
    analyze   Compute PACE signals post-hoc on existing LiC log files
    report    Generate research-quality tables from experiment logs

Usage:
    python run_pace_experiment.py run --task math --model gpt-4o-mini --num-samples 10
    python run_pace_experiment.py run --task math --model gpt-4o-mini --num-samples 103 --intervene
    python run_pace_experiment.py analyze --log-path logs/math/sharded/sharded_math_gpt-4o-mini.jsonl
    python run_pace_experiment.py report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LIC_DIR = PROJECT_ROOT / "lost_in_conversation"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LIC_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)

from pace.trajectory import Trajectory, TurnState, SIGNAL_NAMES
from pace.signals import SignalComputer, HFContradictionScorer
from pace.policy import InterventionPolicy, InterventionConfig, InterventionType
from pace.embeddings import Embedder
from pace.extract import robust_math_eval


def _resolve_pace_log_folder() -> Path:
    """All PACE experiments log to PROJECT_ROOT/logs (not LiC/logs)."""
    d = PROJECT_ROOT / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ──────────────────────────────────────────────
#  run subcommand
# ──────────────────────────────────────────────

def cmd_run(args):
    os.chdir(str(LIC_DIR))

    from model_openai import generate, generate_json
    from simulator_sharded import ConversationSimulatorSharded
    from system_agent import SystemAgent
    from tasks import get_task
    from utils import extract_conversation, date_str
    from utils_log import log_conversation

    log_folder = str(_resolve_pace_log_folder())

    dataset_path = LIC_DIR / "data" / "sharded_instructions_600.json"
    with open(dataset_path) as f:
        samples = json.load(f)
    samples = [s for s in samples if s["task"] == args.task]

    if args.num_samples and args.num_samples < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, args.num_samples)

    print(f"Running {len(samples)} {args.task} samples with {args.model}")
    print(f"Intervention: {'ON' if args.intervene else 'OFF'}")
    print(f"Logs: {log_folder}")

    embedder = Embedder()
    contradiction_scorer = HFContradictionScorer() if not args.no_nli else None
    signal_computer = SignalComputer(
        embedder=embedder,
        contradiction_scorer=contradiction_scorer,
    )
    policy = InterventionPolicy(InterventionConfig(
        intervention_type=InterventionType.RECAP if args.intervene else InterventionType.NONE,
    ))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = []
    total_cost = 0.0

    try:
        from tqdm import tqdm
        sample_iter = tqdm(samples, desc="Samples")
    except ImportError:
        sample_iter = samples

    for sample in sample_iter:
        result = _run_single_sample(
            sample, args, signal_computer, policy, embedder,
            generate, generate_json, log_folder,
        )
        results.append(result)
        total_cost += result.get("cost", 0)

    n_correct = sum(1 for r in results if r.get("is_correct"))
    accuracy = n_correct / len(results) if results else 0

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat() + "Z",
        "mode": "run",
        "task": args.task,
        "model": args.model,
        "dataset": str(dataset_path),
        "intervene": args.intervene,
        "num_samples_requested": args.num_samples or len(samples),
        "num_samples_completed": len(results),
        "accuracy": accuracy,
        "total_cost_usd": total_cost,
        "mean_interventions": sum(r.get("interventions", 0) for r in results) / max(len(results), 1),
        "log_folder": log_folder,
        "results": results,
    }

    manifest_path = Path(log_folder) / "pace_runs" / f"{run_id}_{args.task}_{args.model}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAccuracy: {n_correct}/{len(results)} ({accuracy:.1%})")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Manifest: {manifest_path}")

    _update_report(log_folder)
    _generate_run_report(log_folder, args.task, args.model, args.intervene)


def _run_single_sample(sample, args, signal_computer, policy, embedder,
                        generate, generate_json, log_folder):
    from simulator_sharded import ConversationSimulatorSharded
    from system_agent import SystemAgent
    from tasks import get_task
    from utils import extract_conversation, date_str
    from utils_log import log_conversation

    task_name = sample["task"]
    task = get_task(task_name)
    system_msg = task.generate_system_prompt(sample)
    shards = sample["shards"]
    system_agent = SystemAgent(task_name, "gpt-4o-mini", sample)

    policy.reset()
    goal = shards[0]["shard"] if shards else ""
    revealed_shard_texts = [goal]

    trajectory = Trajectory(
        goal=goal,
        episode_id=sample["task_id"],
        shards=revealed_shard_texts,
    )

    sim = ConversationSimulatorSharded(
        sample,
        assistant_model=args.model,
        system_model="gpt-4o-mini",
        user_model="gpt-4o-mini",
        dataset_fn="data/sharded_instructions_600.json",
        log_folder=log_folder,
    )

    is_correct, score = sim.run(verbose=args.verbose, save_log=True)

    # Post-hoc signal computation on the saved trace
    n_interventions = 0
    for msg in sim.trace:
        if msg.get("role") == "assistant":
            response_text = msg.get("content", "")
            token_top_logprobs = msg.get("token_top_logprobs", [])

            state = TurnState(
                turn_number=len(trajectory.turns),
                response=response_text,
                answer=response_text,
                metadata={"token_top_logprobs": token_top_logprobs} if token_top_logprobs else {},
            )
            trajectory.add_turn(state)

            # Update shards as they're revealed
            trajectory.shards = revealed_shard_texts[:]
            signal_computer.compute(trajectory)

            # Check policy
            result = policy.evaluate(trajectory)
            if result.intervention != InterventionType.NONE:
                n_interventions += 1

        elif msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                revealed_shard_texts.append(content)

    # Apply robust math extraction if applicable
    if task_name == "math" and not is_correct:
        gold = sample.get("answer", "")
        if not gold:
            for shard in shards:
                text = shard.get("shard", "")
                if "####" in text:
                    gold = text.split("####")[-1].strip()
                    break
        if gold and trajectory.turns:
            last_response = trajectory.turns[-1].response
            eval_result = robust_math_eval(last_response, gold)
            if eval_result.get("is_correct"):
                is_correct = True
                score = 1.0

    # Save PACE signals alongside the LiC trace
    conv_type = "sharded-pace" if args.intervene else "sharded-pace-nointervene"
    pace_signals = trajectory.to_dict()

    # Write supplementary PACE log
    pace_log_dir = Path(log_folder) / task_name / conv_type
    pace_log_dir.mkdir(parents=True, exist_ok=True)
    sanitized = args.model.replace("/", "_").replace(":", "_")
    pace_log_file = pace_log_dir / f"{conv_type}_{task_name}_{sanitized}.jsonl"

    pace_record = {
        "conv_id": str(hash(f"{sample['task_id']}_{time.time()}") % (10**24)),
        "conv_type": conv_type,
        "task": task_name,
        "task_id": sample["task_id"],
        "dataset_fn": "sharded_instructions_600.json",
        "assistant_model": args.model,
        "system_model": "gpt-4o-mini",
        "user_model": "gpt-4o-mini",
        "git_version": "",
        "trace": sim.trace,
        "is_correct": is_correct,
        "score": score if score is not None else (1.0 if is_correct else 0.0),
        "pace_signals": pace_signals,
        "interventions_used": n_interventions,
        "conv_cost_usd": sum(
            msg.get("cost_usd", 0)
            for msg in sim.trace
            if isinstance(msg.get("cost_usd"), (int, float))
        ),
    }

    with open(pace_log_file, "a") as f:
        f.write(json.dumps(pace_record) + "\n")

    return {
        "task_id": sample["task_id"],
        "is_correct": is_correct,
        "score": score if score is not None else (1.0 if is_correct else 0.0),
        "interventions": n_interventions,
        "cost": pace_record["conv_cost_usd"],
        "turns": len(trajectory.turns),
    }


# ──────────────────────────────────────────────
#  analyze subcommand
# ──────────────────────────────────────────────

def cmd_analyze(args):
    from pace.lic import ConversationAnalyzer

    analyzer = ConversationAnalyzer(use_nli=not args.no_nli)

    output = args.output
    if not output:
        output = str(_resolve_pace_log_folder() / "signal_analysis.jsonl")

    print(f"Analyzing: {args.log_path}")
    print(f"Output: {output}")

    results = analyzer.analyze_log_file(
        args.log_path,
        show_progress=True,
        output_path=output,
    )

    n = len(results)
    n_correct = sum(1 for r in results if r.get("is_correct"))
    print(f"\nAnalyzed {n} conversations")
    print(f"Correct: {n_correct}/{n} ({n_correct/max(n,1):.1%})")

    for sig in SIGNAL_NAMES:
        vals = [
            r["signals_summary"][sig]["mean"]
            for r in results
            if r["signals_summary"][sig]["mean"] is not None
        ]
        if vals:
            print(f"  {sig}: mean={sum(vals)/len(vals):.4f}, min={min(vals):.4f}, max={max(vals):.4f}")


# ──────────────────────────────────────────────
#  report subcommand
# ──────────────────────────────────────────────

def cmd_report(args):
    from experiments.report import generate_full_report
    log_folder = str(_resolve_pace_log_folder())
    generate_full_report(log_folder, latex=args.latex)


def _update_report(log_folder: str):
    """Auto-update report.md after each run."""
    try:
        from experiments.report import generate_full_report
        generate_full_report(log_folder, latex=False)
    except Exception as e:
        print(f"Warning: could not auto-generate report: {e}")


def _generate_run_report(log_folder: str, task: str, model: str, intervene: bool):
    """Auto-generate per-run signal analysis report to reports/{task}/{model}/."""
    try:
        from experiments.report import generate_run_report, _parse_records
        conv_type = "sharded-pace" if intervene else "sharded-pace-nointervene"
        sanitized = model.replace("/", "_").replace(":", "_")
        log_file = Path(log_folder) / task / conv_type / f"{conv_type}_{task}_{sanitized}.jsonl"
        if log_file.exists():
            records = _parse_records(log_file)
            if records:
                path = generate_run_report(records, task, model, conv_type, latex=True)
                print(f"Run report: {path}")
    except Exception as e:
        print(f"Warning: could not generate run report: {e}")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PACE Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run PACE-instrumented LiC simulations")
    p_run.add_argument("--task", default="math", choices=["math", "code", "database", "actions", "data2text", "summary"])
    p_run.add_argument("--model", default="gpt-4o-mini")
    p_run.add_argument("--num-samples", type=int, default=None)
    p_run.add_argument("--intervene", action="store_true", help="Enable PACE interventions")
    p_run.add_argument("--no-nli", action="store_true", help="Skip NLI model (faster, no contradiction signal)")
    p_run.add_argument("--verbose", "-v", action="store_true")
    p_run.set_defaults(func=cmd_run)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Compute PACE signals on existing logs")
    p_analyze.add_argument("--log-path", required=True)
    p_analyze.add_argument("--output", default=None)
    p_analyze.add_argument("--no-nli", action="store_true")
    p_analyze.set_defaults(func=cmd_analyze)

    # report
    p_report = sub.add_parser("report", help="Generate research-quality tables")
    p_report.add_argument("--latex", action="store_true", help="Include LaTeX tables")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
