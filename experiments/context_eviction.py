"""
Context Eviction Experiment — tests whether evicting multi-turn history
and re-prompting with a consolidated single-shot prompt recovers accuracy.

Three strategies:
  recap       — inject a recap message summarising revealed shards (PACE default)
  evict       — on signal trigger, evict history and re-prompt with all shards in one shot
  evict-always— always consolidate at every evaluation step (upper bound)

The paper shows full (single-shot) conversations significantly outperform
sharded multi-turn. This experiment tests whether automatic eviction +
consolidation can recover that gap when PACE signals detect degradation.

Usage:
    python experiments/context_eviction.py --strategy evict --num-samples 10 --verbose
    python experiments/context_eviction.py --strategy evict-always --num-samples 10
    python experiments/context_eviction.py --strategy recap --num-samples 103
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
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
from pace.extract import robust_math_eval


def build_consolidated_prompt(system_msg: str, shards: list[str]) -> list[dict]:
    """
    Build a fresh single-shot prompt from the system message + all revealed shards.
    This converts a multi-turn conversation back to a single-turn (full) prompt.
    """
    combined = "\n\n".join(shards)
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": combined},
    ]


def run_eviction_experiment(
    sample: dict,
    strategy: str,
    model: str,
    generate_fn,
    generate_json_fn,
    signal_computer: SignalComputer,
    policy: InterventionPolicy,
    verbose: bool = False,
) -> dict:
    """
    Run a single conversation with the specified eviction strategy.

    Returns dict with is_correct, score, interventions, cost, turns, strategy info.
    """
    from tasks import get_task
    from system_agent import SystemAgent
    from utils import date_str

    task_name = sample["task"]
    task = get_task(task_name)
    system_msg = task.generate_system_prompt(sample)
    shards = sample["shards"]
    system_agent = SystemAgent(task_name, "gpt-4o-mini", sample)

    trace: list[dict] = []
    messages: list[dict] = [{"role": "system", "content": system_msg}]
    revealed_shard_texts: list[str] = []
    total_cost = 0.0
    n_evictions = 0

    goal = shards[0]["shard"] if shards else ""
    trajectory = Trajectory(goal=goal, episode_id=sample["task_id"])

    policy.reset()

    for i, shard_info in enumerate(shards):
        shard_text = shard_info["shard"]
        revealed_shard_texts.append(shard_text)
        trajectory.shards = revealed_shard_texts[:]

        messages.append({"role": "user", "content": shard_text})
        trace.append({"role": "user", "content": shard_text, "timestamp": date_str()})

        # Check if we should evict BEFORE generating
        should_evict = False
        if strategy == "evict-always" and i > 0:
            should_evict = True
        elif strategy == "evict" and len(trajectory.turns) > 0:
            result = policy.evaluate(trajectory)
            if result.intervention != InterventionType.NONE:
                should_evict = True

        if should_evict:
            n_evictions += 1
            messages = build_consolidated_prompt(system_msg, revealed_shard_texts)
            trace.append({
                "role": "log",
                "content": {"type": "context_eviction", "eviction_number": n_evictions},
                "timestamp": date_str(),
            })
            if verbose:
                print(f"  [EVICT #{n_evictions}] at turn {i}, consolidating {len(revealed_shard_texts)} shards")

        # Generate response
        response_obj = generate_fn(
            messages, model=model, return_metadata=True,
            logprobs=True, top_logprobs=5,
        )
        response_text = response_obj["message"] if isinstance(response_obj, dict) else response_obj
        cost = response_obj.get("total_usd", 0) if isinstance(response_obj, dict) else 0
        total_cost += cost

        messages.append({"role": "assistant", "content": response_text})
        trace_entry = {"role": "assistant", "content": response_text, "timestamp": date_str(), "cost_usd": cost}
        token_top_logprobs = response_obj.get("token_top_logprobs", []) if isinstance(response_obj, dict) else []
        if token_top_logprobs:
            trace_entry["token_top_logprobs"] = token_top_logprobs
        trace.append(trace_entry)

        # Compute PACE signals
        state = TurnState(
            turn_number=len(trajectory.turns),
            response=response_text,
            answer=response_text,
            metadata={"token_top_logprobs": token_top_logprobs} if token_top_logprobs else {},
        )
        trajectory.add_turn(state)
        signal_computer.compute(trajectory)

        if verbose:
            sigs = trajectory.latest.signals_dict()
            print(f"  Turn {i}: {', '.join(f'{k}={v:.3f}' if v is not None else f'{k}=null' for k,v in sigs.items())}")

        # Run system verification
        try:
            system_check = system_agent.verify(response_text)
        except Exception:
            system_check = {"response_type": "continue"}

    # Evaluate final answer
    gold_answer = sample.get("answer", "")
    if not gold_answer:
        for s in shards:
            text = s.get("shard", "")
            if "####" in text:
                gold_answer = text.split("####")[-1].strip()
                break

    is_correct = False
    score = 0.0
    if trajectory.turns and gold_answer:
        last_resp = trajectory.turns[-1].response
        eval_result = robust_math_eval(last_resp, gold_answer)
        is_correct = eval_result.get("is_correct", False)
        score = 1.0 if is_correct else 0.0

    return {
        "task_id": sample["task_id"],
        "strategy": strategy,
        "is_correct": is_correct,
        "score": score,
        "evictions": n_evictions,
        "turns": len(trajectory.turns),
        "cost": total_cost,
        "trace": trace,
        "trajectory_dict": trajectory.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="PACE Context Eviction Experiment")
    parser.add_argument("--strategy", required=True, choices=["recap", "evict", "evict-always"])
    parser.add_argument("--task", default="math")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--sample-ids", nargs="*", default=None,
                        help="Specific task IDs to run (e.g. sharded-GSM8K/158)")
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.chdir(str(LIC_DIR))
    from model_openai import generate, generate_json

    dataset_path = LIC_DIR / "data" / "sharded_instructions_600.json"
    with open(dataset_path) as f:
        samples = json.load(f)
    samples = [s for s in samples if s["task"] == args.task]

    if args.sample_ids:
        samples = [s for s in samples if s["task_id"] in args.sample_ids]
    elif args.num_samples and args.num_samples < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, args.num_samples)

    if args.dry_run:
        print(f"Strategy: {args.strategy}")
        print(f"Task: {args.task}, Model: {args.model}")
        print(f"Samples: {len(samples)}")
        est_cost = len(samples) * 0.005
        print(f"Estimated cost: ~${est_cost:.2f}")
        return

    embedder = Embedder()
    contradiction_scorer = HFContradictionScorer() if not args.no_nli else None
    signal_computer = SignalComputer(
        embedder=embedder,
        contradiction_scorer=contradiction_scorer,
    )
    policy = InterventionPolicy(InterventionConfig(
        intervention_type=InterventionType.CONTEXT_EVICTION,
    ))

    print(f"Strategy: {args.strategy}")
    print(f"Running {len(samples)} samples with {args.model}")

    try:
        from tqdm import tqdm
        sample_iter = tqdm(samples, desc=f"Eviction ({args.strategy})")
    except ImportError:
        sample_iter = samples

    results = []
    for sample in sample_iter:
        result = run_eviction_experiment(
            sample, args.strategy, args.model,
            generate, generate_json,
            signal_computer, policy,
            verbose=args.verbose,
        )
        results.append(result)

    n_correct = sum(1 for r in results if r.get("is_correct"))
    total_evictions = sum(r.get("evictions", 0) for r in results)
    total_cost = sum(r.get("cost", 0) for r in results)
    n = len(results)

    print(f"\nStrategy: {args.strategy}")
    print(f"Accuracy: {n_correct}/{n} ({n_correct/max(n,1):.1%})")
    print(f"Total evictions: {total_evictions}")
    print(f"Total cost: ${total_cost:.4f}")

    output_dir = PROJECT_ROOT / "logs" / "context_eviction"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"{args.strategy}_{args.task}_{args.model}_{ts}.json"

    manifest = {
        "strategy": args.strategy,
        "task": args.task,
        "model": args.model,
        "num_samples": n,
        "accuracy": n_correct / max(n, 1),
        "total_evictions": total_evictions,
        "total_cost_usd": total_cost,
        "results": [{k: v for k, v in r.items() if k != "trace"} for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
