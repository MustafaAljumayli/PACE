"""
Run a resumable multi-turn baseline with an OpenAI-compatible chat endpoint.

Input:
  tasks_1000.json (list of task dicts from build_training_dataset.py)

Output:
  trajectories_1000.json (dict keyed by task id)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a helpful reasoning assistant.
For EVERY turn, output your current best answer FIRST on line 1 in exactly this format:
ANSWER: <your answer here>

Then you may add brief reasoning (optional). If unsure, still provide your best current guess.
Keep your ANSWER concise — one word, phrase, number, or letter only."""

CONTINUE_PROMPTS = [
    "Review your reasoning carefully. Line 1 must be: ANSWER: <best current answer>.",
    "Double-check your work. Start with ANSWER: <best current answer>.",
    "Reconsider the problem. Keep line 1 as ANSWER: <best current answer>.",
    "Verify your work. Begin with ANSWER: <best current answer>.",
    "Take another careful look. First line: ANSWER: <best current answer>.",
    "Challenge your reasoning. Still begin with ANSWER: <best current answer>.",
    "Review each step. Start with ANSWER: <best current answer>.",
]


def extract_answer(text: str) -> str:
    for line in reversed(text.strip().split("\n")):
        if line.strip().upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()
    return ""


def normalize_for_match(answer: str, source: str) -> str:
    """Normalize extracted answers for weak string-based correctness checks."""
    if not answer:
        return ""
    a = answer.strip()

    # GPQA is multiple choice; capture the first explicit option letter.
    if source == "gpqa_diamond":
        m = re.search(r"\b([A-D])\b", a.upper())
        return m.group(1) if m else a.upper()

    # Strip common formatting noise.
    a = re.sub(r"\\boxed\{(.+?)\}", r"\1", a)
    a = a.replace("$", "")
    a = a.replace(",", "")  # 1,000 -> 1000
    a = a.lower().strip()
    a = re.sub(r"[^\w\.\-\s/]", "", a)
    a = re.sub(r"\s+", " ", a)
    return a


def _join_base_url(base: str, suffix: str) -> str:
    return f"{base.rstrip('/')}/{suffix.lstrip('/')}"


def wait_for_openai_endpoint(api_base: str, timeout_s: float = 600.0, poll_s: float = 2.0) -> None:
    """Wait until an OpenAI-compatible endpoint responds on /models."""
    url = _join_base_url(api_base, "models")
    deadline = time.time() + timeout_s
    last_err: str | None = None

    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_err = str(exc)
        time.sleep(poll_s)

    raise RuntimeError(
        f"Timed out waiting for endpoint readiness at {url}. "
        f"Last error: {last_err}"
    )


def start_vllm_server(
    python_exec: str,
    model: str,
    host: str,
    port: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    extra_args: str,
) -> subprocess.Popen:
    """Start a vLLM OpenAI server subprocess."""
    cmd = [
        python_exec,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if extra_args.strip():
        cmd.extend(extra_args.strip().split())

    print("Starting vLLM server:")
    print(" ".join(cmd))
    return subprocess.Popen(cmd)


def run_task(
    client,
    task: dict[str, Any],
    model: str,
    max_turns: int,
    sleep_s: float,
    max_output_tokens: int | None,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["question"]},
    ]

    trajectory = []
    total_input_tokens = 0
    total_output_tokens = 0
    source = task.get("source", "")
    gold_norm = normalize_for_match(task.get("answer", ""), source)

    for turn in range(max_turns):
        create_kwargs = dict(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        if max_output_tokens is not None:
            create_kwargs["max_tokens"] = max_output_tokens
        if disable_thinking:
            create_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        response = client.chat.completions.create(**create_kwargs)

        response_text = response.choices[0].message.content or ""
        answer = extract_answer(response_text)
        answer_norm = normalize_for_match(answer, source)
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        trajectory.append(
            {
                "turn": turn,
                "response": response_text,
                "answer": answer,
                "answer_normalized": answer_norm,
                "weak_match_gold": bool(answer_norm and gold_norm and answer_norm == gold_norm),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

        messages.append({"role": "assistant", "content": response_text})
        if turn < max_turns - 1:
            messages.append(
                {
                    "role": "user",
                    "content": CONTINUE_PROMPTS[turn % len(CONTINUE_PROMPTS)],
                }
            )
        time.sleep(sleep_s)

    return {
        "id": task["id"],
        "question": task["question"],
        "ground_truth": task["answer"],
        "difficulty": task["difficulty"],
        "source": task["source"],
        "trajectory": trajectory,
        "ground_truth_normalized": gold_norm,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "timestamp": datetime.now().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline trajectories (Together or vLLM/OpenAI-compatible)")
    parser.add_argument("--tasks", default="data/tasks_mixed.json")
    parser.add_argument("--output", default="data/trajectories_1000.json")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument(
        "--backend",
        choices=["openai_compat", "together"],
        default="openai_compat",
        help="Use local vLLM/openai-compatible server or Together",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible API base (used when backend=openai_compat)",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for OpenAI-compatible endpoint (vLLM can use any non-empty string)",
    )
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help=(
            "Per-turn output token cap. "
            "Default: None (no explicit output cap; rely on model/provider limits)."
        ),
    )
    parser.add_argument("--sleep-s", type=float, default=0.5)
    parser.add_argument("--retry-s", type=float, default=2.0)
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Send provider-specific hint to disable explicit reasoning mode",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print aggregate progress every N tasks",
    )
    parser.add_argument(
        "--est-cost-per-token",
        type=float,
        default=0.0000008,
        help="USD estimate per token for progress reporting",
    )
    parser.add_argument(
        "--start-vllm",
        action="store_true",
        help="Start local vLLM server from this script (openai_compat backend only)",
    )
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument(
        "--vllm-python",
        default=sys.executable,
        help="Python executable used to launch vLLM (must have vllm installed)",
    )
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument(
        "--vllm-extra-args",
        default="",
        help="Extra args appended to vLLM command (space-separated)",
    )
    parser.add_argument(
        "--ready-timeout-s",
        type=float,
        default=900.0,
        help="How long to wait for endpoint readiness",
    )
    args = parser.parse_args()

    vllm_proc: subprocess.Popen | None = None
    if args.start_vllm:
        if args.backend != "openai_compat":
            raise SystemExit("--start-vllm requires --backend openai_compat")
        args.api_base = f"http://{args.vllm_host}:{args.vllm_port}/v1"
        try:
            wait_for_openai_endpoint(args.api_base, timeout_s=2.0, poll_s=0.5)
            print(f"Endpoint already active at {args.api_base}; reusing existing server.")
        except RuntimeError:
            vllm_proc = start_vllm_server(
                python_exec=args.vllm_python,
                model=args.model,
                host=args.vllm_host,
                port=args.vllm_port,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_len=args.vllm_max_model_len,
                extra_args=args.vllm_extra_args,
            )
            time.sleep(1.0)
            if vllm_proc.poll() is not None:
                raise SystemExit(
                    "vLLM process exited immediately. "
                    f"Check that vllm is installed in interpreter: {args.vllm_python}"
                )
            wait_for_openai_endpoint(args.api_base, timeout_s=args.ready_timeout_s, poll_s=2.0)
            print(f"vLLM ready at {args.api_base}")

    if args.backend == "together":
        if not os.getenv("TOGETHER_API_KEY"):
            raise SystemExit("TOGETHER_API_KEY is not set")
        try:
            from together import Together  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise SystemExit("Missing together package. Install with: pip install together") from exc
        client = Together()
    else:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("Missing openai package. Install with: pip install openai") from exc
        client = OpenAI(base_url=args.api_base, api_key=args.api_key)
        print(f"Using OpenAI-compatible endpoint: {args.api_base}")

    tasks_path = Path(args.tasks)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(tasks_path.read_text())
    if not isinstance(tasks, list):
        raise SystemExit(f"{tasks_path} must contain a JSON list of tasks")

    if output_path.exists():
        all_trajectories = json.loads(output_path.read_text())
        if not isinstance(all_trajectories, dict):
            raise SystemExit(f"{output_path} must contain a JSON object keyed by task id")
        completed_ids = set(all_trajectories.keys())
        print(f"Resuming from {len(completed_ids)} completed tasks")
    else:
        all_trajectories = {}
        completed_ids = set()

    remaining = [t for t in tasks if t.get("id") not in completed_ids]
    print(f"Running {len(remaining)} remaining tasks")

    try:
        errors = 0
        for i, task in enumerate(remaining):
            task_id = task.get("id", f"task_{i}")
            print(
                f"[{i + 1}/{len(remaining)}] {task_id} "
                f"[{task.get('difficulty', 'unknown')}] [{task.get('source', 'unknown')}]"
            )
            try:
                result = run_task(
                    client=client,
                    task=task,
                    model=args.model,
                    max_turns=args.max_turns,
                    sleep_s=args.sleep_s,
                    max_output_tokens=args.max_output_tokens,
                    disable_thinking=args.disable_thinking,
                )
                answers_found = sum(1 for t in result["trajectory"] if t.get("answer"))
                if answers_found == 0:
                    print(f"  WARNING: No ANSWER extracted for {task_id}")
                all_trajectories[task_id] = result
                output_path.write_text(json.dumps(all_trajectories))

                if (i + 1) % args.progress_every == 0:
                    total_tokens = sum(
                        int(v.get("total_input_tokens", 0)) + int(v.get("total_output_tokens", 0))
                        for v in all_trajectories.values()
                    )
                    est_cost = total_tokens * args.est_cost_per_token
                    print(
                        f"\n--- Progress: {len(all_trajectories)} tasks | "
                        f"tokens={total_tokens:,} | est_cost=${est_cost:.2f}\n"
                    )
            except Exception as exc:
                errors += 1
                print(f"Error task {task_id}: {exc}")
                time.sleep(args.retry_s)
                if errors >= args.max_errors:
                    print(f"Stopping after {errors} errors")
                    break

        print("Baseline trajectory run complete")
        print(f"Saved: {output_path}")
        total_tokens = sum(
            int(v.get("total_input_tokens", 0)) + int(v.get("total_output_tokens", 0))
            for v in all_trajectories.values()
        )
        est_cost = total_tokens * args.est_cost_per_token
        print(f"Total trajectories: {len(all_trajectories)}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Estimated cost: ${est_cost:.2f}")
    finally:
        if vllm_proc is not None:
            print("Stopping vLLM server started by this script...")
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()


if __name__ == "__main__":
    main()

