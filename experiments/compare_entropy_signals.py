"""
Compare unigram-vs-logprob entropy as stop-confidence signals.

This experiment runs selected models on FRAMES and computes two confidence
signals from each generated answer:
  1) unigram_entropy: lexical Shannon entropy over output tokens
  2) logprob_entropy: mean per-token Shannon entropy from top-k logprobs

It then evaluates how well each signal separates correct vs incorrect answers
and writes:
  - per-sample CSV
  - per-model summary JSON
  - comparison plot (AUC + Cohen's d)

Usage:
    python experiments/compare_entropy_signals.py --models gpt-4o,gpt-4o-mini
    python experiments/compare_entropy_signals.py --models gpt-4o --num-questions 20
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

np = None
pd = None
plt = None
roc_auc_score = None
FixedBudgetRunner = None
MultiAgentTeam = None
SingleReActAgent = None
FRAMESBenchmark = None
SignalComputer = None
get_frames_tools = None


def lexical_entropy(text: str) -> float:
    """Unigram Shannon entropy over lexical tokens."""
    if not text:
        return 0.0
    tokens = re.findall(r"\b[\w']+\b", text.lower())
    if len(tokens) < 2:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    h = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def logprob_entropy_from_metadata(metadata: dict) -> float | None:
    """Mean per-token entropy from token_top_logprobs metadata."""
    token_top_logprobs = metadata.get("token_top_logprobs", []) if metadata else []
    if not token_top_logprobs:
        return None

    entropies: list[float] = []
    for token_logprobs in token_top_logprobs:
        if not token_logprobs:
            continue
        arr = np.array(token_logprobs, dtype=float)
        arr = arr - np.max(arr)
        probs = np.exp(arr)
        denom = float(np.sum(probs))
        if denom <= 0:
            continue
        probs = probs / denom
        entropies.append(float(-np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0)))))

    if not entropies:
        return None
    return float(np.mean(entropies))


def cohens_d(correct_vals: np.ndarray, incorrect_vals: np.ndarray) -> float | None:
    """Effect size for entropy separation (incorrect - correct)."""
    if len(correct_vals) < 2 or len(incorrect_vals) < 2:
        return None
    var_c = np.var(correct_vals, ddof=1)
    var_i = np.var(incorrect_vals, ddof=1)
    pooled = ((len(correct_vals) - 1) * var_c + (len(incorrect_vals) - 1) * var_i) / (
        len(correct_vals) + len(incorrect_vals) - 2
    )
    if pooled <= 0:
        return None
    return float((np.mean(incorrect_vals) - np.mean(correct_vals)) / np.sqrt(pooled))


def compute_method_metrics(df: pd.DataFrame, entropy_col: str) -> dict:
    """Compute discrimination metrics for one entropy method."""
    subset = df.dropna(subset=[entropy_col]).copy()
    if subset.empty:
        return {
            "n": 0,
            "auc": None,
            "cohens_d": None,
            "correct_mean": None,
            "incorrect_mean": None,
        }

    y = subset["exact_match"].astype(int).to_numpy()
    e = subset[entropy_col].astype(float).to_numpy()
    correct = e[y == 1]
    incorrect = e[y == 0]

    auc = None
    if len(np.unique(y)) > 1:
        # Lower entropy => more confidence/correctness, so invert sign.
        auc = float(roc_auc_score(y, -e))

    return {
        "n": int(len(subset)),
        "auc": auc,
        "cohens_d": cohens_d(correct, incorrect),
        "correct_mean": float(np.mean(correct)) if len(correct) else None,
        "incorrect_mean": float(np.mean(incorrect)) if len(incorrect) else None,
    }


def build_agent(agent_type: str, model: str):
    tools = get_frames_tools()
    if agent_type == "single":
        return SingleReActAgent(model=model, max_turns=15, tools=tools)
    return MultiAgentTeam(model=model, max_turns=15, tools=tools)


def run_experiment(
    models: list[str],
    num_questions: int,
    turns: int,
    split: str,
    agent_type: str,
) -> pd.DataFrame:
    benchmark = FRAMESBenchmark(split=split, max_questions=num_questions)
    questions = benchmark.load()

    rows: list[dict] = []
    for model in models:
        print(f"\n=== Model: {model} | questions={len(questions)} | turns={turns} ===")
        agent = build_agent(agent_type=agent_type, model=model)
        runner = FixedBudgetRunner(agent=agent, num_turns=turns)
        # Avoid expensive S1/S2 embedding calls: only compute S3.
        sc = SignalComputer(signal_mask={"token_entropy"})

        for i, q in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {q.question[:70]}...")
            started = time.time()
            try:
                answer, traj = runner.run(query=q.question, signal_computer=sc, episode_id=q.question_id)
                eval_result = benchmark.evaluate(answer, q.answer)
                for turn in traj.turns:
                    row = {
                        "model": model,
                        "agent_type": agent_type,
                        "question_id": q.question_id,
                        "turn": turn.turn_number,
                        "exact_match": bool(eval_result.get("exact_match", False)),
                        "score": float(eval_result.get("score", 0.0)),
                        "token_count": int(turn.token_count),
                        "unigram_entropy": lexical_entropy(turn.answer),
                        "logprob_entropy": logprob_entropy_from_metadata(turn.metadata),
                        "logprob_available": bool(turn.metadata.get("token_top_logprobs")),
                    }
                    rows.append(row)
                elapsed = time.time() - started
                print(
                    f"    -> exact={eval_result.get('exact_match', False)} "
                    f"score={eval_result.get('score', 0.0):.2f} "
                    f"time={elapsed:.1f}s"
                )
            except Exception as exc:
                print(f"    -> ERROR: {exc}")
                rows.append(
                    {
                        "model": model,
                        "agent_type": agent_type,
                        "question_id": q.question_id,
                        "turn": 1,
                        "exact_match": False,
                        "score": 0.0,
                        "token_count": 0,
                        "unigram_entropy": None,
                        "logprob_entropy": None,
                        "logprob_available": False,
                        "error": str(exc),
                    }
                )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """Build per-model summary for both entropy methods."""
    out: dict[str, dict] = {}
    for model, g in df.groupby("model"):
        out[model] = {
            "n_rows": int(len(g)),
            "n_questions": int(g["question_id"].nunique()),
            "logprob_coverage": float(g["logprob_available"].mean()),
            "unigram": compute_method_metrics(g, "unigram_entropy"),
            "logprob": compute_method_metrics(g, "logprob_entropy"),
        }
    return out


def plot_summary(summary: dict, out_path: Path) -> None:
    """Plot AUC and Cohen's d per model for both methods."""
    models = list(summary.keys())
    x = np.arange(len(models))
    width = 0.35

    unigram_auc = [summary[m]["unigram"]["auc"] if summary[m]["unigram"]["auc"] is not None else np.nan for m in models]
    logprob_auc = [summary[m]["logprob"]["auc"] if summary[m]["logprob"]["auc"] is not None else np.nan for m in models]
    unigram_d = [summary[m]["unigram"]["cohens_d"] if summary[m]["unigram"]["cohens_d"] is not None else np.nan for m in models]
    logprob_d = [summary[m]["logprob"]["cohens_d"] if summary[m]["logprob"]["cohens_d"] is not None else np.nan for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].bar(x - width / 2, unigram_auc, width, label="Unigram entropy")
    axes[0].bar(x + width / 2, logprob_auc, width, label="Logprob entropy")
    axes[0].set_title("Correctness Discrimination (ROC-AUC)")
    axes[0].set_ylabel("AUC (higher is better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].axhline(0.5, linestyle="--", linewidth=1)
    axes[0].legend()

    axes[1].bar(x - width / 2, unigram_d, width, label="Unigram entropy")
    axes[1].bar(x + width / 2, logprob_d, width, label="Logprob entropy")
    axes[1].set_title("Separation Effect Size (Cohen's d)")
    axes[1].set_ylabel("d (incorrect - correct)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].legend()

    fig.suptitle("Entropy Signal Comparison for Stop-Confidence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare unigram vs logprob entropy signals")
    parser.add_argument("--models", required=True, help="Comma-separated models, e.g. gpt-4o,gpt-4o-mini")
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--turns", type=int, default=1, help="Fixed turns per question (default: 1)")
    parser.add_argument("--split", default="test", help="FRAMES split")
    parser.add_argument("--agent-type", choices=["single", "multi"], default="single")
    parser.add_argument("--output-dir", default="results/entropy_compare")
    args = parser.parse_args()

    global np, pd, plt, roc_auc_score
    try:
        import numpy as _np
        import pandas as _pd
        import matplotlib.pyplot as _plt
        from sklearn.metrics import roc_auc_score as _roc_auc_score
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "required package")
        raise SystemExit(
            f"Missing dependency '{missing}'. Install project deps with: "
            "pip install -e ."
        )

    np = _np
    pd = _pd
    plt = _plt
    roc_auc_score = _roc_auc_score

    global FixedBudgetRunner, MultiAgentTeam, SingleReActAgent
    global FRAMESBenchmark, SignalComputer, get_frames_tools
    try:
        from agents import FixedBudgetRunner as _FixedBudgetRunner
        from agents.multi_agentflow import MultiAgentTeam as _MultiAgentTeam
        from agents.single_react import SingleReActAgent as _SingleReActAgent
        from benchmarks.frames import FRAMESBenchmark as _FRAMESBenchmark
        from pace import SignalComputer as _SignalComputer
        from tools import get_frames_tools as _get_frames_tools
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "required package")
        raise SystemExit(
            f"Missing dependency '{missing}'. Install project deps with: "
            "pip install -e ."
        )

    FixedBudgetRunner = _FixedBudgetRunner
    MultiAgentTeam = _MultiAgentTeam
    SingleReActAgent = _SingleReActAgent
    FRAMESBenchmark = _FRAMESBenchmark
    SignalComputer = _SignalComputer
    get_frames_tools = _get_frames_tools

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    df = run_experiment(
        models=models,
        num_questions=args.num_questions,
        turns=args.turns,
        split=args.split,
        agent_type=args.agent_type,
    )

    csv_path = out_dir / f"samples_{args.agent_type}_{ts}.csv"
    df.to_csv(csv_path, index=False)

    summary = summarize(df)
    json_path = out_dir / f"summary_{args.agent_type}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot_path = out_dir / f"entropy_comparison_{args.agent_type}_{ts}.png"
    plot_summary(summary, plot_path)

    print("\nSaved outputs:")
    print(f"  - Samples CSV: {csv_path}")
    print(f"  - Summary JSON: {json_path}")
    print(f"  - Comparison plot: {plot_path}")

    print("\nQuick read:")
    for model, vals in summary.items():
        u_auc = vals["unigram"]["auc"]
        l_auc = vals["logprob"]["auc"]
        print(
            f"  {model}: "
            f"unigram_auc={u_auc if u_auc is not None else 'n/a'} | "
            f"logprob_auc={l_auc if l_auc is not None else 'n/a'} | "
            f"logprob_coverage={vals['logprob_coverage']:.2f}"
        )


if __name__ == "__main__":
    main()

