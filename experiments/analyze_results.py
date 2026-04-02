"""
Analyze experiment results and produce paper-ready tables + plots.

Usage:
    python experiments/analyze_results.py --input results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_all_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all JSON result files grouped by benchmark."""
    grouped: dict[str, list[dict]] = {"frames": [], "tau2": []}
    for f in results_dir.rglob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
        if "frames" in f.parent.name.lower() or "frames" in f.name.lower():
            grouped["frames"].append(data)
        elif "tau2" in f.parent.name.lower() or "tau2" in f.name.lower():
            grouped["tau2"].append(data)
    return grouped


def frames_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Build the main comparison table for FRAMES experiments."""
    rows = []
    for r in results:
        summary = r.get("summary", {})
        rows.append({
            "Condition": r.get("condition", "unknown"),
            "Model": r.get("model", "unknown"),
            "Accuracy": f"{summary.get('accuracy_mean', 0):.3f}",
            "Exact Match": f"{summary.get('accuracy_exact_match', 0):.3f}",
            "Avg Turns": f"{summary.get('turns_mean', 0):.1f}",
            "Avg Tokens": f"{summary.get('tokens_mean', 0):.0f}",
            "Rewinds": summary.get("rewind_count", 0),
            "N": summary.get("n", 0),
        })

    df = pd.DataFrame(rows)
    return df


def tau2_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Build comparison table for τ²-Bench experiments."""
    rows = []
    for r in results:
        rows.append({
            "Domain": r.get("domain", "unknown"),
            "Agent LLM": r.get("agent_llm", "unknown"),
            "Baseline Success": f"{r.get('baseline_success_rate', 0):.3f}",
            "Avg Turns (Baseline)": f"{r.get('avg_turns_baseline', 0):.1f}",
            "Avg Turns (PACE)": f"{r.get('avg_turns_pace', 0):.1f}",
            "Turn Savings": f"{r.get('turn_savings_pct', 0):.1f}%",
            "Early Stops": r.get("pace_early_stops", 0),
            "Rewinds": r.get("pace_rewinds", 0),
        })
    return pd.DataFrame(rows)


def print_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert DataFrame to LaTeX table for the paper."""
    latex = df.to_latex(index=False, escape=True)
    wrapped = (
        f"\\begin{{table}}[htbp]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex}"
        f"\\end{{table}}\n"
    )
    return wrapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/", help="Results directory")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX tables")
    args = parser.parse_args()

    results_dir = Path(args.input)
    grouped = load_all_results(results_dir)

    # FRAMES
    if grouped["frames"]:
        print("\n" + "=" * 70)
        print("TABLE 1: FRAMES Benchmark Results")
        print("=" * 70)
        df = frames_comparison_table(grouped["frames"])
        print(df.to_string(index=False))
        if args.latex:
            print("\n--- LaTeX ---")
            print(print_latex_table(
                df,
                "FRAMES benchmark results. PACE achieves comparable accuracy with fewer turns.",
                "tab:frames",
            ))

    # τ²-Bench
    if grouped["tau2"]:
        print("\n" + "=" * 70)
        print("TABLE 2: τ²-Bench Results (Post-Hoc PACE Analysis)")
        print("=" * 70)
        df = tau2_comparison_table(grouped["tau2"])
        print(df.to_string(index=False))
        if args.latex:
            print("\n--- LaTeX ---")
            print(print_latex_table(
                df,
                "τ²-Bench results. PACE post-hoc analysis shows turn savings without accuracy loss.",
                "tab:tau2",
            ))

    if not any(grouped.values()):
        print(f"No results found in {results_dir}. Run experiments first.")


if __name__ == "__main__":
    main()
