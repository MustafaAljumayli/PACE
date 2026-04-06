"""
Research Report Generator — auto-generates signal analysis reports after runs.

Output structure:
    reports/{task}/{model}/
        sharded_report.md
        full_report.md
        signal_analysis.md

Also generates:
    reports/summary.md       — cross-task/model overview
    logs/report.md           — legacy location (backward compat)

Usage:
    python experiments/report.py                          # generate all
    python experiments/report.py --task math --model gpt-4o-mini
    python run_pace_experiment.py report                  # via CLI
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_records(path: Path) -> list[dict]:
    """Parse JSONL or pretty-printed JSON (handles both LiC formats)."""
    content = path.read_text().strip()
    if not content:
        return []
    records: list[dict] = []
    if content.startswith("["):
        return json.loads(content)
    buf = ""
    depth = 0
    in_str = False
    esc = False
    for ch in content:
        buf += ch
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"' and not esc:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    records.append(json.loads(buf.strip()))
                except json.JSONDecodeError:
                    pass
                buf = ""
    return records


SIGNAL_NAMES = [
    "goal_drift", "shard_coverage", "contradiction",
    "response_stability", "token_entropy", "repetition",
]


def _signal_analysis(records: list[dict]) -> str:
    """Generate signal analysis comparing correct vs incorrect conversations."""
    n = len(records)
    if n == 0:
        return ""

    correct = [r for r in records if r.get("is_correct") or r.get("score", 0) == 1.0]
    incorrect = [r for r in records if not (r.get("is_correct") or r.get("score", 0) == 1.0)]
    n_c = len(correct)
    n_i = len(incorrect)

    lines = []
    lines.append(f"## Signal Analysis")
    lines.append(f"")
    lines.append(f"Samples: {n} total, {n_c} correct ({n_c/n:.1%}), {n_i} incorrect")
    lines.append(f"")

    # Signal capture rates
    has_entropy = sum(
        1 for r in records
        if any(t.get("token_entropy") is not None for t in r.get("pace_signals", {}).get("turns", []))
    )
    has_contradiction = sum(
        1 for r in records
        if any(t.get("contradiction", 0) > 0.01 for t in r.get("pace_signals", {}).get("turns", []))
    )
    lines.append(f"token_entropy captured: {has_entropy}/{n} ({has_entropy/n:.0%})")
    lines.append(f"contradiction > 0.01:  {has_contradiction}/{n} ({has_contradiction/n:.0%})")
    lines.append(f"")

    # Final-turn analysis
    lines.append(f"### Final-Turn Signal Values (correct vs incorrect)")
    lines.append(f"")
    lines.append(f"| Signal | Correct | Incorrect | Delta | Significance |")
    lines.append(f"|---|---|---|---|---|")

    for sig in SIGNAL_NAMES:
        cv = []
        iv = []
        for r in records:
            is_c = r.get("is_correct") or r.get("score", 0) == 1.0
            turns = r.get("pace_signals", {}).get("turns", [])
            if turns:
                v = turns[-1].get(sig)
                if v is not None:
                    (cv if is_c else iv).append(v)
        if cv and iv:
            cm = sum(cv) / len(cv)
            im = sum(iv) / len(iv)
            d = im - cm
            stars = "***" if abs(d) > 0.05 else "*" if abs(d) > 0.02 else ""
            direction = "↑ wrong" if d > 0 else "↓ wrong"
            lines.append(f"| {sig} | {cm:.4f} | {im:.4f} | {d:+.4f} | {stars} {direction} |")

    # Final-turn velocity
    lines.append(f"")
    lines.append(f"### Final-Turn Velocity (rate of change)")
    lines.append(f"")
    lines.append(f"| Signal | Correct | Incorrect | Delta | Significance |")
    lines.append(f"|---|---|---|---|---|")

    for sig in SIGNAL_NAMES:
        cv = []
        iv = []
        for r in records:
            is_c = r.get("is_correct") or r.get("score", 0) == 1.0
            turns = r.get("pace_signals", {}).get("turns", [])
            if turns:
                v = turns[-1].get(f"{sig}_v")
                if v is not None:
                    (cv if is_c else iv).append(v)
        if cv and iv:
            cm = sum(cv) / len(cv)
            im = sum(iv) / len(iv)
            d = im - cm
            stars = "***" if abs(d) > 0.05 else "*" if abs(d) > 0.02 else ""
            lines.append(f"| {sig}_v | {cm:+.4f} | {im:+.4f} | {d:+.4f} | {stars} |")

    # Turn count distribution
    lines.append(f"")
    lines.append(f"### Turn Count Distribution")
    lines.append(f"")
    ct = [len(r.get("pace_signals", {}).get("turns", [])) for r in correct]
    it = [len(r.get("pace_signals", {}).get("turns", [])) for r in incorrect]
    if ct:
        lines.append(f"- Correct: mean={sum(ct)/len(ct):.1f} turns")
    if it:
        lines.append(f"- Incorrect: mean={sum(it)/len(it):.1f} turns")

    return "\n".join(lines)


def _accuracy_table(records: list[dict], task: str, model: str, conv_type: str) -> str:
    n = len(records)
    if n == 0:
        return ""
    correct = sum(1 for r in records if r.get("is_correct") or r.get("score", 0) == 1.0)
    null_count = sum(1 for r in records if r.get("is_correct") is None)
    total_cost = sum(r.get("conv_cost_usd", 0) for r in records)

    lines = [f"## Results: {task} / {model} / {conv_type}"]
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Samples | {n} |")
    lines.append(f"| Correct | {correct} |")
    lines.append(f"| Accuracy | {correct/n:.1%} |")
    lines.append(f"| is_correct=null | {null_count} |")
    lines.append(f"| Total cost | ${total_cost:.4f} |")
    lines.append(f"| Avg cost/conv | ${total_cost/n:.4f} |")
    return "\n".join(lines)


def _latex_signal_table(records: list[dict]) -> str:
    """Generate LaTeX table for signal analysis."""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Final-Turn Signal Comparison: Correct vs Incorrect}",
        "\\label{tab:signals}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Signal & Correct & Incorrect & $\\Delta$ & Sig. \\\\",
        "\\midrule",
    ]
    for sig in SIGNAL_NAMES:
        cv = []
        iv = []
        for r in records:
            is_c = r.get("is_correct") or r.get("score", 0) == 1.0
            turns = r.get("pace_signals", {}).get("turns", [])
            if turns:
                v = turns[-1].get(sig)
                if v is not None:
                    (cv if is_c else iv).append(v)
        if cv and iv:
            cm = sum(cv) / len(cv)
            im = sum(iv) / len(iv)
            d = im - cm
            stars = "***" if abs(d) > 0.05 else "*" if abs(d) > 0.02 else ""
            sig_tex = sig.replace("_", "\\_")
            lines.append(f"{sig_tex} & {cm:.4f} & {im:.4f} & {d:+.4f} & {stars} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def _export_csv(records: list[dict], path: Path) -> None:
    """Export per-conversation signal summary as CSV for plotting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        turns = r.get("pace_signals", {}).get("turns", [])
        if not turns:
            continue
        last = turns[-1]
        row = {
            "task_id": r.get("task_id", ""),
            "is_correct": r.get("is_correct"),
            "score": r.get("score"),
            "num_turns": len(turns),
        }
        for sig in SIGNAL_NAMES:
            row[sig] = last.get(sig)
            row[f"{sig}_v"] = last.get(f"{sig}_v")
        rows.append(row)

    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def generate_run_report(
    records: list[dict],
    task: str,
    model: str,
    conv_type: str,
    output_dir: Path | None = None,
    latex: bool = False,
) -> Path:
    """
    Generate a report for a single run (task/model/conv_type combo).
    Returns path to the generated report.
    """
    if output_dir is None:
        sanitized = model.replace("/", "_").replace(":", "_")
        output_dir = PROJECT_ROOT / "reports" / task / sanitized

    output_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    sections.append(_accuracy_table(records, task, model, conv_type))

    if any(r.get("pace_signals") for r in records):
        sections.append("")
        sections.append(_signal_analysis(records))

    report = "\n\n".join(s for s in sections if s)
    report_path = output_dir / f"{conv_type}_report.md"
    report_path.write_text(report + "\n")

    _export_csv(records, output_dir / f"{conv_type}_signals.csv")

    if latex and any(r.get("pace_signals") for r in records):
        latex_path = output_dir / f"{conv_type}_signals.tex"
        latex_path.write_text(_latex_signal_table(records) + "\n")

    return report_path


def _discover_logs(log_folder: Path) -> dict[tuple[str, str, str], list[dict]]:
    """Discover and group all log files by (task, model, conv_type)."""
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    for f in sorted(log_folder.rglob("*.jsonl")):
        records = _parse_records(f)
        for rec in records:
            task = rec.get("task", "")
            model = rec.get("assistant_model", "")
            conv_type = rec.get("conv_type", "")
            if task and model:
                grouped[(task, model, conv_type)].append(rec)

    return dict(grouped)


def _load_pace_runs(log_folder: Path) -> list[dict]:
    runs_dir = log_folder / "pace_runs"
    if not runs_dir.exists():
        return []
    runs = []
    for f in sorted(runs_dir.glob("*.json")):
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def generate_full_report(log_folder: str, latex: bool = False) -> None:
    """
    Generate all reports from available logs.
    Creates per-run reports in reports/ and a summary in logs/report.md.
    """
    log_path = Path(log_folder)
    grouped = _discover_logs(log_path)
    runs = _load_pace_runs(log_path)

    generated = []
    for (task, model, conv_type), records in sorted(grouped.items()):
        if not records:
            continue
        path = generate_run_report(records, task, model, conv_type, latex=latex)
        generated.append((task, model, conv_type, len(records), path))
        print(f"  Report: {path}")

    # Summary report at legacy location
    summary_lines = []
    for task, model, conv_type, n, path in generated:
        recs = grouped[(task, model, conv_type)]
        correct = sum(1 for r in recs if r.get("is_correct") or r.get("score", 0) == 1.0)
        summary_lines.append(
            f"| {task} | {model} | {conv_type} | {correct}/{n} | {correct/n:.1%} |"
        )

    if summary_lines:
        summary = "## Experiment Summary\n\n"
        summary += "| Task | Model | Type | Correct | Accuracy |\n"
        summary += "|---|---|---|---|---|\n"
        summary += "\n".join(summary_lines)
    else:
        summary = ""

    # PACE runs table
    if runs:
        summary += "\n\n## PACE Experiment Runs\n\n"
        summary += "| Run ID | Task | Model | Intervention | N | Accuracy | Cost | Avg IV |\n"
        summary += "|---|---|---|---|---|---|---|---|\n"
        for run in runs:
            rid = run.get("run_id", "?")
            task = run.get("task", "?")
            model = run.get("model", "?")
            intervene = "ON" if run.get("intervene") else "OFF"
            n = run.get("num_samples_completed", 0)
            acc = run.get("accuracy", 0)
            cost = run.get("total_cost_usd", 0)
            avg_iv = run.get("mean_interventions", 0)
            summary += f"| {rid} | {task} | {model} | {intervene} | {n} | {acc:.1%} | ${cost:.4f} | {avg_iv:.1f} |\n"

    report_path = log_path / "report.md"
    report_path.write_text(summary + "\n")
    print(f"Report saved to {report_path}")

    # Also save to reports/summary.md
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "summary.md").write_text(summary + "\n")
    print(f"Summary saved to {reports_dir / 'summary.md'}")
