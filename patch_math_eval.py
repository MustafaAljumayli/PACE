#!/usr/bin/env python3
"""
Patch math evaluation: re-evaluate existing LiC logs with robust extraction.

Reads baseline JSONL logs, applies pace.extract.robust_math_eval to find
false negatives, and reports the corrected accuracy.

Usage:
    python patch_math_eval.py --log-path logs/math/sharded/sharded_math_gpt-4o-mini.jsonl
    python patch_math_eval.py --log-path logs/math/sharded/ --fix  # overwrite with corrected is_correct
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pace.extract import robust_math_eval, normalize_numeric


def load_records(path: Path) -> list[dict]:
    content = path.read_text().strip()
    if not content:
        return []
    records: list[dict] = []
    if content.startswith("["):
        return json.loads(content)
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


def patch_log(log_path: Path, fix: bool = False, verbose: bool = False) -> dict:
    records = load_records(log_path)
    if not records:
        print(f"  No records in {log_path}")
        return {"file": str(log_path), "n": 0, "recovered": 0}

    recovered = 0
    details: list[dict] = []

    for rec in records:
        if rec.get("is_correct") or rec.get("score", 0) == 1.0:
            continue

        trace = rec.get("trace", [])
        last_assistant = ""
        for msg in reversed(trace):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break
        if not last_assistant:
            continue

        gold = ""
        # Try to extract gold from evaluation log entry
        for msg in trace:
            if msg.get("role") == "log" and isinstance(msg.get("content"), dict):
                c = msg["content"]
                if c.get("type") == "answer-evaluation":
                    gold = str(c.get("exact_answer", ""))
                    break

        if not gold:
            continue

        result = robust_math_eval(last_assistant, gold)
        if result.get("is_correct"):
            recovered += 1
            task_id = rec.get("task_id", "?")
            method = result.get("extraction_method", "?")
            details.append({"task_id": task_id, "method": method, "gold": gold})

            if verbose:
                print(f"  RECOVERED: {task_id} (gold={gold}, method={method})")

            if fix:
                rec["is_correct"] = True
                rec["score"] = 1.0
                rec["pace_extraction_fix"] = {
                    "method": method,
                    "original_is_correct": False,
                }

    if fix and recovered > 0:
        with open(log_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    n = len(records)
    n_correct_original = sum(1 for r in records if r.get("is_correct") or r.get("score", 0) == 1.0)
    n_correct_after = n_correct_original

    return {
        "file": str(log_path),
        "n": n,
        "correct_original": n_correct_original - recovered if not fix else n_correct_original,
        "recovered": recovered,
        "correct_after": n_correct_original,
        "accuracy_after": n_correct_original / max(n, 1),
        "details": details,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Patch math evaluation with robust extraction")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--fix", action="store_true", help="Overwrite logs with corrected evaluations")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    path = Path(args.log_path)
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]

    total_recovered = 0
    for f in files:
        print(f"Processing: {f}")
        result = patch_log(f, fix=args.fix, verbose=args.verbose)
        total_recovered += result["recovered"]
        n = result["n"]
        if n > 0:
            print(f"  {n} records, {result['recovered']} recovered → {result['correct_after']}/{n} ({result['accuracy_after']:.1%})")

    if total_recovered > 0:
        action = "Fixed" if args.fix else "Would fix"
        print(f"\n{action} {total_recovered} false negatives across {len(files)} files")
    else:
        print("\nNo false negatives found.")


if __name__ == "__main__":
    main()
