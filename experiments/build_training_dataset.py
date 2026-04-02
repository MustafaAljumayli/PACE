"""
Build a mixed-difficulty training set for PACE.

Default composition (current plan):
  - HotpotQA bridge validation: 334 (easy)
  - MuSiQue:                  373 (medium)
  - GPQA Diamond:             193 (hard, full set)
  - OlymMATH hard:            up to 100 (hard)

Output:
  tasks_mixed.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv


load_dotenv()


def normalize_answer(answer: str, source: str) -> str:
    if answer is None:
        return ""
    answer = str(answer)

    if source == "gpqa_diamond":
        match = re.search(r"\b([A-D])\b", answer)
        return match.group(1).upper() if match else answer.strip().upper()

    if source in {"harp", "olympath"}:
        answer = re.sub(r"\\boxed\{(.+?)\}", r"\1", answer)
        answer = re.sub(r"\$", "", answer).strip()
        try:
            return str(float(Fraction(answer)))
        except Exception:
            return answer.lower().strip()

    if source in {"hotpotqa", "musique"}:
        return answer.strip().lower()

    return answer.strip().lower()


def _first_existing(row: dict[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _to_list(dataset_obj):
    if hasattr(dataset_obj, "to_list"):
        return dataset_obj.to_list()
    return list(dataset_obj)


def load_hotpot_bridge_validation() -> list[dict[str, Any]]:
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation")
    rows = _to_list(ds)
    out = []
    for i, row in enumerate(rows):
        qtype = str(row.get("type", "")).lower()
        if qtype != "bridge":
            continue
        question = _first_existing(row, ["question", "Question", "prompt"])
        answer = _first_existing(row, ["answer", "Answer", "final_answer"])
        if not question or answer is None:
            continue
        out.append(
            {
                "id": f"hotpotqa_{row.get('id', i)}",
                "question": str(question).strip(),
                "answer_raw": str(answer).strip(),
                "answer": normalize_answer(str(answer), "hotpotqa"),
                "source": "hotpotqa",
                "difficulty": "easy",
                "metadata": {
                    "split": "validation",
                    "type": row.get("type"),
                },
            }
        )
    return out


def load_musique(split: str) -> list[dict[str, Any]]:
    rows = _to_list(load_dataset("dgslibisey/MuSiQue", split=split))
    out = []
    for i, row in enumerate(rows):
        question = _first_existing(row, ["question", "Question", "prompt"])
        answer = _first_existing(row, ["answer", "answers", "Answer"])
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not question or answer is None:
            continue
        out.append(
            {
                "id": f"medium_{row.get('id', i)}",
                "question": str(question).strip(),
                "answer_raw": str(answer).strip(),
                "answer": normalize_answer(str(answer), "musique"),
                "source": "musique",
                "difficulty": "medium",
                "metadata": {"split": split},
            }
        )
    return out


def load_olymmath_hard(split: str, config: str) -> list[dict[str, Any]]:
    rows = _to_list(load_dataset("RUC-AIBox/OlymMATH", config, split=split))
    out = []
    for i, row in enumerate(rows):
        # Most configs are already difficulty-specific (e.g. "en-hard").
        # Keep this guard only when a level field exists in the row.
        level = str(row.get("level", "")).lower()
        if level and level != "hard":
            continue
        question = _first_existing(row, ["question", "problem", "prompt", "Question"])
        answer = _first_existing(row, ["answer", "final_answer", "solution", "Answer"])
        if not question or answer is None:
            continue
        out.append(
            {
                "id": f"hard_olym_{row.get('id', i)}",
                "question": str(question).strip(),
                "answer_raw": str(answer).strip(),
                "answer": normalize_answer(str(answer), "olympath"),
                "source": "olympath",
                "difficulty": "hard",
                "metadata": {"split": split, "config": config, "level": row.get("level")},
            }
        )
    return out


def load_gpqa_diamond(split: str | None = None) -> list[dict[str, Any]]:
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=split or "train")
    rows = _to_list(ds)
    out = []
    for i, row in enumerate(rows):
        question = _first_existing(row, ["question", "Question", "prompt"])
        answer = _first_existing(
            row,
            [
                "answer",
                "correct_answer",
                "Correct Answer",
                "correct",
                "gold",
                "label",
                "Answer",
            ],
        )
        if not question:
            continue
        if answer is None:
            # Attempt to infer from answer index and choices.
            ans_idx = _first_existing(row, ["answer_idx", "correct_option", "correct_choice"])
            if isinstance(ans_idx, int):
                answer = "ABCD"[ans_idx] if 0 <= ans_idx < 4 else ""
        if answer is None:
            continue

        out.append(
            {
                "id": f"gpqa_{row.get('id', i)}",
                "question": str(question).strip(),
                "answer_raw": str(answer).strip(),
                "answer": normalize_answer(str(answer), "gpqa_diamond"),
                "source": "gpqa_diamond",
                "difficulty": "hard",
                "metadata": {"split": split or "train"},
            }
        )
    return out


def sample_without_replacement(items: list[dict[str, Any]], k: int, rng: random.Random, name: str):
    if len(items) < k:
        raise ValueError(f"Not enough samples in {name}: requested={k}, available={len(items)}")
    return rng.sample(items, k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed training dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="tasks_mixed.json")
    parser.add_argument("--hotpot-count", type=int, default=334)
    parser.add_argument("--musique-count", type=int, default=373)
    parser.add_argument("--gpqa-count", type=int, default=193)
    parser.add_argument("--olym-count", type=int, default=100)
    parser.add_argument("--musique-split", default="train")
    parser.add_argument("--olym-config", default="en-hard")
    parser.add_argument("--olym-split", default="test")
    parser.add_argument("--gpqa-split", default="train")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    hotpot = load_hotpot_bridge_validation()
    musique = load_musique(args.musique_split)
    olym = load_olymmath_hard(args.olym_split, args.olym_config)
    gpqa = load_gpqa_diamond(args.gpqa_split)

    print(f"Available HotpotQA bridge validation: {len(hotpot)}")
    print(f"Available MuSiQue ({args.musique_split}): {len(musique)}")
    print(f"Available OlymMATH {args.olym_config} ({args.olym_split}): {len(olym)}")
    print(f"Available GPQA Diamond ({args.gpqa_split}): {len(gpqa)}")

    tasks = []
    tasks.extend(sample_without_replacement(hotpot, args.hotpot_count, rng, "HotpotQA"))
    tasks.extend(sample_without_replacement(musique, args.musique_count, rng, "MuSiQue"))
    tasks.extend(sample_without_replacement(gpqa, args.gpqa_count, rng, "GPQA Diamond"))
    olym_k = min(args.olym_count, len(olym))
    tasks.extend(sample_without_replacement(olym, olym_k, rng, "OlymMATH hard"))

    rng.shuffle(tasks)

    # Ensure unique ids after mixing.
    seen = set()
    deduped = []
    for t in tasks:
        if t["id"] in seen:
            continue
        seen.add(t["id"])
        deduped.append(t)
    if len(deduped) != len(tasks):
        print(f"Deduplicated {len(tasks) - len(deduped)} id collisions")
        tasks = deduped

    out_path = Path(args.out)
    out_path.write_text(json.dumps(tasks, indent=2))

    by_source = {}
    for t in tasks:
        by_source[t["source"]] = by_source.get(t["source"], 0) + 1

    print(f"\nSaved {len(tasks)} tasks to {out_path}")
    print(f"By source: {by_source}")


if __name__ == "__main__":
    main()

