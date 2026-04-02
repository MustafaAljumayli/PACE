"""
Build per-turn signal dataset from saved trajectories (+ optional labels).

Inputs:
  - trajectories_1000.json
  - labeled_1000.json (optional)

Output:
  signal_dataset_1000.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from pace.signals import HFContradictionScorer, TextSignalExtractor


class STEmbedder:
    """SentenceTransformer-backed embedder exposing cosine methods."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, Any] = {}

    def _encode(self, text: str):
        if text not in self._cache:
            self._cache[text] = self.model.encode(text or "none")
        return self._cache[text]

    def cosine_similarity(self, a: str, b: str) -> float:
        import numpy as np

        va = self._encode(a)
        vb = self._encode(b)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    def cosine_distance(self, a: str, b: str) -> float:
        return 1.0 - self.cosine_similarity(a, b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build signal dataset from trajectories")
    parser.add_argument("--trajectories", default="data/trajectories_1000.json")
    parser.add_argument("--labels", default="data/labeled_1000.json")
    parser.add_argument("--output", default="data/signal_dataset_1000.json")
    parser.add_argument("--embedder-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-small")
    args = parser.parse_args()

    trajectories = json.loads(Path(args.trajectories).read_text())
    labels = {}
    if Path(args.labels).exists():
        labels = json.loads(Path(args.labels).read_text())

    embedder = STEmbedder(args.embedder_model)
    contradiction = HFContradictionScorer(model_name=args.nli_model)
    extractor = TextSignalExtractor(embedder=embedder, contradiction_scorer=contradiction)

    signal_dataset = {}
    task_ids = sorted(trajectories.keys())
    for idx, task_id in enumerate(task_ids):
        data = trajectories[task_id]
        traj = data.get("trajectory", [])
        answers = [t.get("answer", "") for t in traj]
        responses = [t.get("response", "") for t in traj]

        turns_with_signals = []
        for t in range(len(traj)):
            print(f"[{idx + 1}/{len(task_ids)}] Extracting {task_id} turn {t}")
            signals = extractor.compute_turn(answers, responses, t)
            signals["turn_number"] = t
            signals["turn_ratio"] = t / max(len(traj) - 1, 1)

            label = None
            correct = None
            if task_id in labels:
                labeled_turns = labels[task_id].get("labeled_turns", [])
                if t < len(labeled_turns):
                    label = labeled_turns[t].get("label")
                    correct = labeled_turns[t].get("correct")

            turns_with_signals.append(
                {
                    "turn": t,
                    "signals": signals,
                    "label": label,
                    "correct": correct,
                    "answer": answers[t],
                }
            )

        signal_dataset[task_id] = {
            "difficulty": data.get("difficulty"),
            "source": data.get("source"),
            "ground_truth": data.get("ground_truth"),
            "turns": turns_with_signals,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(signal_dataset, indent=2))
    print(f"Signal extraction complete: {out_path}")

    # Baseline health summary by source
    by_source = defaultdict(lambda: {"total": 0, "turns": [], "empty_answers_over_half": 0})
    for _, task in trajectories.items():
        src = task.get("source", "unknown")
        turns = task.get("trajectory", [])
        by_source[src]["total"] += 1
        by_source[src]["turns"].append(len(turns))
        empty = sum(1 for t in turns if not t.get("answer"))
        if empty > max(len(turns) // 2, 0):
            by_source[src]["empty_answers_over_half"] += 1

    print("\n=== TRAJECTORY HEALTH ===")
    for src, stats in sorted(by_source.items()):
        avg_turns = sum(stats["turns"]) / max(len(stats["turns"]), 1)
        print(f"\n[{src}]")
        print(f"  Tasks: {stats['total']}")
        print(f"  Avg turns: {avg_turns:.1f}")
        print(f"  Tasks with >50% empty answers: {stats['empty_answers_over_half']}")


if __name__ == "__main__":
    main()

