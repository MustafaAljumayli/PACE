"""
FRAMES Benchmark Harness.

FRAMES (Factuality, Retrieval, And reasoning for Multi-hop Evaluation and Synthesis)
tests multi-hop retrieval and synthesis. Questions require combining information
from multiple sources.

Dataset: https://huggingface.co/datasets/google/frames-benchmark
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass
class FRAMESQuestion:
    question_id: str
    question: str
    answer: str
    reasoning_type: str  # e.g., "multi-hop", "comparison", "temporal"
    num_hops: int
    metadata: dict[str, Any]


class FRAMESBenchmark:
    """Load and evaluate against the FRAMES dataset."""

    def __init__(self, split: str = "test", max_questions: int | None = None):
        self.split = split
        self.max_questions = max_questions
        self._questions: list[FRAMESQuestion] | None = None

    def load(self) -> list[FRAMESQuestion]:
        """Load FRAMES questions from HuggingFace."""
        if self._questions is not None:
            return self._questions

        ds = load_dataset("google/frames-benchmark", split=self.split)
        questions = []
        for i, row in enumerate(ds):
            if self.max_questions and i >= self.max_questions:
                break
            questions.append(FRAMESQuestion(
                question_id=str(row.get("id", i)),
                question=row["Prompt"],
                answer=row["Answer"],
                reasoning_type=row.get("reasoning_types", "unknown"),
                num_hops=row.get("num_hops", 1),
                metadata=dict(row),
            ))
        self._questions = questions
        return questions

    def evaluate(self, prediction: str, gold: str) -> dict[str, Any]:
        """
        Evaluate a single prediction against gold answer.

        Returns dict with:
          - exact_match: bool
          - contains_match: bool (gold is substring of prediction)
          - normalized_match: bool (after lowercasing + stripping)
        """
        pred_norm = self._normalize(prediction)
        gold_norm = self._normalize(gold)

        exact = pred_norm == gold_norm
        contains = gold_norm in pred_norm
        # Fuzzy: check if all words in gold appear in prediction
        gold_words = set(gold_norm.split())
        pred_words = set(pred_norm.split())
        word_overlap = len(gold_words & pred_words) / max(len(gold_words), 1)

        return {
            "exact_match": exact,
            "contains_match": contains,
            "word_overlap": word_overlap,
            "score": 1.0 if exact else (0.8 if contains else word_overlap),
        }

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text


@dataclass
class FRAMESResult:
    """Result of running one FRAMES question."""
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    evaluation: dict[str, Any]
    num_turns_used: int
    total_tokens: int
    total_latency_ms: float
    pace_decisions: list[dict] | None = None
    trajectory_data: dict | None = None
    rewind_used: bool = False
    rewind_turn: int | None = None
