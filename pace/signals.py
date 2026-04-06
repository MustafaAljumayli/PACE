"""
SignalComputer: computes PACE trajectory signals at each turn.

Signals (6 total — each targets a specific LiC failure mode):
  goal_drift          — cos(response, goal): drift from the conversation goal
  shard_coverage      — min cos(response, shard_i): forgotten sub-goals
  contradiction       — max P_NLI(contradiction | response, past_j): long-range
  response_stability  — cos(response_t, response_{t-1}): abrupt reversals
  token_entropy       — mean per-token Shannon entropy (logprob path only)
  repetition          — windowed, length-normalised n-gram overlap: stuck loops

Each signal gets velocity (Δ) and acceleration (Δ²) computed generically.

The signal_mask controls which signals are active — this is how the
ablation study works. Set signal_mask={"goal_drift", "contradiction"}
to compute only those two.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable

from pace.embeddings import Embedder
from pace.trajectory import Trajectory, TurnState, SIGNAL_NAMES


class SignalComputer:
    """
    Compute PACE signals for the latest turn in-place.

    Args:
        embedder: OpenAI embedder for cosine signals.
        contradiction_scorer: callable(premise, hypothesis) -> float [0,1].
        signal_mask: set of signal names to compute. None = all.
        ngram_n: n-gram size for repetition signal.
        repetition_window: number of previous turns for windowed repetition.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        contradiction_scorer: Callable[[str, str], float] | None = None,
        signal_mask: set[str] | None = None,
        ngram_n: int = 3,
        repetition_window: int = 5,
    ):
        self.embedder = embedder or Embedder()
        self.contradiction_scorer = contradiction_scorer
        self.signal_mask = signal_mask
        self.ngram_n = ngram_n
        self.repetition_window = repetition_window

    def _active(self, signal_name: str) -> bool:
        if self.signal_mask is None:
            return True
        return signal_name in self.signal_mask

    def compute(self, trajectory: Trajectory) -> None:
        """Compute all active signals for the latest turn in-place."""
        if not trajectory.turns:
            return

        t = len(trajectory.turns) - 1
        current = trajectory.turns[t]

        if self._active("goal_drift"):
            current.goal_drift = self._compute_goal_drift(trajectory, t)

        if self._active("shard_coverage"):
            current.shard_coverage = self._compute_shard_coverage(trajectory, t)

        if self._active("contradiction"):
            current.contradiction = self._compute_contradiction(trajectory, t)

        if self._active("response_stability"):
            current.response_stability = self._compute_response_stability(trajectory, t)

        if self._active("token_entropy"):
            current.token_entropy = self._compute_token_entropy(current)

        if self._active("repetition"):
            current.repetition = self._compute_repetition(trajectory, t)

        self._compute_all_derivatives(trajectory, t)

    # ── Signal implementations ──

    def _compute_goal_drift(self, trajectory: Trajectory, t: int) -> float:
        """
        Cosine similarity between the current response and the original goal.
        High = on-track, low = drifted away from the original question.
        Anchored to turn 0, NOT to turn t-1.
        """
        current = trajectory.turns[t]
        goal = trajectory.goal
        if not goal:
            if trajectory.turns:
                goal = trajectory.turns[0].response
            else:
                return 0.0
        if not current.response:
            return 0.0
        return max(0.0, self.embedder.cosine_similarity(current.response, goal))

    def _compute_shard_coverage(self, trajectory: Trajectory, t: int) -> float:
        """
        How well the current response covers the revealed shards.
        Returns the MINIMUM cosine similarity between the response and each shard.
        Low = at least one shard has been forgotten / ignored.
        If no shards available, falls back to goal_drift.
        """
        current = trajectory.turns[t]
        if not trajectory.shards or not current.response:
            return self._compute_goal_drift(trajectory, t)

        sims = [
            self.embedder.cosine_similarity(current.response, shard)
            for shard in trajectory.shards
            if shard.strip()
        ]
        if not sims:
            return 0.0
        return min(sims)

    def _compute_contradiction(self, trajectory: Trajectory, t: int) -> float:
        """
        Long-range contradiction: max NLI contradiction probability between
        the current response and ALL previous responses (not just t-1).
        Uses HFContradictionScorer (NLI DeBERTa cross-encoder).
        """
        if self.contradiction_scorer is None:
            return 0.0
        current = trajectory.turns[t]
        if t == 0 or not current.response:
            return 0.0
        max_score = 0.0
        for prev_turn in trajectory.turns[:t]:
            if prev_turn.response:
                score = self.contradiction_scorer(current.response, prev_turn.response)
                max_score = max(max_score, score)
        return max_score

    def _compute_response_stability(self, trajectory: Trajectory, t: int) -> float:
        """
        Cosine similarity between consecutive responses.
        High = stable, low = abrupt shift in content.
        """
        if t == 0:
            return 1.0
        current = trajectory.turns[t]
        prev = trajectory.turns[t - 1]
        if not current.response or not prev.response:
            return 0.0
        return max(0.0, self.embedder.cosine_similarity(current.response, prev.response))

    def _compute_token_entropy(self, turn: TurnState) -> float | None:
        """
        Mean per-token Shannon entropy from logprobs.
        Returns None if logprobs are unavailable (e.g. Anthropic, reasoning models).
        The fallback lexical entropy path is intentionally removed —
        mixing two fundamentally different measures would confound analysis.
        """
        token_top_logprobs = turn.metadata.get("token_top_logprobs", [])
        if not token_top_logprobs:
            return None

        entropy_values: list[float] = []
        for token_logprobs in token_top_logprobs:
            if not token_logprobs:
                continue
            max_logp = max(token_logprobs)
            probs = [math.exp(lp - max_logp) for lp in token_logprobs]
            z = sum(probs)
            if z <= 0:
                continue
            probs = [p / z for p in probs]
            h_t = 0.0
            for p in probs:
                if p > 0:
                    h_t -= p * math.log2(p)
            entropy_values.append(h_t)

        if entropy_values:
            return sum(entropy_values) / len(entropy_values)
        return None

    def _compute_repetition(self, trajectory: Trajectory, t: int) -> float:
        """
        Windowed, length-normalised n-gram overlap.
        Compares the current response's n-grams against a WINDOW of recent
        previous responses (not the full history). This avoids the monotonic
        growth problem where any response will overlap with long histories.
        """
        current_response = trajectory.turns[t].response
        if not current_response:
            return 0.0
        current_ngrams = self._ngrams(current_response)
        if not current_ngrams:
            return 0.0

        window_start = max(0, t - self.repetition_window)
        history_ngrams: set[tuple[str, ...]] = set()
        for prev_turn in trajectory.turns[window_start:t]:
            if prev_turn.response:
                history_ngrams |= self._ngrams(prev_turn.response)

        if not history_ngrams:
            return 0.0

        overlap = len(current_ngrams & history_ngrams)
        return overlap / len(current_ngrams)

    def _ngrams(self, text: str) -> set[tuple[str, ...]]:
        words = re.findall(r"\b[\w']+\b", text.lower())
        n = self.ngram_n
        if n <= 0 or len(words) < n:
            return set()
        return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}

    # ── Derivatives ──

    def _compute_all_derivatives(self, trajectory: Trajectory, t: int) -> None:
        """Generic velocity and acceleration for every signal."""
        current = trajectory.turns[t]
        for sig in SIGNAL_NAMES:
            level = current.get_signal(sig)
            if level is None:
                continue
            if t >= 1:
                prev = trajectory.turns[t - 1]
                prev_level = prev.get_signal(sig)
                v = (level - prev_level) if prev_level is not None else 0.0
            else:
                v = 0.0
            setattr(current, f"{sig}_v", v)

            if t >= 2:
                prev = trajectory.turns[t - 1]
                prev_v = prev.get_velocity(sig)
                a = (v - prev_v) if prev_v is not None else 0.0
            else:
                a = 0.0
            setattr(current, f"{sig}_a", a)


class HFContradictionScorer:
    """
    HuggingFace NLI contradiction scorer (cross-encoder/nli-deberta-v3-small).
    Returns P(contradiction) in [0, 1] for a premise–hypothesis pair.
    Lazy-loaded — ~80MB, runs on CPU in <50ms per pair.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._contradiction_idx = 0

    def _lazy_load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        id2label = getattr(self._model.config, "id2label", {}) or {}
        for idx, label in id2label.items():
            if "contradiction" in str(label).lower():
                self._contradiction_idx = int(idx)
                break

    def __call__(self, premise: str, hypothesis: str) -> float:
        if not premise or not hypothesis:
            return 0.0
        self._lazy_load()
        import torch

        encoded = self._tokenizer(
            premise[:1024], hypothesis[:1024],
            return_tensors="pt", truncation=True, max_length=512,
        )
        with torch.no_grad():
            logits = self._model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1)
        return max(0.0, min(1.0, float(probs[self._contradiction_idx].item())))
