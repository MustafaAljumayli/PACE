"""
SignalComputer: computes PACE trajectory signals at each turn.

Signals (5 total):
  S1: answer_similarity  — cosine sim between consecutive answers
  S2: info_gain          — cosine distance of new context vs. cumulative
  S3: token_entropy      — sequence entropy from token logprobs (fallback: text entropy)
  S4: tool_entropy       — Shannon entropy of tool selection distribution
  S5: agent_agreement    — NOT computed here; set by DualAgentRunner via NLIScorer

Each signal gets velocity (Δ) and acceleration (Δ²) computed generically.

The signal_mask controls which signals are active — this is how the
ablation study works. Set signal_mask={"answer_similarity", "info_gain"}
to compute only those two.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

from pace.embeddings import Embedder
from pace.trajectory import Trajectory, TurnState, SIGNAL_NAMES


class SignalComputer:
    """
    Compute PACE signals for the latest turn in a trajectory.

    Args:
        embedder: OpenAI embedder for S1/S2.
        signal_mask: set of signal names to compute. None = all.
            Used for ablation: SignalComputer(signal_mask={"answer_similarity"})
            computes only S1 and its derivatives.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        signal_mask: set[str] | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.signal_mask = signal_mask  # None = compute all

    def _active(self, signal_name: str) -> bool:
        """Check if this signal is in the active mask."""
        if self.signal_mask is None:
            return True
        return signal_name in self.signal_mask

    def compute(self, trajectory: Trajectory) -> None:
        """Compute all active signals for the latest turn in-place."""
        if not trajectory.turns:
            return

        t = len(trajectory.turns) - 1
        current = trajectory.turns[t]

        # ── S1: Answer Cosine Similarity ──
        if self._active("answer_similarity"):
            if t == 0:
                current.answer_similarity = 0.0
            else:
                prev = trajectory.turns[t - 1]
                current.answer_similarity = self.embedder.cosine_similarity(
                    current.answer, prev.answer
                )

        # ── S2: Information Gain ──
        if self._active("info_gain"):
            if not current.retrieved_context:
                current.info_gain = 0.0
            elif not current.cumulative_context:
                current.info_gain = 1.0
            else:
                current.info_gain = self.embedder.cosine_distance(
                    current.retrieved_context, current.cumulative_context
                )

        # ── S3: Token Entropy (token logprobs when available) ──
        if self._active("token_entropy"):
            current.token_entropy = self._token_entropy(current)

        # ── S4: Tool Selection Entropy ──
        if self._active("tool_entropy"):
            current.tool_entropy = self._tool_entropy(trajectory, t)

        # ── S5: agent_agreement is set externally by DualAgentRunner ──
        # We still compute its derivatives if the value was set.

        # ── Derivatives for ALL signals (including S5 if present) ──
        self._compute_all_derivatives(trajectory, t)

    def _token_entropy(self, turn: TurnState) -> float:
        """
        Sequence-level entropy signal in bits.
        This is good for epistemic confidence. AKA uncertainty about the answer.
        More uncertainty means more uncertainty about the answer.
        Less uncertainty means less uncertainty about the answer.

        Preferred path (paper-aligned):
          - Use per-token top-k logprobs from turn.metadata["token_top_logprobs"]
          - Compute per-token Shannon entropy over normalized probabilities
          - Return mean entropy across generated tokens

        Fallback path (for providers without logprobs):
          - Lexical unigram entropy of output text
        """
        token_top_logprobs = turn.metadata.get("token_top_logprobs", [])
        if token_top_logprobs:
            entropy_values: list[float] = []
            for token_logprobs in token_top_logprobs:
                if not token_logprobs:
                    continue
                # Stable softmax over logprobs.
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

        return self._lexical_entropy(turn.answer)

    def _lexical_entropy(self, text: str) -> float:
        """
        Fallback lexical entropy when token logprobs are unavailable.
        This is good for aleatoric confidence A.K.A. variation in textual frequency.
        More variation in frequency means more uncertainty about the textual content of the answer.
        Less variation in frequency means more certainty about the textual content of the answer.
        """
        if not text:
            return 0.0
        # Simple lexical tokenization is more stable than whitespace splitting:
        # punctuation-only chunks are excluded and values become less brittle.
        tokens = re.findall(r"\b[\w']+\b", text.lower())
        if len(tokens) < 2:
            return 0.0
        counts = Counter(tokens)
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _tool_entropy(self, trajectory: Trajectory, t: int) -> float:
        """Shannon entropy of tool selection distribution up to turn t."""
        tools = [
            turn.tool_called
            for turn in trajectory.turns[: t + 1]
            if turn.tool_called
        ]
        if not tools:
            return 0.0
        counts = Counter(tools)
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy


    def _compute_all_derivatives(self, trajectory: Trajectory, t: int) -> None:
        """
        Generic derivative computation for every signal.

        For each signal s in SIGNAL_NAMES:
          velocity:     s_v(t)  = s(t) - s(t-1)
          acceleration: s_a(t)  = s_v(t) - s_v(t-1)
        """
        current = trajectory.turns[t]

        for sig in SIGNAL_NAMES:
            level = current.get_signal(sig)
            if level is None:
                continue

            # Velocity
            if t >= 1:
                prev = trajectory.turns[t - 1]
                prev_level = prev.get_signal(sig)
                if prev_level is not None:
                    v = level - prev_level
                else:
                    v = 0.0
            else:
                v = 0.0
            setattr(current, f"{sig}_v", v)

            # Acceleration
            if t >= 2:
                prev = trajectory.turns[t - 1]
                prev_v = prev.get_velocity(sig)
                if prev_v is not None:
                    a = v - prev_v
                else:
                    a = 0.0
            else:
                a = 0.0
            setattr(current, f"{sig}_a", a)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _tokenize_words(text: str) -> list[str]:
    return [w for w in text.lower().split() if w]


def _slope(y_values: list[float]) -> float:
    """Simple least-squares slope for x = 0..n-1."""
    n = len(y_values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(y_values) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(y_values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


@dataclass
class TextSignalConfig:
    """Configuration for text-level RL signals."""

    window_k: int = 3
    ngram_n: int = 3
    hedge_words: set[str] | None = None
    # UQS initial weights
    w_as: float = 0.2
    w_cs: float = 0.2
    w_ct: float = 0.2
    w_sd: float = 0.2
    w_rr: float = 0.2
    # Dynamic budget terms
    base_budget: float = 3.0
    alpha: float = 1.0
    beta_avr: float = 1.0
    beta_ct: float = 1.0
    beta_cs: float = 1.0

    def __post_init__(self) -> None:
        if self.hedge_words is None:
            self.hedge_words = {
                "maybe",
                "might",
                "possibly",
                "perhaps",
                "likely",
                "unlikely",
                "seems",
                "appears",
                "could",
                "unclear",
                "unsure",
                "probably",
            }


class TextSignalExtractor:
    """
    Extract text-level PACE-RL signals from turn-wise answers/responses.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        contradiction_scorer: Callable[[str, str], float] | None = None,
        config: TextSignalConfig | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.config = config or TextSignalConfig()
        self.contradiction_scorer = contradiction_scorer or self._default_contradiction_scorer

    def compute_all(self, answers: list[str], responses: list[str] | None = None) -> list[dict[str, float]]:
        """
        Compute signal vectors for every turn index.

        Args:
            answers: canonical answer per turn.
            responses: optional full response text per turn.
                       If omitted, answers are used as responses.
        """
        responses = responses or answers
        if len(responses) != len(answers):
            raise ValueError("answers and responses must have identical length")
        return [self.compute_turn(answers, responses, i) for i in range(len(answers))]

    def compute_turn(self, answers: list[str], responses: list[str], t: int) -> dict[str, float]:
        if t < 0 or t >= len(answers):
            raise IndexError("turn index out of range")

        as_t = self._answer_stability(answers, t)
        cs_t = self._contradiction_score(answers, t)
        ct_t = self._confidence_trajectory(responses, t)
        sd_t = self._semantic_divergence(answers, t)
        rr_t = self._repetition_ratio(responses, t)
        avr_t = self._answer_volatility_rate(answers, t)
        rcld_t = self._reasoning_chain_length_delta(responses, t)

        uqs = (
            self.config.w_as * as_t
            + self.config.w_cs * (1.0 - cs_t)
            + self.config.w_ct * ct_t
            + self.config.w_sd * (1.0 - sd_t)
            + self.config.w_rr * (1.0 - rr_t)
        )
        difficulty = (
            self.config.beta_avr * avr_t
            + self.config.beta_ct * (1.0 - ct_t)
            + self.config.beta_cs * cs_t
        )
        dynamic_budget = self.config.base_budget + self.config.alpha * difficulty

        return {
            "answer_stability": as_t,
            "contradiction_score": cs_t,
            "confidence_trajectory": ct_t,
            "semantic_divergence": sd_t,
            "repetition_ratio": rr_t,
            "answer_volatility_rate": avr_t,
            "reasoning_chain_length_delta": rcld_t,
            "uqs": uqs,
            "difficulty": difficulty,
            "dynamic_budget": dynamic_budget,
        }

    def _answer_stability(self, answers: list[str], t: int) -> float:
        if t == 0:
            return 0.0
        return _clamp01(self.embedder.cosine_similarity(answers[t], answers[t - 1]))

    def _contradiction_score(self, answers: list[str], t: int) -> float:
        if t == 0:
            return 0.0
        return _clamp01(self.contradiction_scorer(answers[t], answers[t - 1]))

    def _confidence_trajectory(self, responses: list[str], t: int) -> float:
        start = max(0, t - self.config.window_k + 1)
        window = responses[start : t + 1]
        confidence_series = [1.0 - self._hedge_density(r) for r in window]
        return _clamp01(0.5 + 0.5 * _slope(confidence_series))

    def _semantic_divergence(self, answers: list[str], t: int) -> float:
        start = max(0, t - self.config.window_k + 1)
        window = answers[start : t + 1]
        if len(window) < 2:
            return 0.0
        distances = [
            self.embedder.cosine_distance(a, b)
            for a, b in combinations(window, 2)
        ]
        if not distances:
            return 0.0
        return _clamp01(sum(distances) / len(distances))

    def _repetition_ratio(self, responses: list[str], t: int) -> float:
        current = self._ngrams(responses[t], self.config.ngram_n)
        if not current:
            return 0.0

        history = set()
        for prev in responses[:t]:
            history |= self._ngrams(prev, self.config.ngram_n)
        if not history:
            return 0.0
        return _clamp01(len(current & history) / len(current))

    def _answer_volatility_rate(self, answers: list[str], t: int) -> float:
        if t == 0:
            return 0.0
        changes = [1.0 - self._answer_stability(answers, i) for i in range(1, t + 1)]
        return _clamp01(sum(changes) / len(changes))

    def _reasoning_chain_length_delta(self, responses: list[str], t: int) -> float:
        if t == 0:
            return 0.0
        return float(len(_tokenize_words(responses[t])) - len(_tokenize_words(responses[t - 1])))

    def _hedge_density(self, text: str) -> float:
        words = _tokenize_words(text)
        if not words:
            return 0.0
        hedge_hits = sum(1 for w in words if w in self.config.hedge_words)
        return hedge_hits / len(words)

    def _ngrams(self, text: str, n: int) -> set[tuple[str, ...]]:
        words = _tokenize_words(text)
        if n <= 0 or len(words) < n:
            return set()
        return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}

    def _default_contradiction_scorer(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        return 0.0


class HFContradictionScorer:
    """
    HuggingFace contradiction scorer using NLI classification models.

    Default model aligns with the study design:
      cross-encoder/nli-deberta-v3-small
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
            premise[:1024],
            hypothesis[:1024],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = self._model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1)
        return _clamp01(float(probs[self._contradiction_idx].item()))
