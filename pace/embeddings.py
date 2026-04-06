"""
Embedding and semantic similarity utilities.

Two models for two different jobs:
  - OpenAI text-embedding-3-small: fast vector similarity for S1 (answer
    convergence) and S2 (info gain). Measures topical closeness.
  - sentence-transformers cross-encoder (NLI-trained): calibrated semantic
    entailment score for S5 (inter-agent agreement). Distinguishes "same
    topic, different conclusion" from "same meaning."

The NLI model is loaded lazily and cached — it's ~80MB and runs on CPU
in <50ms per pair, so near-zero overhead per turn.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class Embedder:
    """OpenAI embedding wrapper with cache. Used for S1 and S2."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self._client = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def embed(self, text: str) -> np.ndarray:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        truncated = text[:32000]
        resp = self.client.embeddings.create(input=truncated, model=self.model)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        self._cache[key] = vec
        return vec

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        va, vb = self.embed(text_a), self.embed(text_b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    def cosine_distance(self, text_a: str, text_b: str) -> float:
        return 1.0 - self.cosine_similarity(text_a, text_b)


class NLIScorer:
    """
    Cross-encoder NLI model for inter-agent agreement (S5).

    Uses 'cross-encoder/nli-deberta-v3-small' by default — a DeBERTa
    model fine-tuned on STS-B (semantic textual similarity). Returns a score
    in [0, 1] where 1 = semantically identical.

    Why this instead of cosine embeddings:
      - Two answers about the same topic that reach OPPOSITE conclusions will
        have high embedding cosine (same topic space) but low NLI score
        (different meaning). This distinction is exactly what S5 needs.
      - Calibrated: 0.9 actually means "these say the same thing."
      - Runs on CPU in ~30ms per pair. No GPU needed.
    """

    # entailment/contradiction/neutral labels instead of a continuous score.
    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Semantic similarity score in [0, 1].
        Higher = answers mean the same thing.
        """
        if not text_a or not text_b:
            return 0.0
        # CrossEncoder.predict returns a float for regression models
        score = float(self.model.predict([(text_a[:1024], text_b[:1024])]))
        # STS-B model outputs 0-5 scale; normalize to 0-1
        return max(0.0, min(1.0, score / 5.0))
