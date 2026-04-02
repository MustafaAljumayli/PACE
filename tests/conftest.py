"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import pytest

from pace.signals import SignalComputer
from pace.trajectory import TurnState


class MockEmbedder:
    def cosine_similarity(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        shared = sum(1 for ca, cb in zip(a, b) if ca == cb)
        return shared / max(len(a), len(b), 1)

    def cosine_distance(self, a: str, b: str) -> float:
        return 1.0 - self.cosine_similarity(a, b)


@pytest.fixture
def sc():
    return SignalComputer(embedder=MockEmbedder())


@pytest.fixture
def make_turn():
    def _make_turn(n, answer="ans", context="", tool="", cumul="", metadata=None):
        return TurnState(
            turn_number=n,
            answer=answer,
            retrieved_context=context,
            cumulative_context=cumul,
            tool_called=tool,
            metadata=metadata or {},
        )

    return _make_turn
