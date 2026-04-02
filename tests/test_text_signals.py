"""Tests for text-level RL signal extractor (S1-S7 + UQS)."""

from pace.signals import TextSignalConfig, TextSignalExtractor


class WordSetEmbedder:
    """Deterministic lexical overlap embedder for tests."""

    def cosine_similarity(self, a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / (len(sa) * len(sb)) ** 0.5

    def cosine_distance(self, a: str, b: str) -> float:
        return 1.0 - self.cosine_similarity(a, b)


def fake_contradiction(a: str, b: str) -> float:
    a_l, b_l = a.lower(), b.lower()
    if ("yes" in a_l and "no" in b_l) or ("no" in a_l and "yes" in b_l):
        return 0.95
    return 0.05


def test_compute_all_returns_one_vector_per_turn():
    ext = TextSignalExtractor(
        embedder=WordSetEmbedder(),
        contradiction_scorer=fake_contradiction,
    )
    answers = ["yes the value is 4", "no the value is 5", "no the value is 5"]
    out = ext.compute_all(answers)
    assert len(out) == len(answers)


def test_answer_stability_high_when_answers_repeat():
    ext = TextSignalExtractor(
        embedder=WordSetEmbedder(),
        contradiction_scorer=fake_contradiction,
    )
    answers = ["final answer is 42", "final answer is 42"]
    row = ext.compute_turn(answers, answers, 1)
    assert row["answer_stability"] > 0.99


def test_contradiction_score_spikes_for_yes_no_flip():
    ext = TextSignalExtractor(
        embedder=WordSetEmbedder(),
        contradiction_scorer=fake_contradiction,
    )
    answers = ["yes it is true", "no it is true"]
    row = ext.compute_turn(answers, answers, 1)
    assert row["contradiction_score"] > 0.9


def test_repetition_ratio_increases_when_response_repeats():
    ext = TextSignalExtractor(
        embedder=WordSetEmbedder(),
        contradiction_scorer=fake_contradiction,
        config=TextSignalConfig(ngram_n=2),
    )
    responses = [
        "we should test with two tools",
        "we should test with two tools",
    ]
    row = ext.compute_turn(responses, responses, 1)
    assert row["repetition_ratio"] > 0.9


def test_reasoning_chain_length_delta_tracks_length_change():
    ext = TextSignalExtractor(
        embedder=WordSetEmbedder(),
        contradiction_scorer=fake_contradiction,
    )
    responses = [
        "short answer",
        "this is a much longer and more detailed answer",
    ]
    row = ext.compute_turn(responses, responses, 1)
    assert row["reasoning_chain_length_delta"] > 0


def test_uqs_and_budget_fields_exist_and_are_finite():
    ext = TextSignalExtractor(
        embedder=WordSetEmbedder(),
        contradiction_scorer=fake_contradiction,
    )
    answers = ["maybe it is 4", "likely it is 5", "it is 5"]
    rows = ext.compute_all(answers, answers)
    last = rows[-1]
    assert isinstance(last["uqs"], float)
    assert isinstance(last["difficulty"], float)
    assert isinstance(last["dynamic_budget"], float)
