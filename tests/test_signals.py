"""Tests for signal computation and derivatives."""

import math

from pace.trajectory import Trajectory, SIGNAL_NAMES


class TestSignalComputation:
    def test_all_five_signal_names(self):
        assert len(SIGNAL_NAMES) == 5
        assert "agent_agreement" in SIGNAL_NAMES

    def test_first_turn_defaults(self, sc, make_turn):
        traj = Trajectory(query="test")
        traj.add_turn(make_turn(1))
        sc.compute(traj)
        t = traj.turns[0]
        assert t.answer_similarity == 0.0
        assert t.answer_similarity_v == 0.0
        assert t.answer_similarity_a == 0.0

    def test_identical_answers_converge(self, sc, make_turn):
        traj = Trajectory(query="test")
        traj.add_turn(make_turn(1, "The answer is 42"))
        sc.compute(traj)
        traj.add_turn(make_turn(2, "I suspect that the answer may be 42"))
        sc.compute(traj)
        assert traj.turns[1].answer_similarity == 1.0

    def test_token_entropy_non_negative(self, sc, make_turn):
        traj = Trajectory(query="test")
        t = make_turn(1, answer="the the the cat cat dog")
        traj.add_turn(t)
        sc.compute(traj)
        assert t.token_entropy >= 0.0

    def test_token_entropy_uniform_is_high(self, sc, make_turn):
        traj = Trajectory(query="test")
        t = make_turn(1, answer="alpha bravo charlie delta echo foxtrot")
        traj.add_turn(t)
        sc.compute(traj)
        assert t.token_entropy > 2.0

    def test_token_entropy_repetitive_is_low(self, sc, make_turn):
        traj = Trajectory(query="test")
        t = make_turn(1, answer="the the the the the the")
        traj.add_turn(t)
        sc.compute(traj)
        assert t.token_entropy < 0.01

    def test_token_entropy_uses_logprobs_when_present(self, sc, make_turn):
        traj = Trajectory(query="test")
        t = make_turn(
            1,
            answer="irrelevant for this test",
            metadata={"token_top_logprobs": [[-0.6931, -0.6931], [-0.6931, -0.6931]]},
        )
        traj.add_turn(t)
        sc.compute(traj)
        assert 0.95 <= t.token_entropy <= 1.05

    def test_token_entropy_logprobs_override_text_fallback(self, sc, make_turn):
        traj = Trajectory(query="test")
        t = make_turn(
            1,
            answer="the the the the the the",
            metadata={"token_top_logprobs": [[-0.6931, -0.6931]]},
        )
        traj.add_turn(t)
        sc.compute(traj)
        assert t.token_entropy > 0.9

    def test_tool_entropy_single_tool(self, sc, make_turn):
        traj = Trajectory(query="test")
        for i in range(3):
            traj.add_turn(make_turn(i + 1, tool="search"))
            sc.compute(traj)
        assert traj.turns[-1].tool_entropy == 0.0

    def test_tool_entropy_diverse(self, sc, make_turn):
        traj = Trajectory(query="test")
        for i, tool in enumerate(["search", "wiki", "python"]):
            traj.add_turn(make_turn(i + 1, tool=tool))
            sc.compute(traj)
        assert abs(traj.turns[-1].tool_entropy - math.log2(3)) < 0.01

    def test_derivatives_computed_for_all_signals(self, sc, make_turn):
        traj = Trajectory(query="test")
        for i in range(4):
            t = make_turn(i + 1, answer=f"answer v{i}", tool="search")
            traj.add_turn(t)
            sc.compute(traj)

        last = traj.turns[-1]
        for sig in ["answer_similarity", "info_gain", "token_entropy", "tool_entropy"]:
            assert last.get_velocity(sig) is not None, f"{sig} velocity missing"
            assert last.get_acceleration(sig) is not None, f"{sig} acceleration missing"
