"""Tests for signal mask behavior in SignalComputer."""

from pace.signals import SignalComputer
from pace.trajectory import Trajectory


class TestSignalMask:
    def test_mask_only_computes_selected(self, make_turn):
        sc = SignalComputer(signal_mask={"token_entropy"})
        traj = Trajectory(query="test")
        traj.add_turn(make_turn(1, answer="hello world foo bar"))
        sc.compute(traj)
        t = traj.turns[0]
        assert t.token_entropy is not None
        assert t.answer_similarity is None
        assert t.info_gain is None

    def test_mask_s1_only(self, make_turn):
        sc = SignalComputer(signal_mask={"answer_similarity"})
        traj = Trajectory(query="test")
        traj.add_turn(make_turn(1, answer="aaa"))
        sc.compute(traj)
        traj.add_turn(make_turn(2, answer="aab"))
        sc.compute(traj)
        assert traj.turns[1].answer_similarity is not None
        assert traj.turns[1].token_entropy is None
