"""Tests for PACE policy decisions."""

from pace.policy import Decision, PACEPolicy, PolicyConfig
from pace.trajectory import Trajectory


class TestPolicy:
    def test_continue_below_min(self, sc, make_turn):
        policy = PACEPolicy(PolicyConfig(min_turns=3))
        traj = Trajectory(query="test")
        traj.add_turn(make_turn(1))
        sc.compute(traj)
        assert policy.decide(traj).decision == Decision.CONTINUE

    def test_stop_at_max(self, make_turn):
        policy = PACEPolicy(PolicyConfig(max_turns=3, min_turns=1))
        traj = Trajectory(query="test")
        for i in range(3):
            t = make_turn(i + 1, answer=f"diff {i}")
            t.answer_similarity = 0.5
            t.info_gain = 0.3
            traj.add_turn(t)
        assert policy.decide(traj).decision == Decision.STOP

    def test_convergence_with_multiple_signals(self, make_turn):
        policy = PACEPolicy(
            PolicyConfig(
                similarity_threshold=0.95,
                convergence_window=2,
                info_gain_floor=0.05,
                token_entropy_ceiling=0.3,
                relative_thresholds=False,
                min_turns=2,
                signal_mask={"answer_similarity", "info_gain", "token_entropy"},
            )
        )
        traj = Trajectory(query="test")
        for i in range(3):
            t = make_turn(i + 1)
            t.answer_similarity = 0.98
            t.info_gain = 0.02
            t.token_entropy = 0.15
            traj.add_turn(t)
        assert policy.decide(traj).decision == Decision.STOP

    def test_no_convergence_if_one_signal_disagrees(self, make_turn):
        policy = PACEPolicy(
            PolicyConfig(
                similarity_threshold=0.95,
                token_entropy_ceiling=0.3,
                relative_thresholds=False,
                min_turns=2,
                signal_mask={"answer_similarity", "token_entropy"},
            )
        )
        traj = Trajectory(query="test")
        for i in range(3):
            t = make_turn(i + 1)
            t.answer_similarity = 0.98
            t.token_entropy = 0.8
            traj.add_turn(t)
        assert policy.decide(traj).decision == Decision.CONTINUE

    def test_rewind_on_degradation(self, make_turn):
        policy = PACEPolicy(
            PolicyConfig(
                min_turns=2,
                signal_mask={"answer_similarity"},
            )
        )
        traj = Trajectory(query="test")
        t1 = make_turn(1)
        t1.answer_similarity = 0.0
        traj.add_turn(t1)
        t2 = make_turn(2)
        t2.answer_similarity = 0.97
        traj.add_turn(t2)
        t3 = make_turn(3)
        t3.answer_similarity = 0.3
        traj.add_turn(t3)
        result = policy.decide(traj)
        assert result.decision == Decision.REWIND
        assert result.rewind_to_turn == 2

    def test_scale_on_confident_conflict(self, make_turn):
        policy = PACEPolicy(
            PolicyConfig(
                min_turns=2,
                token_entropy_ceiling=0.3,
                signal_mask={"agent_agreement", "token_entropy"},
            )
        )
        traj = Trajectory(query="test")
        for i in range(3):
            t = make_turn(i + 1)
            t.agent_agreement = 0.3
            t.token_entropy = 0.15
            t.answer_similarity = 0.9
            t.info_gain = 0.1
            traj.add_turn(t)
        result = policy.decide(traj)
        assert result.decision == Decision.SCALE

    def test_policy_mask_ignores_inactive_signals(self, make_turn):
        policy = PACEPolicy(
            PolicyConfig(
                similarity_threshold=0.95,
                relative_thresholds=False,
                min_turns=2,
                signal_mask={"answer_similarity"},
            )
        )
        traj = Trajectory(query="test")
        for i in range(3):
            t = make_turn(i + 1)
            t.answer_similarity = 0.98
            t.info_gain = 0.5
            t.agent_agreement = 0.1
            traj.add_turn(t)
        assert policy.decide(traj).decision == Decision.STOP
