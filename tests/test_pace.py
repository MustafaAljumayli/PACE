"""Tests for PACE signals, policy, ablation, and dual-agent agreement."""

import math
import pytest

from pace.trajectory import Trajectory, TurnState, SIGNAL_NAMES
from pace.signals import SignalComputer
from pace.policy import PACEPolicy, PolicyConfig, Decision


# ── Mock embedder for deterministic testing ──

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


def make_turn(n, answer="ans", context="", tool="", cumul="", metadata=None):
    return TurnState(
        turn_number=n, answer=answer, retrieved_context=context,
        cumulative_context=cumul, tool_called=tool,
        metadata=metadata or {},
    )


# ════════════════════════════════════════════════════════════
# Signal computation tests
# ════════════════════════════════════════════════════════════

class TestSignalComputation:

    def test_all_five_signal_names(self):
        assert len(SIGNAL_NAMES) == 5
        assert "agent_agreement" in SIGNAL_NAMES

    def test_first_turn_defaults(self, sc):
        traj = Trajectory(query="test")
        traj.add_turn(make_turn(1))
        sc.compute(traj)
        t = traj.turns[0]
        assert t.answer_similarity == 0.0
        assert t.answer_similarity_v == 0.0
        assert t.answer_similarity_a == 0.0

    def test_identical_answers_converge(self, sc):
        traj = Trajectory(query="test")
        traj.add_turn(make_turn(1, "The answer is 42"))
        sc.compute(traj)
        traj.add_turn(make_turn(2, "I suspect that the answer may be 42"))
        sc.compute(traj)
        assert traj.turns[1].answer_similarity == 1.0

    def test_token_entropy_non_negative(self, sc):
        """Token entropy is Shannon entropy in bits, so it is non-negative."""
        traj = Trajectory(query="test")
        t = make_turn(1, answer="the the the cat cat dog")
        traj.add_turn(t)
        sc.compute(traj)
        assert t.token_entropy >= 0.0

    def test_token_entropy_uniform_is_high(self, sc):
        """All unique tokens should have high entropy (in bits)."""
        traj = Trajectory(query="test")
        t = make_turn(1, answer="alpha bravo charlie delta echo foxtrot")
        traj.add_turn(t)
        sc.compute(traj)
        assert t.token_entropy > 2.0

    def test_token_entropy_repetitive_is_low(self, sc):
        """Repeated token → entropy should be low."""
        traj = Trajectory(query="test")
        t = make_turn(1, answer="the the the the the the")
        traj.add_turn(t)
        sc.compute(traj)
        assert t.token_entropy < 0.01

    def test_token_entropy_uses_logprobs_when_present(self, sc):
        """When logprobs exist, use sequence-level entropy from metadata."""
        traj = Trajectory(query="test")
        t = make_turn(
            1,
            answer="irrelevant for this test",
            metadata={"token_top_logprobs": [[-0.6931, -0.6931], [-0.6931, -0.6931]]},
        )
        traj.add_turn(t)
        sc.compute(traj)
        # Uniform binary distribution -> ~1 bit entropy per token.
        assert 0.95 <= t.token_entropy <= 1.05

    def test_token_entropy_logprobs_override_text_fallback(self, sc):
        """Metadata logprobs should take precedence over lexical fallback."""
        traj = Trajectory(query="test")
        t = make_turn(
            1,
            answer="the the the the the the",
            metadata={"token_top_logprobs": [[-0.6931, -0.6931]]},
        )
        traj.add_turn(t)
        sc.compute(traj)
        assert t.token_entropy > 0.9

    def test_tool_entropy_single_tool(self, sc):
        traj = Trajectory(query="test")
        for i in range(3):
            traj.add_turn(make_turn(i + 1, tool="search"))
            sc.compute(traj)
        assert traj.turns[-1].tool_entropy == 0.0

    def test_tool_entropy_diverse(self, sc):
        traj = Trajectory(query="test")
        for i, tool in enumerate(["search", "wiki", "python"]):
            traj.add_turn(make_turn(i + 1, tool=tool))
            sc.compute(traj)
        assert abs(traj.turns[-1].tool_entropy - math.log2(3)) < 0.01

    def test_derivatives_computed_for_all_signals(self, sc):
        traj = Trajectory(query="test")
        for i in range(4):
            t = make_turn(i + 1, answer=f"answer v{i}", tool="search")
            traj.add_turn(t)
            sc.compute(traj)

        last = traj.turns[-1]
        # All signals that were computed should have derivatives
        for sig in ["answer_similarity", "info_gain", "token_entropy", "tool_entropy"]:
            assert last.get_velocity(sig) is not None, f"{sig} velocity missing"
            assert last.get_acceleration(sig) is not None, f"{sig} acceleration missing"


# ════════════════════════════════════════════════════════════
# Signal mask (ablation) tests
# ════════════════════════════════════════════════════════════

class TestSignalMask:

    def test_mask_only_computes_selected(self):
        sc = SignalComputer(embedder=MockEmbedder(), signal_mask={"token_entropy"})
        traj = Trajectory(query="test")
        traj.add_turn(make_turn(1, answer="hello world foo bar"))
        sc.compute(traj)
        t = traj.turns[0]
        assert t.token_entropy is not None
        assert t.answer_similarity is None  # Not in mask
        assert t.info_gain is None

    def test_mask_s1_only(self):
        sc = SignalComputer(embedder=MockEmbedder(), signal_mask={"answer_similarity"})
        traj = Trajectory(query="test")
        traj.add_turn(make_turn(1, answer="aaa"))
        sc.compute(traj)
        traj.add_turn(make_turn(2, answer="aab"))
        sc.compute(traj)
        assert traj.turns[1].answer_similarity is not None
        assert traj.turns[1].token_entropy is None


# ════════════════════════════════════════════════════════════
# Policy tests
# ════════════════════════════════════════════════════════════

class TestPolicy:

    def test_continue_below_min(self, sc):
        policy = PACEPolicy(PolicyConfig(min_turns=3))
        traj = Trajectory(query="test")
        traj.add_turn(make_turn(1))
        sc.compute(traj)
        assert policy.decide(traj).decision == Decision.CONTINUE

    def test_stop_at_max(self, sc):
        policy = PACEPolicy(PolicyConfig(max_turns=3, min_turns=1))
        traj = Trajectory(query="test")
        for i in range(3):
            t = make_turn(i + 1, answer=f"diff {i}")
            t.answer_similarity = 0.5
            t.info_gain = 0.3
            traj.add_turn(t)
        assert policy.decide(traj).decision == Decision.STOP

    def test_convergence_with_multiple_signals(self):
        policy = PACEPolicy(PolicyConfig(
            similarity_threshold=0.95,
            convergence_window=2,
            info_gain_floor=0.05,
            token_entropy_ceiling=0.3,
            relative_thresholds=False,
            min_turns=2,
            signal_mask={"answer_similarity", "info_gain", "token_entropy"},
        ))
        traj = Trajectory(query="test")
        for i in range(3):
            t = make_turn(i + 1)
            t.answer_similarity = 0.98
            t.info_gain = 0.02
            t.token_entropy = 0.15
            traj.add_turn(t)
        assert policy.decide(traj).decision == Decision.STOP

    def test_no_convergence_if_one_signal_disagrees(self):
        """If token entropy is high, don't stop even if other signals converge."""
        policy = PACEPolicy(PolicyConfig(
            similarity_threshold=0.95,
            token_entropy_ceiling=0.3,
            relative_thresholds=False,
            min_turns=2,
            signal_mask={"answer_similarity", "token_entropy"},
        ))
        traj = Trajectory(query="test")
        for i in range(3):
            t = make_turn(i + 1)
            t.answer_similarity = 0.98  # Converged
            t.token_entropy = 0.8        # NOT confident
            traj.add_turn(t)
        assert policy.decide(traj).decision == Decision.CONTINUE

    def test_rewind_on_degradation(self):
        policy = PACEPolicy(PolicyConfig(
            min_turns=2,
            signal_mask={"answer_similarity"},
        ))
        traj = Trajectory(query="test")
        t1 = make_turn(1)
        t1.answer_similarity = 0.0
        traj.add_turn(t1)
        t2 = make_turn(2)
        t2.answer_similarity = 0.97  # Peak
        traj.add_turn(t2)
        t3 = make_turn(3)
        t3.answer_similarity = 0.3   # Degraded
        traj.add_turn(t3)
        result = policy.decide(traj)
        assert result.decision == Decision.REWIND
        assert result.rewind_to_turn == 2

    def test_scale_on_confident_conflict(self):
        """Scale when agents disagree but each is confident."""
        policy = PACEPolicy(PolicyConfig(
            min_turns=2,
            token_entropy_ceiling=0.3,
            signal_mask={"agent_agreement", "token_entropy"},
        ))
        traj = Trajectory(query="test")
        for i in range(3):
            t = make_turn(i + 1)
            t.agent_agreement = 0.3    # Agents disagree
            t.token_entropy = 0.15      # But each is confident
            t.answer_similarity = 0.9   # Within-agent stable
            t.info_gain = 0.1
            traj.add_turn(t)
        result = policy.decide(traj)
        assert result.decision == Decision.SCALE

    def test_policy_mask_ignores_inactive_signals(self):
        """Policy with mask={S1} should not check S5 for convergence."""
        policy = PACEPolicy(PolicyConfig(
            similarity_threshold=0.95,
            relative_thresholds=False,
            min_turns=2,
            signal_mask={"answer_similarity"},
        ))
        traj = Trajectory(query="test")
        for i in range(3):
            t = make_turn(i + 1)
            t.answer_similarity = 0.98
            t.info_gain = 0.5  # Would prevent convergence if checked
            t.agent_agreement = 0.1  # Would prevent convergence if checked
            traj.add_turn(t)
        # Should still stop because only S1 is in the mask
        assert policy.decide(traj).decision == Decision.STOP


# ════════════════════════════════════════════════════════════
# Agent agreement (S5) field tests
# ════════════════════════════════════════════════════════════

class TestAgentAgreement:

    def test_agreement_field_exists(self):
        t = TurnState(turn_number=1, answer="test")
        t.agent_agreement = 0.9
        assert t.get_signal("agent_agreement") == 0.9

    def test_agreement_derivatives(self):
        t = TurnState(turn_number=1, answer="test")
        t.agent_agreement_v = 0.1
        t.agent_agreement_a = -0.05
        assert t.get_velocity("agent_agreement") == 0.1
        assert t.get_acceleration("agent_agreement") == -0.05


# ════════════════════════════════════════════════════════════
# Ablation condition generation test
# ════════════════════════════════════════════════════════════

class TestAblation:

    def test_generates_31_conditions(self):
        from experiments.run_ablation import generate_ablation_conditions
        conditions = generate_ablation_conditions()
        # 2^5 - 1 = 31 non-empty subsets
        assert len(conditions) == 31

    def test_solo_conditions(self):
        from experiments.run_ablation import generate_ablation_conditions
        conditions = generate_ablation_conditions()
        solos = [(n, m) for n, m in conditions if len(m) == 1]
        assert len(solos) == 5

    def test_full_condition(self):
        from experiments.run_ablation import generate_ablation_conditions
        conditions = generate_ablation_conditions()
        full = [(n, m) for n, m in conditions if len(m) == 5]
        assert len(full) == 1
        assert full[0][1] == set(SIGNAL_NAMES)
