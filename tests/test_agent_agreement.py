"""Tests for S5 (agent agreement) TurnState fields."""

from pace.trajectory import TurnState


class TestAgentAgreement:
    def test_agreement_field_exists(self):
        t = TurnState(turn_number=1, answer="test")
        t.agent_agreement = 0.9
        signal = t.get_signal("agent_agreement")
        print(f"signal: {signal}")
        assert t.get_signal("agent_agreement") == 0.9

    def test_agreement_derivatives(self):
        t = TurnState(turn_number=1, answer="test")
        t.agent_agreement_v = 0.1
        t.agent_agreement_a = -0.05
        velocity = t.get_velocity("agent_agreement")
        acceleration = t.get_acceleration("agent_agreement")
        print(f"velocity: {velocity}, acceleration: {acceleration}")
        assert t.get_velocity("agent_agreement") == 0.1
        assert t.get_acceleration("agent_agreement") == -0.05
