"""Tests for ablation condition generation."""

from pace.trajectory import SIGNAL_NAMES


class TestAblation:
    def test_generates_31_conditions(self):
        from experiments.run_ablation import generate_ablation_conditions

        conditions = generate_ablation_conditions()
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
