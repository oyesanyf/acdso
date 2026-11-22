"""
Tests for Pareto Optimization Module
====================================

Tests covering Pareto dominance, front construction,
and knee-point selection.
"""

import numpy as np
import pytest

from acdso.pareto import (
    CandidateResult,
    dominates,
    pareto_front,
    knee_point_selection,
    filter_by_constraints,
)


def make_candidate(
    name: str,
    metric: float,
    train_time: float,
    peak_mem: float,
    latency: float,
    risk: float,
    rob: float = 0.0,
) -> CandidateResult:
    """Helper to create CandidateResult for testing."""
    import math
    return CandidateResult(
        name=name,
        pipeline=None,
        metric=metric,
        train_time_s=train_time,
        peak_mem_mb=peak_mem,
        latency_ms_per_row=latency,
        risk_penalty=risk,
        objectives=(
            metric,
            math.log1p(train_time),
            math.log1p(peak_mem),
            math.log1p(latency),
            risk,
            rob,
        ),
    )


class TestDominance:
    """Tests for Pareto dominance."""

    def test_clear_dominance(self):
        """Test clear dominance case."""
        # A is better in all objectives
        a = make_candidate("A", metric=0.9, train_time=10, peak_mem=100, latency=1, risk=0.1)
        b = make_candidate("B", metric=0.8, train_time=20, peak_mem=200, latency=2, risk=0.2)

        assert dominates(a, b)
        assert not dominates(b, a)

    def test_no_dominance(self):
        """Test case where neither dominates."""
        # A is better in metric, B is better in time
        a = make_candidate("A", metric=0.9, train_time=20, peak_mem=100, latency=1, risk=0.1)
        b = make_candidate("B", metric=0.8, train_time=10, peak_mem=100, latency=1, risk=0.1)

        assert not dominates(a, b)
        assert not dominates(b, a)

    def test_equal_not_dominating(self):
        """Test that equal candidates don't dominate each other."""
        a = make_candidate("A", metric=0.9, train_time=10, peak_mem=100, latency=1, risk=0.1)
        b = make_candidate("B", metric=0.9, train_time=10, peak_mem=100, latency=1, risk=0.1)

        assert not dominates(a, b)
        assert not dominates(b, a)


class TestParetoFront:
    """Tests for Pareto front construction."""

    def test_single_candidate(self):
        """Test Pareto front with single candidate."""
        a = make_candidate("A", metric=0.9, train_time=10, peak_mem=100, latency=1, risk=0.1)
        front = pareto_front([a])

        assert len(front) == 1
        assert front[0].name == "A"

    def test_two_nondominated(self):
        """Test Pareto front with two non-dominated candidates."""
        a = make_candidate("A", metric=0.9, train_time=20, peak_mem=100, latency=1, risk=0.1)
        b = make_candidate("B", metric=0.8, train_time=10, peak_mem=100, latency=1, risk=0.1)

        front = pareto_front([a, b])

        assert len(front) == 2

    def test_dominated_filtered(self):
        """Test that dominated candidates are filtered."""
        a = make_candidate("A", metric=0.9, train_time=10, peak_mem=100, latency=1, risk=0.1)
        b = make_candidate("B", metric=0.8, train_time=20, peak_mem=200, latency=2, risk=0.2)

        front = pareto_front([a, b])

        assert len(front) == 1
        assert front[0].name == "A"

    def test_complex_front(self):
        """Test Pareto front with multiple candidates."""
        candidates = [
            make_candidate("A", metric=0.95, train_time=50, peak_mem=500, latency=5, risk=0.1),
            make_candidate("B", metric=0.90, train_time=20, peak_mem=200, latency=2, risk=0.1),
            make_candidate("C", metric=0.85, train_time=10, peak_mem=100, latency=1, risk=0.1),
            make_candidate("D", metric=0.80, train_time=15, peak_mem=150, latency=1.5, risk=0.1),  # Dominated by C
        ]

        front = pareto_front(candidates)

        # D should be dominated by C (better in all but risk which is equal)
        names = [c.name for c in front]
        assert "A" in names
        assert "B" in names
        assert "C" in names
        assert "D" not in names


class TestKneePointSelection:
    """Tests for knee-point selection."""

    def test_single_candidate(self):
        """Test knee-point with single candidate."""
        a = make_candidate("A", metric=0.9, train_time=10, peak_mem=100, latency=1, risk=0.1)

        selected = knee_point_selection([a])
        assert selected.name == "A"

    def test_empty_front_raises(self):
        """Test that empty front raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            knee_point_selection([])

    def test_knee_selection(self):
        """Test knee-point selection logic."""
        # Create candidates along a Pareto front
        candidates = [
            make_candidate("HighAcc", metric=0.95, train_time=100, peak_mem=500, latency=10, risk=0.1),
            make_candidate("Balanced", metric=0.90, train_time=30, peak_mem=150, latency=3, risk=0.1),
            make_candidate("Fast", metric=0.80, train_time=5, peak_mem=50, latency=1, risk=0.1),
        ]

        selected = knee_point_selection(candidates)

        # The balanced option should typically be the knee
        assert selected is not None

    def test_two_candidates(self):
        """Test knee-point with two candidates."""
        a = make_candidate("A", metric=0.9, train_time=50, peak_mem=300, latency=5, risk=0.1)
        b = make_candidate("B", metric=0.8, train_time=10, peak_mem=100, latency=1, risk=0.1)

        selected = knee_point_selection([a, b])
        # Should return one of them
        assert selected.name in ["A", "B"]

    def test_knee_ignores_subms_jitter_and_small_robustness_noise(self):
        """Sub-ms timing jitter and <0.005 robustness_drop noise must not override real CV gaps."""
        strong = make_candidate("Ridge_alpha0.01", metric=0.985, train_time=0.030, peak_mem=0.25, latency=0.0630, risk=0.016, rob=0.0075)
        mid = make_candidate("ElasticNet_mid", metric=0.920, train_time=0.026, peak_mem=0.25, latency=0.0626, risk=0.016, rob=0.0055)
        weak = make_candidate("ElasticNet_weak", metric=0.835, train_time=0.022, peak_mem=0.25, latency=0.0621, risk=0.016, rob=0.0038)

        front = pareto_front([strong, mid, weak])
        assert {c.name for c in front} == {"Ridge_alpha0.01", "ElasticNet_mid", "ElasticNet_weak"}

        assert knee_point_selection([strong, mid, weak]).name == "Ridge_alpha0.01"
        assert knee_point_selection([weak, mid, strong]).name == "Ridge_alpha0.01"

    def test_knee_rejects_dominated_side_above_chord_and_picks_utopia_side(self):
        """Dominated-side points above the a-b chord must never beat anchors or utopia knees."""
        anchor = make_candidate("AnchorBest", metric=0.99, train_time=2.0, peak_mem=20.0, latency=0.5, risk=0.05, rob=0.01)
        above_chord = make_candidate("AboveChord", metric=0.99, train_time=25.0, peak_mem=250.0, latency=8.0, risk=0.25, rob=0.05)
        fast_low = make_candidate("FastLow", metric=0.80, train_time=0.5, peak_mem=10.0, latency=0.1, risk=0.04, rob=0.01)

        assert knee_point_selection([above_chord, anchor, fast_low]).name == "AnchorBest"

        high = make_candidate("High", metric=0.95, train_time=50.0, peak_mem=400.0, latency=10.0, risk=0.10, rob=0.02)
        dom_side = make_candidate("DominatedSide", metric=0.94, train_time=49.0, peak_mem=395.0, latency=9.9, risk=0.45, rob=0.20)
        utopia_knee = make_candidate("UtopiaKnee", metric=0.91, train_time=1.0, peak_mem=20.0, latency=0.2, risk=0.05, rob=0.01)
        low = make_candidate("Low", metric=0.75, train_time=0.2, peak_mem=5.0, latency=0.05, risk=0.04, rob=0.005)

        assert knee_point_selection([high, dom_side, utopia_knee, low]).name == "UtopiaKnee"
        assert knee_point_selection([high, dom_side, low]).name != "DominatedSide"

    def test_knee_two_point_front_is_order_independent(self):
        """Two-candidate fronts must not pick based on list order."""
        high = make_candidate("High", metric=0.99, train_time=0.03, peak_mem=1.0, latency=0.05, risk=0.02)
        low = make_candidate("Low", metric=0.84, train_time=0.02, peak_mem=1.0, latency=0.04, risk=0.02)

        assert knee_point_selection([high, low]).name == "High"
        assert knee_point_selection([low, high]).name == "High"

    def test_knee_constant_axes_and_negative_metrics(self):
        """Constant cost axes and negative scorers (e.g. neg_mse) select deterministically."""
        # All cost/risk axes identical: highest metric wins regardless of order
        c1 = make_candidate("BestConst", metric=0.95, train_time=1.0, peak_mem=10.0, latency=0.1, risk=0.05, rob=0.01)
        c2 = make_candidate("MidConst", metric=0.90, train_time=1.0, peak_mem=10.0, latency=0.1, risk=0.05, rob=0.01)
        c3 = make_candidate("LowConst", metric=0.85, train_time=1.0, peak_mem=10.0, latency=0.1, risk=0.05, rob=0.01)
        assert knee_point_selection([c3, c2, c1]).name == "BestConst"
        assert knee_point_selection([c1, c2, c3]).name == "BestConst"

        # Negative metric values (e.g., neg_mean_squared_error: -10 > -12 > -50)
        n_best = make_candidate("NegBest", metric=-10.0, train_time=8.0, peak_mem=100.0, latency=2.0, risk=0.05, rob=0.01)
        n_mid = make_candidate("NegMid", metric=-12.0, train_time=1.0, peak_mem=15.0, latency=0.1, risk=0.04, rob=0.008)
        n_bad = make_candidate("NegBad", metric=-50.0, train_time=0.2, peak_mem=5.0, latency=0.05, risk=0.03, rob=0.005)
        assert knee_point_selection([n_bad, n_mid, n_best], preference="balanced").name == "NegMid"
        assert knee_point_selection([n_bad, n_mid, n_best], preference="accuracy").name == "NegBest"
        assert knee_point_selection([n_bad, n_mid, n_best], preference="speed").name in {"NegMid", "NegBad"}
        assert knee_point_selection([n_bad, n_mid, n_best], preference="low_risk").name in {"NegMid", "NegBad"}

    def test_knee_preference_modes(self):
        """Preference modes steer selection appropriately and unknown values fall back to balanced."""
        acc_model = make_candidate("HighAccSlow", metric=0.96, train_time=120.0, peak_mem=800.0, latency=15.0, risk=0.22, rob=0.07)
        bal_model = make_candidate("BalancedKnee", metric=0.94, train_time=15.0, peak_mem=120.0, latency=2.5, risk=0.10, rob=0.02)
        fast_safe = make_candidate("FastSafe", metric=0.86, train_time=0.1, peak_mem=5.0, latency=0.02, risk=0.01, rob=0.002)
        front = [acc_model, bal_model, fast_safe]

        assert knee_point_selection(front, preference="balanced").name == "BalancedKnee"
        assert knee_point_selection(front, preference="unknown_mode").name == "BalancedKnee"
        assert knee_point_selection(front, preference="accuracy").name == "HighAccSlow"
        assert knee_point_selection(front, preference="metric").name == "HighAccSlow"
        assert knee_point_selection(front, preference="speed").name == "FastSafe"
        assert knee_point_selection(front, preference="fast").name == "FastSafe"
        assert knee_point_selection(front, preference="low_risk").name in {"BalancedKnee", "FastSafe"}
        assert knee_point_selection(front, preference="safe").name in {"BalancedKnee", "FastSafe"}

    def test_knee_small_scale_negative_metrics(self):
        """Small-scale negative metrics (e.g. neg_mse ~ -1e-6 or -1e-10) must not collapse to 0."""
        import itertools

        best_neg = make_candidate("BestNeg", metric=-1e-6, train_time=0.5, peak_mem=2.0, latency=0.1, risk=0.04, rob=0.005)
        bad_neg = make_candidate("TerribleNeg", metric=-5e-5, train_time=0.003, peak_mem=0.01, latency=0.005, risk=0.01, rob=0.001)
        for perm in itertools.permutations([best_neg, bad_neg]):
            for pref in ("balanced", "accuracy", "low_risk"):
                assert knee_point_selection(list(perm), preference=pref).name == "BestNeg"

        tiny_best = make_candidate("TinyBest", metric=-1e-10, train_time=0.5, peak_mem=2.0, latency=0.1, risk=0.04, rob=0.005)
        tiny_bad = make_candidate("TinyTerrible", metric=-5e-9, train_time=0.003, peak_mem=0.01, latency=0.005, risk=0.01, rob=0.001)
        for perm in itertools.permutations([tiny_best, tiny_bad]):
            assert knee_point_selection(list(perm), preference="balanced").name == "TinyBest"
            assert knee_point_selection(list(perm), preference="accuracy").name == "TinyBest"

        # 3-point tiny-scale front: knee or best must never pick the 50x worse loss
        tiny_mid = make_candidate("TinyMid", metric=-1.5e-10, train_time=0.05, peak_mem=0.2, latency=0.02, risk=0.02, rob=0.002)
        for perm in itertools.permutations([tiny_best, tiny_mid, tiny_bad]):
            assert knee_point_selection(list(perm), preference="balanced").name in {"TinyBest", "TinyMid"}

    def test_knee_all_non_finite_pos_neg_inf_front(self):
        """All-non-finite [+inf, -inf] fronts deterministically prefer +inf on metric (higher_better=True)."""
        pos_inf = make_candidate("PosInf", metric=float("inf"), train_time=0.1, peak_mem=1.0, latency=0.01, risk=0.01)
        neg_inf = make_candidate("NegInf", metric=float("-inf"), train_time=0.1, peak_mem=1.0, latency=0.01, risk=0.01)

        assert knee_point_selection([pos_inf, neg_inf]).name == "PosInf"
        assert knee_point_selection([neg_inf, pos_inf]).name == "PosInf"

        # Also verify non-finite cost axis (lower is better -> -inf beats +inf)
        low_cost = make_candidate("LowCost", metric=0.9, train_time=0.1, peak_mem=1.0, latency=0.01, risk=float("-inf"))
        high_cost = make_candidate("HighCost", metric=0.9, train_time=0.1, peak_mem=1.0, latency=0.01, risk=float("inf"))
        assert knee_point_selection([high_cost, low_cost]).name == "LowCost"
        assert knee_point_selection([low_cost, high_cost]).name == "LowCost"


class TestConstraintFiltering:
    """Tests for constraint-based filtering."""

    def test_no_constraints(self):
        """Test filtering with no constraints."""
        candidates = [
            make_candidate("A", metric=0.9, train_time=10, peak_mem=100, latency=1, risk=0.1),
            make_candidate("B", metric=0.8, train_time=20, peak_mem=200, latency=2, risk=0.2),
        ]

        passing, pruned = filter_by_constraints(candidates)

        assert len(passing) == 2
        assert len(pruned) == 0

    def test_latency_constraint(self):
        """Test latency constraint filtering."""
        candidates = [
            make_candidate("A", metric=0.9, train_time=10, peak_mem=100, latency=1, risk=0.1),
            make_candidate("B", metric=0.8, train_time=20, peak_mem=200, latency=15, risk=0.2),
        ]

        passing, pruned = filter_by_constraints(
            candidates,
            max_latency_ms_per_row=10.0,
        )

        assert len(passing) == 1
        assert passing[0].name == "A"
        assert len(pruned) == 1
        assert pruned[0]["name"] == "B"

    def test_memory_constraint(self):
        """Test memory constraint filtering."""
        candidates = [
            make_candidate("A", metric=0.9, train_time=10, peak_mem=100, latency=1, risk=0.1),
            make_candidate("B", metric=0.8, train_time=20, peak_mem=600, latency=2, risk=0.2),
        ]

        passing, pruned = filter_by_constraints(
            candidates,
            max_peak_mem_mb=500.0,
        )

        assert len(passing) == 1
        assert passing[0].name == "A"

    def test_train_time_constraint(self):
        """Test training time constraint filtering."""
        candidates = [
            make_candidate("A", metric=0.9, train_time=30, peak_mem=100, latency=1, risk=0.1),
            make_candidate("B", metric=0.8, train_time=120, peak_mem=200, latency=2, risk=0.2),
        ]

        passing, pruned = filter_by_constraints(
            candidates,
            max_train_time_s=60.0,
        )

        assert len(passing) == 1
        assert passing[0].name == "A"

    def test_multiple_constraints(self):
        """Test multiple constraints simultaneously."""
        candidates = [
            make_candidate("A", metric=0.9, train_time=10, peak_mem=100, latency=1, risk=0.1),
            make_candidate("B", metric=0.8, train_time=120, peak_mem=200, latency=2, risk=0.2),  # Fails time
            make_candidate("C", metric=0.85, train_time=20, peak_mem=600, latency=1, risk=0.1),  # Fails memory
        ]

        passing, pruned = filter_by_constraints(
            candidates,
            max_train_time_s=60.0,
            max_peak_mem_mb=500.0,
            max_latency_ms_per_row=10.0,
        )

        assert len(passing) == 1
        assert passing[0].name == "A"
        assert len(pruned) == 2


class TestCandidateResult:
    """Tests for CandidateResult dataclass."""

    def test_to_dict(self):
        """Test CandidateResult serialization."""
        c = make_candidate("Test", metric=0.9, train_time=10, peak_mem=100, latency=1, risk=0.1)
        c.meta = {"cv_scores": [0.88, 0.90, 0.92]}

        d = c.to_dict()

        assert isinstance(d, dict)
        assert d["name"] == "Test"
        assert d["metric"] == 0.9
        assert "pipeline" not in d  # Pipeline should be excluded
        assert d["meta"]["cv_scores"] == [0.88, 0.90, 0.92]

