"""
ACDSO v3.0 - Pareto Optimization Module
========================================

Multi-objective optimization components including:
- Pareto dominance checking (6+ objectives supported)
- Pareto front construction
- Knee-point selection for optimal tradeoffs

v3 Changes:
- Extended to support 6 objectives (added robustness_drop)
- Explicit objective direction handling (maximize vs minimize)
- Improved dominance checking for variable-length objectives
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Sequence

import math
import numpy as np


# Objective directions: True = maximize, False = minimize
# Index: 0=metric, 1=train_time, 2=memory, 3=latency, 4=risk, 5=robustness_drop
OBJECTIVE_DIRECTIONS: Tuple[bool, ...] = (
    True,   # 0: metric (higher is better)
    False,  # 1: log(train_time) (lower is better)
    False,  # 2: log(peak_mem) (lower is better)
    False,  # 3: log(latency) (lower is better)
    False,  # 4: risk_penalty (lower is better)
    False,  # 5: robustness_drop (lower is better) - v3 addition
)

OBJECTIVE_NAMES: Tuple[str, ...] = (
    "metric",
    "log_train_time",
    "log_peak_mem", 
    "log_latency",
    "risk_penalty",
    "robustness_drop",
)


@dataclass
class CandidateResult:
    """
    Result of evaluating a candidate pipeline (v3).

    Contains all metrics needed for multi-objective optimization
    and constraint checking.

    Attributes
    ----------
    name : str
        Name/identifier of the candidate.
    pipeline : Any
        Fitted sklearn Pipeline.
    metric : float
        Primary evaluation metric (e.g., accuracy, F1).
    train_time_s : float
        Training time in seconds.
    peak_mem_mb : float
        Peak memory usage in megabytes.
    latency_ms_per_row : float
        Inference latency in milliseconds per row.
    risk_penalty : float
        Computed risk penalty (0.0 to 1.0).
    objectives : Tuple[float, ...]
        6-objective vector (v3):
        [0] metric         - primary score (MAXIMIZE)
        [1] log_train_time - training cost (minimize)
        [2] log_peak_mem   - memory cost (minimize)
        [3] log_latency    - inference cost (minimize)
        [4] risk_penalty   - data risk (minimize)
        [5] robustness_drop - worst-case perf drop under stress (minimize)
    meta : Dict[str, Any]
        Additional metadata (e.g., cv_scores, cv_score_std, stress_tests).
    """

    name: str
    pipeline: Any
    metric: float
    train_time_s: float
    peak_mem_mb: float
    latency_ms_per_row: float
    risk_penalty: float
    objectives: Tuple[float, ...]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding pipeline) for JSON serialization."""
        result = {
            "name": self.name,
            "metric": self.metric,
            "train_time_s": self.train_time_s,
            "peak_mem_mb": self.peak_mem_mb,
            "latency_ms_per_row": self.latency_ms_per_row,
            "risk_penalty": self.risk_penalty,
            "objectives": list(self.objectives),
            "meta": self.meta,
        }
        # Include robustness_drop if available in objectives
        if len(self.objectives) >= 6:
            result["robustness_drop"] = self.objectives[5]
        return result
    
    @property
    def robustness_drop(self) -> float:
        """Get robustness drop (6th objective) if available."""
        return self.objectives[5] if len(self.objectives) >= 6 else 0.0


def dominates(a: CandidateResult, b: CandidateResult) -> bool:
    """
    Check if candidate 'a' Pareto-dominates candidate 'b'.

    A dominates B if A is at least as good as B in all objectives
    and strictly better in at least one.

    Objectives (v3 - 6 objectives):
    - objectives[0]: metric (MAXIMIZE - higher is better)
    - objectives[1]: log(train_time) (minimize - lower is better)
    - objectives[2]: log(peak_mem) (minimize - lower is better)
    - objectives[3]: log(latency) (minimize - lower is better)
    - objectives[4]: risk_penalty (minimize - lower is better)
    - objectives[5]: robustness_drop (minimize - lower is better)

    Parameters
    ----------
    a : CandidateResult
        First candidate.
    b : CandidateResult
        Second candidate.

    Returns
    -------
    bool
        True if a dominates b.
    """
    obj_a = a.objectives
    obj_b = b.objectives
    
    # Handle variable-length objectives (5 or 6)
    n_obj = min(len(obj_a), len(obj_b), len(OBJECTIVE_DIRECTIONS))
    
    better_or_equal = True
    strictly_better = False
    
    for i in range(n_obj):
        maximize = OBJECTIVE_DIRECTIONS[i]
        va, vb = obj_a[i], obj_b[i]
        a_nan = math.isnan(float(va))
        b_nan = math.isnan(float(vb))

        if a_nan and b_nan:
            continue
        if a_nan:
            better_or_equal = False
            break
        if b_nan:
            strictly_better = True
            continue

        if maximize:
            # Higher is better
            if va < vb:
                better_or_equal = False
                break
            if va > vb:
                strictly_better = True
        else:
            # Lower is better
            if va > vb:
                better_or_equal = False
                break
            if va < vb:
                strictly_better = True
    
    return better_or_equal and strictly_better


def pareto_front(candidates: List[CandidateResult]) -> List[CandidateResult]:
    """
    Extract the Pareto front from a list of candidates.

    The Pareto front contains all non-dominated solutions, i.e.,
    solutions where no other solution is better in all objectives.

    Parameters
    ----------
    candidates : List[CandidateResult]
        List of evaluated candidates.

    Returns
    -------
    List[CandidateResult]
        Pareto-optimal candidates.
    """
    front: List[CandidateResult] = []
    for c in candidates:
        # Check if any other candidate dominates this one
        is_dominated = any(dominates(o, c) for o in candidates if o is not c)
        if not is_dominated:
            front.append(c)
    return front


def knee_point_selection(
    front: List[CandidateResult],
    preference: str = "balanced",
) -> CandidateResult:
    """
    Select the knee point from the Pareto front.

    The knee point represents the solution where additional improvement
    in the primary metric requires disproportionately higher costs/risks.
    It is found by maximizing signed perpendicular distance toward the utopia
    point ``(1, 0)`` from the chord connecting the extreme points on the
    normalized Pareto front, with noise-resistant minimum scale floors and an
    order-independent utopia-distance fallback for degenerate/small fronts.

    v3: Includes robustness_drop in cost/risk aggregation and preference modes.

    Parameters
    ----------
    front : List[CandidateResult]
        Pareto-optimal candidates.
    preference : str, default="balanced"
        Selection preference mode: ``"balanced"``, ``"accuracy"``/``"metric"``,
        ``"speed"``/``"fast"``, or ``"low_risk"``/``"safe"``. Unknown values
        fall back to ``"balanced"``.

    Returns
    -------
    CandidateResult
        Selected candidate at the knee point.

    Raises
    ------
    ValueError
        If the front is empty.
    """
    if not front:
        raise ValueError("Pareto front is empty; cannot select knee point.")

    if len(front) == 1:
        return front[0]

    # Prefer candidates with finite metric (and ideally all-finite objectives) so
    # NaN-metric candidates cannot beat valid finite candidates.
    finite_metric_front = [c for c in front if math.isfinite(float(c.metric))]
    if finite_metric_front:
        all_finite_front = [
            c for c in finite_metric_front
            if all(
                math.isfinite(float(v))
                for v in (
                    c.metric,
                    c.train_time_s,
                    c.peak_mem_mb,
                    c.latency_ms_per_row,
                    c.risk_penalty,
                    c.robustness_drop,
                )
            )
        ]
        front = all_finite_front if all_finite_front else finite_metric_front
        if len(front) == 1:
            return front[0]

    # Extract objective components (aligned with log1p scaling in dominates / pareto_front)
    m = np.array([float(c.metric) for c in front], dtype=float)
    t = np.log1p(np.maximum(0.0, np.array([float(c.train_time_s) for c in front], dtype=float)))
    mem = np.log1p(np.maximum(0.0, np.array([float(c.peak_mem_mb) for c in front], dtype=float)))
    lat = np.log1p(np.maximum(0.0, np.array([float(c.latency_ms_per_row) for c in front], dtype=float)))
    r = np.array([float(c.risk_penalty) for c in front], dtype=float)

    # v3: Extract robustness drop (6th objective)
    rob = np.array([float(c.robustness_drop) for c in front], dtype=float)

    finite_m = m[np.isfinite(m)]
    if finite_m.size > 0:
        max_abs_m = float(np.max(np.abs(finite_m)))
        if float(np.max(finite_m)) <= 0.0:
            m_floor = 0.01 * max(1e-12, max_abs_m)
        else:
            m_floor = 0.01 * max(1.0, max_abs_m)
    else:
        max_abs_m = 1.0
        m_floor = 0.01
    m_tie_scale = max(1e-30, max_abs_m) if max_abs_m > 0.0 else 1.0

    def normalize(x: np.ndarray, higher_better: bool, min_scale: float) -> np.ndarray:
        """Normalize to [0, 1] where 1 is always better, with noise-resistant scale floor."""
        finite_mask = np.isfinite(x)
        if not np.any(finite_mask):
            z = np.full_like(x, 0.5, dtype=float)
        else:
            x_fin = x[finite_mask]
            x_min, x_max = float(np.min(x_fin)), float(np.max(x_fin))
            span = x_max - x_min
            const_tol = 1e-12 * max(abs(x_min), abs(x_max))
            if span <= const_tol:
                z = np.ones_like(x, dtype=float)
            else:
                scale = max(span, float(min_scale))
                if higher_better:
                    z_raw = np.where(finite_mask, 1.0 - (x_max - x) / scale, 0.5)
                else:
                    z_raw = np.where(finite_mask, 1.0 - (x - x_min) / scale, 0.5)
                z = np.clip(z_raw, 0.0, 1.0)
        if not np.all(finite_mask):
            pos_inf = np.isposinf(x)
            neg_inf = np.isneginf(x)
            z = np.where(pos_inf, 1.0 if higher_better else 0.0, z)
            z = np.where(neg_inf, 0.0 if higher_better else 1.0, z)
            z = np.where(np.isnan(x), 0.5, z)
        return z

    # Normalize all objectives (1 = better) with minimum meaningful scale floors
    mb = normalize(m, higher_better=True, min_scale=m_floor)     # metric: higher is better
    tb = normalize(t, higher_better=False, min_scale=0.25)       # time: lower is better
    memb = normalize(mem, higher_better=False, min_scale=0.25)   # memory: lower is better
    latb = normalize(lat, higher_better=False, min_scale=0.10)   # latency: lower is better
    rb = normalize(r, higher_better=False, min_scale=0.03)       # risk: lower is better
    robb = normalize(rob, higher_better=False, min_scale=0.02)   # robustness: lower drop is better

    pref = str(preference or "balanced").strip().lower()
    if pref in ("accuracy", "metric"):
        w_t, w_mem, w_lat, w_r, w_rob = 0.15, 0.10, 0.10, 0.30, 0.35
        w_ben, w_cost = 0.90, 0.10
    elif pref in ("speed", "fast"):
        w_t, w_mem, w_lat, w_r, w_rob = 0.35, 0.15, 0.30, 0.10, 0.10
        w_ben, w_cost = 0.35, 0.65
    elif pref in ("low_risk", "safe"):
        w_t, w_mem, w_lat, w_r, w_rob = 0.10, 0.10, 0.10, 0.35, 0.35
        w_ben, w_cost = 0.55, 0.45
    else:
        pref = "balanced"
        w_t, w_mem, w_lat, w_r, w_rob = 0.25, 0.15, 0.15, 0.20, 0.25
        w_ben, w_cost = 0.60, 0.40

    # Aggregate into benefit vs cost/risk axes (including robustness)
    benefit = mb
    costrisk = 1.0 - (w_t * tb + w_mem * memb + w_lat * latb + w_r * rb + w_rob * robb)

    def _q(val: Any, decimals: int | None = None, negate: bool = False) -> float:
        f = float(val)
        if not math.isfinite(f):
            return float("inf")
        if decimals is not None:
            f = round(f, decimals)
        return -f if negate else f

    def _q_metric(val: Any) -> float:
        f = float(val)
        if math.isnan(f):
            return float("inf")
        if math.isinf(f):
            return -f
        return -round(f / m_tie_scale, 8)

    def tie_key(i: int) -> Tuple[float, float, float, float, float, float, str]:
        c = front[i]
        return (
            _q_metric(c.metric),
            _q(c.risk_penalty, 3),
            _q(c.robustness_drop, 4),
            _q(c.peak_mem_mb, 1),
            _q(c.latency_ms_per_row, 1),
            _q(c.train_time_s, 1),
            str(c.name),
        )

    if pref != "balanced":
        scores = [w_ben * float(benefit[i]) - w_cost * float(costrisk[i]) for i in range(len(front))]
        max_s = max(scores)
        near_best = [i for i, s in enumerate(scores) if s >= max_s - 0.015]
        best_i = min(near_best, key=tie_key)
        return front[best_i]

    pts = np.vstack([benefit, costrisk]).T

    # Find extreme anchors deterministically
    a_idx = min(
        range(len(front)),
        key=lambda i: (-round(float(benefit[i]), 12), round(float(costrisk[i]), 12), tie_key(i)),
    )
    b_idx = min(
        range(len(front)),
        key=lambda i: (round(float(costrisk[i]), 12), -round(float(benefit[i]), 12), tie_key(i)),
    )

    if len(front) >= 3 and a_idx != b_idx:
        a = pts[a_idx]
        b = pts[b_idx]
        ab = b - a
        db = float(ab[0])
        dc = float(ab[1])
        ab_norm = float(math.hypot(db, dc))
        # Guard against near-vertical/near-horizontal chord singularities and noise-only costrisk spread
        if abs(db) >= 0.05 and abs(dc) >= 0.08 and ab_norm >= 0.05:
            utopia = np.array([1.0, 0.0], dtype=float)
            cross_u = db * (float(utopia[1]) - float(a[1])) - dc * (float(utopia[0]) - float(a[0]))
            orient = 1.0 if cross_u >= 0.0 else -1.0
            sdists = [
                orient * (db * (float(p[1]) - float(a[1])) - dc * (float(p[0]) - float(a[0]))) / ab_norm
                for p in pts
            ]
            max_sd = max(sdists)
            if max_sd > 0.01:
                near_best = [i for i, sd in enumerate(sdists) if sd >= max_sd - 1e-9]
                best_i = min(near_best, key=tie_key)
                return front[best_i]

    # Order-independent weighted L2 utopia distance fallback (with noise-tolerant tie band)
    udists = [
        math.sqrt(w_ben * (1.0 - float(benefit[i])) ** 2 + w_cost * float(costrisk[i]) ** 2)
        for i in range(len(front))
    ]
    min_u = min(udists)
    near_best = [i for i, u in enumerate(udists) if u <= min_u + 0.015]
    best_i = min(near_best, key=tie_key)
    return front[best_i]


def filter_by_constraints(
    candidates: List[CandidateResult],
    max_train_time_s: float | None = None,
    max_peak_mem_mb: float | None = None,
    max_latency_ms_per_row: float | None = None,
) -> Tuple[List[CandidateResult], List[Dict[str, Any]]]:
    """
    Filter candidates by constraint violations.

    Parameters
    ----------
    candidates : List[CandidateResult]
        List of evaluated candidates.
    max_train_time_s : float, optional
        Maximum allowed training time.
    max_peak_mem_mb : float, optional
        Maximum allowed peak memory.
    max_latency_ms_per_row : float, optional
        Maximum allowed inference latency.

    Returns
    -------
    Tuple[List[CandidateResult], List[Dict[str, Any]]]
        Tuple of (passing_candidates, pruned_info).
    """
    passing = []
    pruned = []

    for c in candidates:
        violations = []

        if max_train_time_s is not None and c.train_time_s > max_train_time_s:
            violations.append(f"train_time={c.train_time_s:.2f}s > {max_train_time_s}s")

        if max_peak_mem_mb is not None and c.peak_mem_mb > max_peak_mem_mb:
            violations.append(f"peak_mem={c.peak_mem_mb:.1f}MB > {max_peak_mem_mb}MB")

        if max_latency_ms_per_row is not None and c.latency_ms_per_row > max_latency_ms_per_row:
            violations.append(f"latency={c.latency_ms_per_row:.2f}ms > {max_latency_ms_per_row}ms")

        if violations:
            pruned.append({
                "name": c.name,
                "reason": "constraints",
                "violations": violations,
                "metrics": {
                    "train_time_s": c.train_time_s,
                    "peak_mem_mb": c.peak_mem_mb,
                    "latency_ms_per_row": c.latency_ms_per_row,
                    "metric": c.metric,
                    "risk_penalty": c.risk_penalty,
                },
            })
        else:
            passing.append(c)

    return passing, pruned

