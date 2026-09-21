"""Intent-aware MILP benchmark and LP shadow prices (paper equations 6--15).

Rates are Mbit/ms, backlogs and delay slacks are Mbit, arrival rates are
Mbit/ms, and delays are ms. The paper's objective adds these quantities with
the supplied penalty weights; this module does not silently normalize them.

The integer benchmark and continuous price problem are solved separately by
SciPy/HiGHS. Prices are *only* the negated LP inequality marginals for the
RRB-exclusivity rows. Both the global resource constraint and explicit x<=1
bounds are retained, as in the paper. Their redundancy can make the dual
solution nonunique and give zero exclusivity prices even at full utilization.
The other duals are reported for diagnosis, never folded into the broadcast
price. These LP prices are not dual variables of the integer problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csc_matrix


@dataclass(frozen=True)
class LeaderInput:
    """Time-averaged inputs for one cell and one non-RT epoch."""

    rates: NDArray[np.float64]
    backlog: NDArray[np.float64]
    arrivals: NDArray[np.float64]
    minimum_rates: NDArray[np.float64]
    delay_limits: NDArray[np.float64]
    throughput_target: float
    latency_target: float
    resource_budget: int


@dataclass(frozen=True)
class LeaderResult:
    """Feasible benchmark plus optimal LP-relaxation scarcity signals.

    ``objective`` is the maximization objective, evaluated at the returned
    integer allocation and its smallest feasible slacks. Check diagnostics
    for the MILP termination condition; a time-limited incumbent is not a
    proof of optimality. ``requests`` is the Section 3.3 benchmark request count
    and is not clipped to a simulator's potentially smaller action space.
    """

    allocation: NDArray[np.int64]
    requests: NDArray[np.int64]
    rrb_prices: NDArray[np.float64]
    price: float
    rate_slack: NDArray[np.float64]
    delay_slack: NDArray[np.float64]
    throughput_slack: float
    latency_slack: float
    objective: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _LinearProgram:
    objective: NDArray[np.float64]
    matrix: csc_matrix
    rhs: NDArray[np.float64]
    upper: NDArray[np.float64]
    n_rrbs: int
    n_ues: int
    effective_arrivals: NDArray[np.float64]


def _nonnegative_scalar(name: str, value: float, *, positive: bool = False) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(scalar) or scalar < 0 or (positive and scalar == 0):
        relation = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {relation}")
    return scalar


def _validated(problem: LeaderInput) -> LeaderInput:
    rates = np.asarray(problem.rates, dtype=float)
    if rates.ndim != 2 or min(rates.shape) == 0:
        raise ValueError("rates must have nonempty shape (rrbs, ues)")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("rates must contain finite, nonnegative values")
    vectors = {}
    for name in ("backlog", "arrivals", "minimum_rates", "delay_limits"):
        value = np.asarray(getattr(problem, name), dtype=float)
        if value.shape != (rates.shape[1],):
            raise ValueError(f"{name} must have shape ({rates.shape[1]},)")
        if not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError(f"{name} must contain finite, nonnegative values")
        vectors[name] = value
    budget = problem.resource_budget
    if (
        isinstance(budget, (bool, np.bool_))
        or not isinstance(budget, (int, np.integer))
        or budget < 0
    ):
        raise ValueError("resource_budget must be a nonnegative integer")
    return LeaderInput(
        rates=rates,
        **vectors,
        throughput_target=_nonnegative_scalar("throughput_target", problem.throughput_target),
        latency_target=_nonnegative_scalar("latency_target", problem.latency_target),
        resource_budget=int(budget),
    )


def _formulate(
    problem: LeaderInput, sla_weight: float, intent_weight: float, arrival_floor: float
) -> _LinearProgram:
    """Construct A z <= b, with z = [x.ravel(), xiR, xiD, deltaT, deltaL]."""
    n, u = problem.rates.shape
    nx = n * u
    columns = nx + 2 * u + 2
    row_count = n + 2 * u + 3
    rate_start, delay_start = nx, nx + u
    throughput_col, latency_col = nx + 2 * u, nx + 2 * u + 1
    objective = np.zeros(columns)
    objective[:nx] = -problem.rates.ravel()  # HiGHS minimizes.
    objective[rate_start:throughput_col] = sla_weight
    objective[throughput_col:] = intent_weight
    upper = np.full(columns, np.inf)
    upper[:nx] = 1.0  # Retain literal Eq. (13) bounds in both problems.
    effective_arrivals = np.maximum(problem.arrivals, arrival_floor)

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    rhs = np.zeros(row_count)

    def put(row: int, col: int, value: float) -> None:
        if value != 0:
            row_indices.append(row)
            col_indices.append(col)
            values.append(float(value))

    # (7) RRB exclusivity; (8) total resource budget.
    rhs[:n] = 1.0
    rhs[n] = problem.resource_budget
    rate_row = n + 1
    delay_row = rate_row + u
    throughput_row = delay_row + u
    latency_row = throughput_row + 1
    for rrb in range(n):
        for ue in range(u):
            col = rrb * u + ue
            rate = problem.rates[rrb, ue]
            put(rrb, col, 1.0)
            put(n, col, 1.0)
            put(rate_row + ue, col, -rate)  # (9)
            put(delay_row + ue, col, -problem.delay_limits[ue] * rate)  # (10)
            put(throughput_row, col, -rate)  # (11)
    for ue in range(u):
        put(rate_row + ue, rate_start + ue, -1.0)
        put(delay_row + ue, delay_start + ue, -1.0)
        put(latency_row, delay_start + ue, 1.0 / (u * effective_arrivals[ue]))
    rhs[rate_row:delay_row] = -problem.minimum_rates
    rhs[delay_row:throughput_row] = -problem.backlog
    put(throughput_row, throughput_col, -1.0)
    rhs[throughput_row] = -problem.throughput_target
    put(latency_row, latency_col, -1.0)  # (12)
    rhs[latency_row] = problem.latency_target
    matrix = csc_matrix((values, (row_indices, col_indices)), shape=(row_count, columns))
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("Input scales overflowed the optimization coefficients")
    return _LinearProgram(objective, matrix, rhs, upper, n, u, effective_arrivals)


def _check_incumbent(result: Any, model: _LinearProgram) -> NDArray[np.float64]:
    """Reject missing, fractional, or infeasible solver incumbents explicitly."""
    if result.status not in (0, 1) or result.x is None:
        raise RuntimeError(f"Leader MILP failed (status {result.status}): {result.message}")
    values = np.asarray(result.x, dtype=float)
    if values.shape != model.objective.shape or not np.all(np.isfinite(values)):
        raise RuntimeError("Leader MILP returned no finite incumbent")
    nx = model.n_rrbs * model.n_ues
    if np.max(np.abs(values[:nx] - np.rint(values[:nx]))) > 1e-6:
        raise RuntimeError("Leader MILP returned a fractional incumbent")
    # Scale row residuals to their activity/RHS; roundoff is not infeasibility.
    activities = np.asarray(model.matrix @ values).ravel()
    tolerance = 1e-6 * np.maximum(1.0, np.maximum(np.abs(activities), np.abs(model.rhs)))
    if (
        np.any(values < -1e-6)
        or np.any(values[:nx] > 1 + 1e-6)
        or np.any(activities - model.rhs > tolerance)
    ):
        raise RuntimeError("Leader MILP returned an infeasible incumbent")
    return values


def solve_leader(
    problem: LeaderInput,
    sla_weight: float = 10.0,
    intent_weight: float = 10.0,
    arrival_floor: float = 1e-9,
    time_limit: float = 10.0,
    mip_gap: float = 0.01,
) -> LeaderResult:
    """Solve the paper's MILP, then its LP relaxation for Eq. (15) prices.

    ``arrival_floor`` replaces zero/tiny arrivals only in Eq. (12)'s
    denominator. This is an explicit numerical convention for a case where
    the paper's division is undefined; a nonempty queue with zero arrivals
    remains heavily penalized, rather than being silently ignored.

    ``time_limit`` applies separately to each HiGHS solve. A time-limited
    integer incumbent is accepted only after feasibility and integrality
    checks and is labeled in diagnostics. Failure to obtain an optimal LP
    price solution raises ``RuntimeError``. No heuristic fallback or fabricated
    optimality/price is returned. Zero penalty weights are supported; minimal
    feasible slacks are reported even when the solver has slack degeneracy.
    """
    problem = _validated(problem)
    sla_weight = _nonnegative_scalar("sla_weight", sla_weight)
    intent_weight = _nonnegative_scalar("intent_weight", intent_weight)
    arrival_floor = _nonnegative_scalar("arrival_floor", arrival_floor, positive=True)
    time_limit = _nonnegative_scalar("time_limit", time_limit, positive=True)
    mip_gap = _nonnegative_scalar("mip_gap", mip_gap)
    if mip_gap >= 1:
        raise ValueError("mip_gap must be smaller than 1")
    model = _formulate(problem, sla_weight, intent_weight, arrival_floor)
    n, u = model.n_rrbs, model.n_ues
    nx = n * u
    integrality = np.zeros(model.objective.size, dtype=int)
    integrality[:nx] = 1
    integer = milp(
        model.objective,
        integrality=integrality,
        bounds=Bounds(np.zeros_like(model.upper), model.upper),
        constraints=LinearConstraint(model.matrix, -np.inf, model.rhs),
        options={"time_limit": time_limit, "mip_rel_gap": mip_gap},
    )
    incumbent = _check_incumbent(integer, model)
    allocation = np.rint(incumbent[:nx]).astype(np.int64).reshape(n, u)
    if np.any(allocation.sum(axis=1) > 1) or allocation.sum() > problem.resource_budget:
        raise RuntimeError("Rounded MILP incumbent violates physical resource constraints")

    relaxed = linprog(
        model.objective,
        A_ub=model.matrix,
        b_ub=model.rhs,
        bounds=list(zip(np.zeros_like(model.upper), model.upper)),
        method="highs",
        options={"time_limit": time_limit},
    )
    if not relaxed.success or relaxed.status != 0:
        raise RuntimeError(f"Leader LP pricing failed (status {relaxed.status}): {relaxed.message}")
    marginals = np.asarray(relaxed.ineqlin.marginals, dtype=float)
    upper_marginals = np.asarray(relaxed.upper.marginals, dtype=float)
    if not np.all(np.isfinite(marginals)) or not np.all(np.isfinite(upper_marginals)):
        raise RuntimeError("Leader LP returned nonfinite shadow prices")
    # The minimization marginal is nonpositive for a <= capacity constraint.
    rrb_prices = np.maximum(-marginals[:n], 0.0)

    service = (problem.rates * allocation).sum(axis=0)
    rate_slack = np.maximum(problem.minimum_rates - service, 0.0)
    delay_slack = np.maximum(problem.backlog - problem.delay_limits * service, 0.0)
    throughput_slack = max(problem.throughput_target - float(service.sum()), 0.0)
    latency_slack = max(
        float(np.mean(delay_slack / model.effective_arrivals)) - problem.latency_target, 0.0
    )
    objective = float(
        service.sum()
        - sla_weight * (rate_slack.sum() + delay_slack.sum())
        - intent_weight * (throughput_slack + latency_slack)
    )
    if not np.isfinite(objective):
        raise RuntimeError(
            "Leader objective overflowed; rescale rates, backlogs, or the arrival floor"
        )
    gap = getattr(integer, "mip_gap", None)
    bound = getattr(integer, "mip_dual_bound", None)
    diagnostics: dict[str, Any] = {
        "milp_status": int(integer.status),
        "milp_message": str(integer.message),
        "milp_termination": "optimal_within_requested_gap"
        if integer.status == 0
        else "limit_with_feasible_incumbent",
        "optimality_proven": bool(integer.status == 0 and gap is not None and gap <= 1e-9),
        "mip_gap": None if gap is None else float(gap),
        "mip_bound_maximization": None if bound is None else -float(bound),
        "lp_status": int(relaxed.status),
        "lp_objective": -float(relaxed.fun),
        "lp_fractional_allocation": np.asarray(relaxed.x[:nx]).reshape(n, u).tolist(),
        "resource_budget_price": float(max(-marginals[n], 0.0)),
        "allocation_upper_bound_prices": np.maximum(-upper_marginals[:nx], 0.0)
        .reshape(n, u)
        .tolist(),
        "arrival_floor": float(arrival_floor),
        "arrival_floor_applied_count": int(np.count_nonzero(problem.arrivals < arrival_floor)),
        "dual_caveat": "Redundant budget and x<=1 bounds can split scarcity across nonunique LP duals; only exclusivity duals enter the broadcast price.",
    }
    return LeaderResult(
        allocation=allocation,
        requests=allocation.sum(axis=0),
        rrb_prices=rrb_prices,
        price=float(rrb_prices.mean()),
        rate_slack=rate_slack,
        delay_slack=delay_slack,
        throughput_slack=throughput_slack,
        latency_slack=latency_slack,
        objective=objective,
        diagnostics=diagnostics,
    )
