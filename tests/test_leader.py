"""Independent small-instance and sensitivity checks for equations (6)--(15)."""

from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog

import st_intel.leader as leader
from st_intel.leader import LeaderInput, solve_leader


def example() -> LeaderInput:
    return LeaderInput(
        rates=np.array([[0.8, 1.2], [1.0, 0.3], [0.2, 0.9]]),
        backlog=np.array([2.2, 1.1]),
        arrivals=np.array([0.2, 0.4]),
        minimum_rates=np.array([1.5, 1.0]),
        delay_limits=np.array([1.0, 0.8]),
        throughput_target=3.0,
        latency_target=0.5,
        resource_budget=2,
    )


def brute_force_objective(problem, sla_weight, intent_weight, arrival_floor=1e-9):
    """Enumerate 'unassigned or UE' for each RRB without a solver matrix."""
    n, u = problem.rates.shape
    best = -np.inf
    for choices in product(range(-1, u), repeat=n):
        if sum(ue != -1 for ue in choices) > problem.resource_budget:
            continue
        service = np.zeros(u)
        for rrb, ue in enumerate(choices):
            if ue != -1:
                service[ue] += problem.rates[rrb, ue]
        rate_deficit = np.maximum(problem.minimum_rates - service, 0)
        backlog_deficit = np.maximum(problem.backlog - service * problem.delay_limits, 0)
        throughput_deficit = max(problem.throughput_target - service.sum(), 0)
        delay_excess = max(
            np.mean(backlog_deficit / np.maximum(problem.arrivals, arrival_floor))
            - problem.latency_target,
            0,
        )
        value = (
            service.sum()
            - sla_weight * (rate_deficit.sum() + backlog_deficit.sum())
            - intent_weight * (throughput_deficit + delay_excess)
        )
        best = max(best, value)
    return best


@pytest.mark.parametrize("budget", [0, 1, 2, 3])
@pytest.mark.parametrize("weights", [(0.0, 0.0), (2.0, 4.0)])
def test_milp_matches_exhaustive_oracle(budget, weights):
    problem = replace(example(), resource_budget=budget)
    result = solve_leader(problem, sla_weight=weights[0], intent_weight=weights[1], mip_gap=0)
    assert result.objective == pytest.approx(brute_force_objective(problem, *weights), abs=1e-7)
    assert np.isin(result.allocation, [0, 1]).all()
    assert (result.allocation.sum(axis=1) <= 1).all()
    assert result.allocation.sum() <= budget
    np.testing.assert_array_equal(result.requests, result.allocation.sum(axis=0))
    assert result.diagnostics["lp_objective"] >= result.objective - 1e-7
    assert result.diagnostics["optimality_proven"]


def test_reported_slacks_satisfy_all_qos_and_intent_constraints():
    problem = example()
    result = solve_leader(problem, mip_gap=0)
    service = (result.allocation * problem.rates).sum(axis=0)
    assert np.all(service + result.rate_slack >= problem.minimum_rates - 1e-9)
    assert np.all(service * problem.delay_limits + result.delay_slack >= problem.backlog - 1e-9)
    assert service.sum() + result.throughput_slack >= problem.throughput_target - 1e-9
    assert (
        np.mean(result.delay_slack / problem.arrivals)
        <= problem.latency_target + result.latency_slack + 1e-9
    )


def test_zero_arrivals_are_explicitly_floored_without_losing_backlog_penalty():
    problem = replace(example(), arrivals=np.zeros(2), resource_budget=0)
    result = solve_leader(problem, arrival_floor=0.01, mip_gap=0)
    assert result.diagnostics["arrival_floor_applied_count"] == 2
    np.testing.assert_allclose(result.delay_slack, problem.backlog)
    assert result.latency_slack == pytest.approx(
        problem.backlog.mean() / 0.01 - problem.latency_target
    )
    assert result.objective == pytest.approx(brute_force_objective(problem, 10, 10, 0.01))
    # The default floor is also accepted for a silent, empty traffic window.
    quiet = replace(problem, backlog=np.zeros(2), minimum_rates=np.zeros(2), throughput_target=0)
    assert solve_leader(quiet).objective == 0


def test_exclusivity_price_matches_lp_capacity_sensitivity():
    # Both x values are fractional, so x<=1 bounds are inactive; the budget is
    # loose. This avoids assuming a particular dual under degeneracy.
    problem = LeaderInput(
        rates=np.array([[1.0, 2.0]]),
        backlog=np.zeros(2),
        arrivals=np.ones(2),
        minimum_rates=np.array([0.6, 0.6]),
        delay_limits=np.ones(2),
        throughput_target=0.0,
        latency_target=0.0,
        resource_budget=2,
    )
    result = solve_leader(problem, mip_gap=0)
    assert result.rrb_prices[0] == pytest.approx(2.0)
    assert result.price == pytest.approx(result.rrb_prices.mean())
    assert result.diagnostics["resource_budget_price"] == pytest.approx(0)
    # Perturb the right hand side of the capacity inequality while retaining
    # all paper constraints, including literal variable upper bounds.
    model = leader._formulate(problem, 10, 10, 1e-9)
    epsilon = 1e-5
    perturbed_rhs = model.rhs.copy()
    perturbed_rhs[0] += epsilon
    perturbed = linprog(
        model.objective,
        A_ub=model.matrix,
        b_ub=perturbed_rhs,
        bounds=list(zip(np.zeros_like(model.upper), model.upper)),
        method="highs",
    )
    assert perturbed.success
    observed = (-perturbed.fun - result.diagnostics["lp_objective"]) / epsilon
    assert observed == pytest.approx(result.rrb_prices[0], rel=1e-5)


def test_dual_diagnostics_do_not_fabricate_nonzero_exclusivity_prices():
    problem = LeaderInput(
        np.array([[3.0]]), np.zeros(1), np.ones(1), np.zeros(1), np.ones(1), 0.0, 0.0, 1
    )
    result = solve_leader(problem)
    # All three capacities are redundant in this one-variable problem. Their
    # joint marginal is fixed, but the solver may distribute it differently.
    joint = (
        result.price
        + result.diagnostics["resource_budget_price"]
        + result.diagnostics["allocation_upper_bound_prices"][0][0]
    )
    assert joint == pytest.approx(3.0)
    assert result.price >= 0
    assert "nonunique" in result.diagnostics["dual_caveat"]


@pytest.mark.parametrize(
    "change",
    [
        {"rates": np.zeros((0, 2))},
        {"rates": np.array([[np.nan, 0]])},
        {"rates": np.array([[-1.0, 0]])},
        {"backlog": np.zeros(3)},
        {"arrivals": np.array([-1.0, 0])},
        {"latency_target": -1},
        {"resource_budget": -1},
        {"resource_budget": 1.5},
        {"resource_budget": True},
    ],
)
def test_invalid_physical_inputs_are_rejected(change):
    with pytest.raises(ValueError):
        solve_leader(replace(example(), **change))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sla_weight": -1},
        {"intent_weight": np.inf},
        {"arrival_floor": 0},
        {"time_limit": 0},
        {"mip_gap": 1},
    ],
)
def test_invalid_solver_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        solve_leader(example(), **kwargs)


def test_time_limited_feasible_incumbent_is_not_labeled_optimal(monkeypatch):
    problem = replace(example(), resource_budget=0)
    x = np.r_[
        np.zeros(6),
        problem.minimum_rates,
        problem.backlog,
        problem.throughput_target,
        np.mean(problem.backlog / problem.arrivals) - problem.latency_target,
    ]
    monkeypatch.setattr(
        leader,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(
            status=1, x=x, message="time limit", mip_gap=0.2, mip_dual_bound=-10.0
        ),
    )
    result = solve_leader(problem)
    assert not result.diagnostics["optimality_proven"]
    assert result.diagnostics["milp_termination"] == "limit_with_feasible_incumbent"


@pytest.mark.parametrize("bad_x", [None, np.full(12, np.nan), np.ones(12)])
def test_unusable_solver_incumbents_raise(monkeypatch, bad_x):
    monkeypatch.setattr(
        leader,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(status=1, x=bad_x, message="time limit"),
    )
    with pytest.raises(RuntimeError, match="MILP"):
        solve_leader(example())


def test_failed_lp_pricing_raises_instead_of_returning_placeholder_prices(monkeypatch):
    monkeypatch.setattr(
        leader,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(success=False, status=1, message="time limit"),
    )
    with pytest.raises(RuntimeError, match="LP pricing failed"):
        solve_leader(example())
