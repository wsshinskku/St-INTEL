# Method and equation mapping

This document maps the St-INTEL paper to its official implementation. Equation numbers refer to the paper; the following sections specify the simulation dynamics, numerical conventions, and comparison methods.

## Control hierarchy

1. The environment supplies per-RRB achievable rates, UE packet queues, and exogenous arrivals.
2. The non-RT leader averages these observations and solves a binary benchmark problem plus its LP relaxation.
3. The leader turns the benchmark into reference requests and averages RRB-capacity dual prices into one cell-level price.
4. Each UE chooses an integer request. The gNB issues grants that respect resource exclusivity and request bounds.
5. Actual served traffic changes packet queues, rewards, and intent measurements.
6. Local DDQN learning, cell aggregation, cross-cell aggregation, monitoring, and leader optimization run on their configured clocks.

These are logical simulation clocks in one Python process. The 1 ms TTI advances the simulated service timeline; wall-clock computation time is measured separately.

## Units and observables

The internal queue unit is **Mbit**, time is **milliseconds**, and rates are **Mbit/ms**. Multiply Mbit/ms by 1,000 to report Mbps. A served amount is a rate multiplied by the TTI duration. Preserve these conversions when changing the environment.

Four distinct quantities appear in the method:

- Per-RRB achievable rate: the channel service capacity if that RRB is granted.
- Queue-service rate: payload removed through allocated service during the step; an empty queue cannot use its nominal radio capacity. This quantity enters the reward.
- Delivered goodput: original payload of successfully completed packets. Partial service is counted toward goodput only when a packet finishes, and packets that fail on completion do not contribute.
- Offered rate: generated traffic, used to define offered load and demand attainment.

HOL delay is the age of the first queued packet. Delivered-packet delay is measured from a packet's arrival to its completion. The two are reported separately; an empty queue has no waiting HOL packet. The optimization's delay proxy is a third quantity and must not be presented as measured packet latency.

## Leader benchmark: Eqs. (6)–(14)

For each cell, let `x[n,u]` denote the binary benchmark grant, `r[n,u]` the time-averaged rate, `Q[u]` the average backlog, and `a[u]` the average arrival rate. The leader maximizes

```text
sum(r * x)
  - w_sla * sum(rate_slack + backlog_slack)
  - w_intent * (throughput_slack + latency_slack)
```

subject to:

```text
sum_u x[n,u] <= 1                                      for each RRB
sum_n,u x[n,u] <= total_RRB_budget
sum_n r[n,u] x[n,u] + rate_slack[u] >= min_rate[u]
delay_limit[u] * sum_n r[n,u] x[n,u] + backlog_slack[u] >= Q[u]
sum_n,u r[n,u] x[n,u] + throughput_slack >= min_cell_throughput
mean_u(backlog_slack[u] / max(a[u], epsilon)) <= max_cell_delay + latency_slack
x[n,u] in {0,1}; all slack variables >= 0
```

The positive arrival floor is a numerical convention for UEs with no observed arrivals. Slack variables make the benchmark solvable when intents cannot all be satisfied. Feasibility of the radio allocation is therefore distinct from satisfaction of the soft intent constraints.

The manuscript adds slacks with different physical units in one weighted objective. This implementation retains that structure in the declared internal units; it does not silently rescale the individual slack terms to dimensionless values. Changing rate or time units changes the interpretation of the objective weights and of its prices. Document any such change in comparisons.

The solver is SciPy's HiGHS-backed mixed-integer optimizer. It replaces the manuscript's CPLEX setup. A returned feasible incumbent and a certified optimum are different outcomes; inspect the recorded solve status when interpreting the benchmark.

The MILP accepts a time-limited incumbent only after integrality and feasibility checks. LP pricing requires an optimal LP result. A failed solve raises an error rather than returning a heuristic allocation labeled as optimal or fabricating a price. Reported slack values are the minimum feasible values for the returned integer allocation.

### LP shadow prices: Eq. (15)

A separate LP uses the same objective and constraints with `0 <= x[n,u] <= 1`. The price for RRB `n` comes from the capacity constraint `sum_u x[n,u] <= 1`. SciPy solves a minimization form, so the capacity marginal is sign-converted to the maximization scarcity convention; numerical negative noise is clipped at zero. The broadcast scalar is the arithmetic mean of those RRB prices.

The binary allocation itself does not supply continuous LP dual variables. The code does not invent prices from MILP multipliers or substitute the global capacity multiplier for the per-RRB multiplier.

Explicit `x <= 1` bounds and redundant capacity constraints can make the relaxation dual solution non-unique. Some capacity marginals may legitimately be zero. The implementation keeps the literal relaxation and does not force prices to be positive. Prices are slow-timescale scarcity guidance, not a guarantee of an exact Stackelberg equilibrium for the learned policies.

## Request–grant separation: Eqs. (1) and (16)

A UE chooses a request from `0` through the configured maximum request. The scheduler then assigns each RRB to at most one UE and gives no UE more RRBs than requested. The final allocation can differ from the leader's benchmark because local requests and channel realizations change.

The scheduler visits RRBs in order and greedily chooses an eligible UE using achievable rate divided by an exponentially averaged past service rate, multiplied by a bounded HOL-urgency factor. It skips empty queues and updates the remaining request and queue budgets as resources are assigned. The rule combines achievable rate, past service, and HOL urgency.

Section 3 permits requests through `N`, while the experiment description caps them at **10**. The action space follows the capped experiment convention. The manuscript's benchmark MILP has no corresponding per-UE cap. Benchmark allocations can therefore request more than the action space allows; the teacher request is clipped to the configured maximum. The benchmark allocation and its resulting feasible teacher rollout must not be treated as identical.

The implemented reward is

```text
log(1 + queue_service_rate_Mbit_per_ms)
  - broadcast_price * requested_RRBs
  - alpha * (
      max(min_rate - queue_service_rate, 0) / max(min_rate, epsilon)
      + max(HOL_delay - delay_limit, 0) / max(delay_limit, epsilon)
    )
```

The aggregate penalty `Phi` combines normalized rate shortfall and normalized HOL excess as defined above. The price is charged for **requested**, not granted, RRBs.

Queue-service rate counts serviced payload before the packet-completion loss outcome. Reported goodput instead counts successfully completed packets and can include bytes serviced in previous TTIs. Consequently, the reward's throughput term and the evaluation's delivered-throughput metric need not coincide at a given step. This convention is explicit; neither quantity is the uncapped potential capacity of the granted RRBs.

Local state has four components: mean per-RRB rate, backlog, HOL delay, and the scalar broadcast price. State features are scaled respectively by the configured rate scale, queue limit, UE delay bound, and price scale, then clipped to `[0,10]`. Price scaling applies only to the observation; it does not rescale the raw price charged in the reward.

## Teacher replay and DDQN: Eq. (17)

Benchmark requests drive actual environment transitions during teacher rollout. A replay tuple therefore contains a state, the capped teacher request, the observed reward after feasible scheduling, the resulting next state, and its terminal marker. The implementation does not attach an invented reward or unrelated next state to an arbitrary teacher action.

During training, the rollout adds `warm_steps` samples **per UE at each leader refresh**, including initialization, periodic refreshes, and event refreshes. A copied cell environment preserves the main training trajectory while producing valid transitions under the new benchmark. Evaluation never collects teacher replay.

This rollout assumes access to the analytical simulator's dynamics. It is not a method for inferring counterfactual next states and rewards from arbitrary real-network telemetry. External integration needs its own validated simulator or logged teacher executions to supply such transitions.

DDQN selects the next action using the online network and evaluates that action using the target network:

```text
target = reward + gamma * (1 - done)
                     * Q_target(next_state, argmax_a Q_online(next_state, a))
```

Target synchronization, replay capacity, batch size, optimizer settings, exploration, and network width are configurable. The supplied short configurations reduce workload to exercise the pipeline on CPU.

The policy network is an MLP with four inputs, two hidden ReLU layers of `hidden_dim` units, and `request_cap + 1` action values. Training uses Adam, smooth-L1 TD loss, and gradient-norm clipping at 10. Epsilon decays linearly over `epsilon_decay_steps`. `target_sync_steps` counts optimizer updates, which can be less frequent than environment TTIs.

## Federated coordination: Eqs. (18)–(19)

UEs train locally using replay transitions. Their DDQN loss includes a proximal penalty toward the preceding federated reference:

```text
DDQN_loss + mu / 2 * sum_parameters ||local - reference||^2
```

Cell-level averaging weights UEs uniformly. A slower cross-cell stage weights cell models by their UE counts. These are different clocks. The configured aggregation weights are uniform within cells and proportional to UE count across cells. On a federated broadcast, the online model, target model, and proximal reference all receive the shared weights, and Adam state is reset to avoid retaining momentum for overwritten parameters.

Raw queue and channel trajectories are not aggregated as model parameters. This design alone is not a formal privacy guarantee; the implementation does not add differential privacy, secure aggregation, or encrypted transport.

## Periodic and event refresh: Eqs. (20)–(21)

Monitoring averages cell throughput and UE HOL delay over a recent window and tracks packet loss. A refresh is requested when **any** condition is true:

```text
abs(measured_throughput - throughput_target) > throughput_tolerance
abs(measured_HOL_delay - latency_target) > latency_tolerance
packet_loss_rate > loss_tolerance
```

This is the literal absolute-deviation event rule in Eq. (21). It can fire when throughput is above its target or latency is below its bound. It is not silently replaced by a one-sided violation detector. Periodic refresh remains active independently of the event monitor, and the cooldown bounds event-driven solve frequency.

## Evaluation and interpretation

Training produces a saved policy. A separate evaluation environment disables exploration, gradient updates, teacher replay collection, and federated weight updates. For methods that use the leader, periodic/event optimization can still update prices and benchmark guidance during evaluation: the learned policy is frozen, the controller's measurements are not.

Metrics are computed from the simulation logs. Throughput attainment, SLA, HOL delay, delivered-packet latency, resource efficiency, and control activity answer different questions. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the experiment protocol and limitations.

## Source map

| Source | Responsibility |
| --- | --- |
| [`config.py`](../src/st_intel/config.py) | Validated configuration, units, method names |
| [`leader.py`](../src/st_intel/leader.py) | Binary benchmark, LP relaxation, capacity prices |
| [`environment.py`](../src/st_intel/environment.py) | Packet queues, rate process, request–grant scheduler, reward |
| [`learning.py`](../src/st_intel/learning.py) | DDQN, replay, FedProx, parameter averaging |
| [`control.py`](../src/st_intel/control.py) | Statistics windows, periodic/event refresh and cooldown |
| [`runner.py`](../src/st_intel/runner.py) | Teacher rollout, logical clocks, federation, phase logs and checkpoints |
| [`metrics.py`](../src/st_intel/metrics.py) | Goodput, HOL/packet delay, SLA and seed-level uncertainty |
| [`cli.py`](../src/st_intel/cli.py) | Run, suite, evaluate and cross-run summary commands |
