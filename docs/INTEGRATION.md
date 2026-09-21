# External emulation integration

The default backend is a Python packet-queue simulator with SciPy/HiGHS optimization. The interfaces below define how to connect external emulation, RIC control, channel traces, and alternative solvers.

## Platform components

The paper uses UERANSIM 3.1, Open5GS 2.4, QuaDRiGa, and CPLEX 20.1. Their integration requires a common observation and scheduling interface.

The public [UERANSIM feature documentation](https://github.com/aligungr/UERANSIM/wiki/Feature-Set) states that PHY, MAC, RLC, and PDCP layers are not implemented. Use a radio/link model for per-RRB achievable rates, grant decisions, queue timing, and channel interaction, and exchange those values through the observation contract below.

## Required observation contract

An adapter must provide, for each simulation/control step:

| Field | Required meaning |
| --- | --- |
| Cell and UE IDs | Stable identifiers with explicit association changes |
| Timestamp and step duration | A common clock; distinguish simulator time from wall time |
| Per-RRB achievable rates | Cell/RRB/UE matrix before request and grant decisions, with units |
| Queue backlog | Queued payload, including a declared policy for overhead and retransmissions |
| HOL age | Age of the front queued packet, measured from a timestamp |
| Arrivals | New packet payload and arrival time, not nominal requested throughput |
| Delivered service | Actual successful payload and completion timestamps |
| Drops/loss | Counters and denominators for overflow, channel loss, and other drops |
| QoS/intent | Per-UE rate and delay bounds plus cell throughput and latency targets |

Normalize units at this boundary. The simulation engine uses Mbit, milliseconds, and Mbit/ms. External byte counters and bit/s rates must be converted before the solver or reward code consumes them.

## Required action contract

The learner emits integer **requests** per UE. An adapter must send these to a scheduler that makes final **grants** and reports the outcome. It must enforce all of the following:

- Each RRB has at most one served UE at a given step.
- Each UE receives at most its requested RRB count.
- The total grant count stays within the cell resource budget.
- Resource indices, numerology, timing, and UE membership are aligned with the observation matrix.

Returning the request count as the grant count without handling contention changes the modeled problem. Similarly, converting an optimization benchmark directly into a measured throughput bypasses queues, channel realization, packet loss, and protocol behavior.

The manuscript combines scheduling-request terminology with a downlink throughput definition. A real integration must resolve that direction and protocol mapping explicitly; the analytical request–grant abstraction is not itself a standards-compliant implementation of uplink SR or downlink scheduling signaling.

## Connecting channel traces

For a QuaDRiGa-based experiment, document the channel-generation version, scenario, carrier, antenna setup, UE positions, mobility, and seeds. The bridge also needs an explicit conversion from channel coefficients to achievable per-RRB rate, including interference, link adaptation, coding overhead, and retransmission assumptions.

The analytical simulator supplies rate samples directly. Replacing those samples with traces without validating dimensions, temporal correlation, capacity units, and trace reuse across methods is insufficient for a reproducible link experiment.

## Optimizer substitution

The optimizer uses SciPy/HiGHS. A CPLEX adapter would need to implement the same binary benchmark and then solve a **separate LP relaxation** for dual prices. Record:

- Variable bounds, resource constraints, slack weights, and unit conventions.
- Feasibility/optimality status, time limit, MIP gap, and returned incumbent.
- LP objective direction and capacity-dual sign convention.
- Handling of redundant constraints and non-unique dual solutions.

Different solvers may return different valid dual optima for the same degenerate LP. Compare primal objectives and constraint residuals as well as the resulting scalar price; bitwise equality of all dual entries is not a reliable correctness criterion.

## Control timing and accounting

The adapter must specify whether local inference, local training, aggregation, monitoring, and leader solving happen synchronously or asynchronously. It should record solve/inference latency, late updates, communication delays, stale observations, and price/model version identifiers. A logical 1 ms step in Python is not proof that the implementation meets a real-time deadline.

Measure real transport traffic if making signaling claims. The repository's model-payload accounting omits protocol headers, retransmissions, serialization, encryption, controller telemetry, and deployment-specific overhead.

## Suggested validation sequence

1. Replay a fixed trace through the analytical and external boundaries and compare units and timestamps.
2. Check resource/request invariants using deliberately overloaded requests.
3. Validate packet conservation, HOL aging, packet completion delay, and all loss counters.
4. Compare benchmark feasibility and LP pricing on a small deterministic instance.
5. Check event windows and cooldown behavior before enabling online learning.
6. Run paired independent seeds and archive resolved configurations, trace checksums, revisions, and raw logs.

This sequence describes validation to perform when an adapter is implemented. It is not a record that those integrations or experiments have already been completed.
