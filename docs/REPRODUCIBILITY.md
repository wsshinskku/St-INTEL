# Reproduction scope and experiment protocol

## What a run establishes

This repository reconstructs the manuscript's control structure in a runnable Python environment. It can test implementation behavior and compare its own controlled variants. It does not include the original deployment, network traces, solver settings, trained models, or evaluation logs needed to reproduce the manuscript's numerical results.

The current publication record lists the manuscript as **under review**. A target journal, a PDF formatted with journal headers, or this code release does not establish acceptance.

## Starting with a small run

Use a fresh environment and record the Git revision and installed dependency versions:

```bash
python -m pip install -e ".[dev]"
python -m st_intel run --config configs/smoke.json --output runs/demo --seed 1
python -m st_intel evaluate --checkpoint runs/demo/final.pt --output runs/eval
pytest
```

The short run checks that optimization, learning, scheduling, measurement, checkpointing, and evaluation connect correctly. It is too short to establish learning convergence. See [VALIDATION.md](VALIDATION.md) for measured validation performed for this release.

## Manuscript settings and reference choices

The manuscript describes four cells with 100 UEs each and 600 seconds per run. Its service mix is:

| Cell | eMBB | URLLC | mMTC |
| --- | ---: | ---: | ---: |
| Commercial | 50 | 30 | 20 |
| Education | 40 | 40 | 20 |
| Residential | 20 | 30 | 50 |
| Industrial | 10 | 20 | 70 |

Known numerical settings include:

| Parameter | Manuscript value |
| --- | --- |
| TTI | 1 ms |
| Episode length | 40 TTIs |
| Cell aggregation | 25 episodes, approximately 1 s |
| Cross-cell aggregation | Every 3 s |
| Periodic MILP refresh | Every 10 s |
| Event observation and detection | 1 s window, every 1 s |
| Event cooldown | 5 s |
| Throughput tolerance | 5% of throughput target |
| Latency tolerance | 10% of latency target |
| Loss threshold | 0.01 |
| Maximum UE request | 10 RRBs/TTI in the experiment section |
| Replay capacity / batch | 100,000 / 64 |
| Learning rate / discount | 0.0001 / 0.99 |
| Target synchronization | Every 1,000 steps; the precise clock is interpreted in the reference implementation |
| Teacher warm-start | 500 samples |
| FedProx coefficient | 0.1 |
| Independent seeds | 5 |

These parameters alone do not determine a complete experiment. The original material leaves the RRB count/bandwidth assignment, detailed scheduler, network architecture, reward penalty scaling, operator intent values, several QoS targets, initialization and exploration schedules, and exact inter-component interfaces incompletely specified. The JSON configurations record concrete reference choices. Increasing UE count or duration does not convert the analytical simulator into the original emulation platform.

`configs/smoke.json` uses two cells/eight UEs and reduces duration and control intervals for a CPU check. `configs/paper_reference.json` records known manuscript settings with explicit reference choices for the missing values:

| Setting | Reference configuration | Provenance |
| --- | --- | --- |
| Cells / UEs | Four / 100 per cell | Manuscript population and traffic mix |
| RRBs / RRB bandwidth | 100 / 180 kHz, or 18 MHz per cell | Reconstruction choice |
| Per-UE minimum rates | 50 / 2.048 / 0.002048 Mbps for eMBB/URLLC/mMTC | Reference targets aligned with specified traffic rates |
| Per-UE delay bounds | 30 / 5 / 100 ms | eMBB/URLLC choices; manuscript gives mMTC 100 ms |
| Cell intent | 100 Mbps minimum throughput, 10 ms maximum mean delay | Reconstruction choice |
| DDQN hidden layers | Two layers of 128 ReLU units | Reconstruction choice |
| Training horizon | 600,000 steps at 1 ms | Manuscript's 600 s scale |
| Additional held-out evaluation | 600,000 steps at 1 ms | Reference evaluation protocol |

This full-scale configuration has **400 independent UE agents**, each with a replay capacity of **100,000 transitions**, and **1.2 million environment steps** across training and evaluation. It has not been run at full scale for this release. Runtime and memory can be substantial; replay storage, learner updates, optimization, and teacher refreshes all add cost.

### Traffic and channel differences

The manuscript describes full-buffer TCP eMBB with a 50 Mbps reporting target, periodic 32-byte URLLC packets every 125 microseconds, and 256-byte Poisson mMTC packets at one packet/second. A 1 ms analytical step must represent sub-TTI arrivals in a discrete update, and a finite analytical queue is not a TCP protocol stack. The simulator models offered packet traffic and service rather than recovering original TCP/UDP transport behavior.

The concrete reference choices are finite constant-rate eMBB packet arrivals, deterministic URLLC packet counts per step (eight at a 1 ms TTI with the manuscript period), and Poisson mMTC arrivals. All packets generated within a step share its timestamp, so sub-TTI latency is unresolved. A single per-UE/TTI Bernoulli outcome marks the packets completed in that step as delivered or lost. The implementation has no HARQ/RLC retransmission process; queue overflow also contributes to the recorded loss count.

The paper's QuaDRiGa configuration includes UMi-NLOS at 3.5 GHz, 4×4 MIMO, 32 dBm gNB transmit power, and a campus topology. No QuaDRiGa channel traces or scenario files are supplied. The repository's stochastic rate process is an analytical stand-in, not a propagation or link-level validation of those physical settings.

The unstable scenario injects fading/loss disturbances. Shock intervals are explicit in the JSON configuration; stochastic channel and loss realizations depend on the seed. They do not reconstruct the original run histories. The paper samples 6–10 five-second shocks with separated random start times. The reference configuration instead fixes eight five-second shocks starting at 60, 120, 180, 240, 300, 360, 420, and 480 seconds, with a 20% completion-loss probability and a 1.5× fading-variance multiplier. This fixed schedule is a reference choice to align the controls; random timing studies require generating and recording a schedule for each seed. Short smoke runs compress control timing and disturbance placement to exercise behavior.

### Numerical conventions that affect results

- Rates, queues, and time use Mbit/ms, Mbit, and ms internally. The leader's mixed-unit slack objective is retained as written. The weights and LP prices therefore depend on these units.
- The reward's unspecified QoS penalty uses normalized rate shortfall and HOL excess; the scalar price term itself is not normalized.
- The reward uses actual queued payload service before completion loss. Goodput metrics count only successfully completed packet payload, including any partial service from earlier TTIs.
- The LP relaxation uses literal `[0,1]` grant bounds. Dual degeneracy can yield zero per-RRB prices even when resources are used.
- MILP benchmark allocations are not UE-request-capped in the manuscript. Teacher requests are clipped to the action cap before actual rollouts.
- Training adds `warm_steps` teacher transitions per UE at every leader refresh, using a copied simulator. The interpretation of repeated warm-start insertion is an explicit reference choice; it is not a recovered original replay schedule.
- Eq. (21) uses absolute target deviations, including favorable deviations. The event detector follows that rule.
- The MILP backlog/arrival delay proxy, sampled HOL age, and completed-packet delay are distinct quantities.

The [method document](METHOD.md) gives the equations and explains these choices in more detail.

## Comparing methods and seeds

Use the same environment, horizon, traffic, radio-budget, and evaluation protocol for every method. Run stable and unstable scenarios separately. Pair seed identifiers across methods to support comparisons under matching exogenous generation rules; method-dependent queues and learning trajectories will still differ.

The analytical environment keys random draws by seed, cell, and step so policy actions and teacher-rollout length do not consume and shift the future exogenous random stream.

```bash
python -m st_intel suite --config configs/smoke.json --output runs/suite --methods st-intel ddqn fl-rl milp no-price no-warmstart no-fl no-events --seeds 1 2 3 4 5
```

`suite` is a repeatable experiment driver. The listed controls are implementations in this repository; they do not reproduce the manuscript's cited SG, SG+RL, FL, and FL+RL methods. Report their exact names and configuration instead of attaching the paper's baseline scores to them.

For uncertainty estimates, use one aggregate metric per independent run, then summarize across seeds. TTIs or UE observations within a run are correlated and must not be presented as independent seeds. A Student-t interval with five seeds uses four degrees of freedom; a one-seed result has no estimated confidence interval.

The suite stores each run as `<output>/<method>/seed-<seed>/` and writes an aggregate `summary.json`. Summaries use run-root `metrics.json` files and their held-out network measurements. Paired t-tests compare St-INTEL minus each control on shared seeds only; their p-values are unadjusted for multiple comparisons. The summarizer rejects duplicate method/seed pairs, mismatched configurations, and incompatible evaluation seeds.

## Interpreting measurements

| Measurement | Interpretation |
| --- | --- |
| Delivered throughput | Full payload of successfully completed packets, divided by measured duration |
| Offered-load attainment | Delivered bits relative to generated offered bits over the measured phase |
| Resource efficiency | Delivered bit rate divided by the configured cell bandwidth |
| SLA satisfaction | UE fraction meeting both average rate and average HOL-delay requirements |
| Mean HOL delay | Time-sampled age of each queue's front packet |
| Delivered-packet delay | Completion delay for packets actually delivered; pending/dropped packets are excluded |
| Packet loss rate | Completed/dropped outcomes: lost packets divided by lost plus delivered packets; still-pending packets are excluded |
| Solver activity | Benchmark refresh counts and reasons, separate from policy inference |
| Model communication | Parameter-payload accounting in the reference process, not measured A1/E2 or transport traffic |

An empty queue contributes zero to time-sampled HOL delay, while a UE with no completed packets has no measured delivered-packet delay. A low delivered-packet delay can coexist with queue growth or loss, so assess these measures together. Do not rename policy output, offered load, or nominal radio capacity as delivered throughput.

The manuscript discusses RL convergence using a persistent 5% band and FL convergence using model-norm change below 0.01. Short runs and local model-difference logs do not by themselves establish the manuscript's convergence episodes/rounds. Report a convergence result only with a prespecified horizon, smoothing window, persistence rule, and treatment of methods that never meet it.

`pending_mbit_end` counts the full original payload of packets still pending at the last measured boundary (including partially served packets); `served_mbit` separately counts radio service. Pending totals exclude arrivals prepared for the following unmeasured TTI.

The exported `global_convergence_round` is only the first global model-change crossing below 0.01, not proof that subsequent rounds remain stable. `rl_convergence_episodes` is left null rather than inferring convergence from an underspecified or short trace. Network and traffic-class efficiency divide goodput by the sum of all configured cell bandwidths; per-cell efficiency uses that cell's bandwidth.

## Saved artifacts and evaluation

Each run saves its resolved configuration, phase metrics, and a final checkpoint for policy evaluation. Keep those files together with the Git commit and dependency versions. Checkpoints produced by `run` can be evaluated with `evaluate`; they are not a claim of exact mid-run replay/RNG resume unless a resume command is explicitly provided.

`config.json` and the combined `metrics.json` sit at the run root beside `final.pt`. Both `train/` and `evaluation/` contain `metrics.json`, `leader.json`, `federation.json`, and `episodes.json`. A standalone `evaluate` call writes phase logs directly under its output directory. Its `metrics.json` is a phase report; the combined run-root `metrics.json` is the input intended for cross-run summaries.

Evaluation uses fresh environment state and frozen neural weights. Leader-based methods retain their operational benchmark and event updates, while local learning, exploration, and federated weight updates are disabled. Keep evaluation results separate from the experiences used to train the policies.

By default, the evaluation environment seed is `training_seed + 100000`. Loading the same checkpoint with the same evaluation seed intentionally repeats that evaluation. An explicitly different evaluation seed changes the holdout environment without retraining the policy. This is distinct from a multi-seed study in which each seed trains a separate policy.

## What is needed for manuscript-level reproduction

1. Original topology, traffic generator configuration, radio/channel traces, seeds, and UE association history.
2. The custom bridge between network emulation, link/MAC behavior, and RIC control, including scheduling-request/grant semantics.
3. Exact CPLEX models, solver limits, intent weights, network architecture, and training schedules.
4. Baseline source revisions and their method-specific hyperparameters.
5. Raw logs and metric scripts with consistent HOL, packet-delay, offered-load, and signaling definitions.

[INTEGRATION.md](INTEGRATION.md) describes those interface requirements. No results from the original manuscript are embedded as generated measurements in this repository.
