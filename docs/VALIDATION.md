# Validation record

Validated on 2026-09-21 using Windows, Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, and PyTorch 2.14.0+cpu (CPU).

## Automated checks

`pytest -q`: **117 passed**. `ruff check .` and `ruff format --check .`: passed.

- 31 optimization checks include exhaustive tiny MILP oracles, finite-difference LP price signs, degenerate duals, zero arrivals, and checked solver failures/incumbents.
- 36 environment checks cover feasible grants, FIFO service, timestamp delay, conservation, overflow/loss, shock variance/timing, seeded randomness, and input validation.
- 29 learning/control checks cover DDQN targets, actual FedProx gradients, target synchronization, model averaging/broadcast, event thresholds/cooldown, and unit conversion.
- 15 integration/metric checks cover teacher isolation, frozen evaluation state, checkpoint round trips, all component ablations, metric denominators, packet conservation, and seed-level statistics.
- 6 CLI/configuration checks cover summaries, paired seeds, duplicate/mismatched runs, absent metrics, and supplied configurations.

GitHub Actions repeats lint, format, tests, a complete smoke run, and checkpoint evaluation on Python 3.10 and 3.12. See the [workflow runs](https://github.com/wsshinskku/St-INTEL/actions).

## Executed experiments

The following commands completed successfully; paths here are clean-directory equivalents of the validation runs:

```bash
python -m st_intel run --config configs/smoke.json --output runs/stable
python -m st_intel evaluate --checkpoint runs/stable/final.pt --output runs/reloaded
python -m st_intel run --config configs/smoke.json --scenario unstable --output runs/unstable
python -m st_intel suite --config configs/smoke.json --output runs/suite --methods st-intel ddqn fl-rl milp --seeds 1 2 3 4 5
```

The smoke configuration contains two cells/eight UEs, 160 training TTIs, and 120 fresh evaluation TTIs per run. The suite executes 20 independently trained method/seed combinations as a pipeline check.

| Method | Evaluation goodput mean (Mbps) | Seed-level Student-t 95% interval |
| --- | ---: | ---: |
| `st-intel` | 10.790507 | [9.542521, 12.038492] |
| `ddqn` | 10.961120 | [9.667660, 12.254580] |
| `fl-rl` | 10.867520 | [9.618726, 12.116314] |
| `milp` | 10.450080 | [9.918581, 10.981579] |

The [machine-readable smoke summary](validation/smoke-suite.json) contains the resolved common configuration, eight measured metrics, and paired comparisons. P-values are unadjusted; no significance claim is made.

Reloading the seed-1 checkpoint reproduced every network metric exactly in this environment. Its stable evaluation goodput was 10.059733 Mbps. The unstable smoke case completed at 9.098933 Mbps with disposition-based packet loss 0.082623; pending packets are excluded from that loss denominator. These are computed analytical-simulator outputs.

## Configuration coverage

- Local experiment coverage: two cells/eight UEs with the smoke configuration.
- Full-size configuration: 400 UEs, 600-second training, and 600-second held-out evaluation; full-size execution remains pending.
- Tested execution: CPU Python simulation and checkpoint evaluation. External emulation and GPU runs require separate validation.