"""Command-line experiments and paired, seed-level result summaries."""

import argparse
import glob
import json
from pathlib import Path

from .config import METHODS, Config
from .metrics import interval, paired
from .runner import evaluate, run, write_json

METRICS = (
    "throughput_mbps",
    "throughput_achievement_pct",
    "spectral_efficiency_bps_hz",
    "sla_satisfaction_pct",
    "mean_hol_ms",
    "delivered_packet_latency_ms",
    "loss_rate",
    "mean_reward",
)


def summarize(paths, metric=None):
    records = []
    for pattern in paths:
        matches = sorted(glob.glob(str(pattern)))
        if not matches:
            raise FileNotFoundError(f"no result files match {pattern}")
        records.extend(json.loads(Path(p).read_text(encoding="utf-8")) for p in matches)
    if not records:
        raise ValueError("at least one run metrics file is required")
    groups, seen, comparison_config = {}, set(), None
    for record in records:
        method, seed = record["method"], record["seed"]
        key = method, seed
        if key in seen:
            raise ValueError(f"duplicate method/seed pair: {key}")
        seen.add(key)
        config = {
            k: v
            for k, v in record["config"].items()
            if k not in ("method", "seed", "device", "threads")
        }
        if comparison_config is not None and comparison_config != config:
            raise ValueError("cannot compare different experiment configurations or scenarios")
        comparison_config = config
        if record["evaluation"]["environment_seed"] != seed + 100000:
            raise ValueError("evaluation seed does not match the paired run protocol")
        groups.setdefault(method, {})[seed] = record["evaluation"]["network"]
    summaries, comparisons = {}, {}
    for metric_name in [metric] if metric else METRICS:
        values = {
            method: {
                s: data[metric_name] for s, data in seeds.items() if data[metric_name] is not None
            }
            for method, seeds in groups.items()
        }
        summaries[metric_name] = {
            method: dict(
                seeds=sorted(samples),
                statistics=interval(list(samples.values())) if samples else None,
            )
            for method, samples in values.items()
        }
        if "st-intel" in values:
            comparisons[metric_name] = {
                method: paired(values["st-intel"], samples)
                for method, samples in values.items()
                if method != "st-intel"
            }
    return dict(
        phase="frozen_policy_evaluation",
        configuration=comparison_config,
        metrics=summaries,
        paired_st_intel_minus_control=comparisons,
        notes="Student-t 95% intervals across independent seeds. Paired tests use shared seeds only; p-values are unadjusted.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="St-INTEL optimization and federated learning experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "suite"):
        p = sub.add_parser(command)
        p.add_argument("--config", type=Path)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--scenario", choices=("stable", "unstable"))
        p.add_argument("--device")
        if command == "run":
            p.add_argument("--method", choices=METHODS)
            p.add_argument("--seed", type=int)
        else:
            p.add_argument(
                "--methods",
                nargs="+",
                choices=METHODS,
                default=["st-intel", "ddqn", "fl-rl", "milp"],
            )
            p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, help="Direct evaluation environment seed (no offset added)")
    p.add_argument("--device")
    p = sub.add_parser("summarize")
    p.add_argument("paths", nargs="+", help="Root run metrics.json files or quoted glob patterns")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--metric", choices=METRICS)
    args = parser.parse_args(argv)
    if args.command == "run":
        config = Config.load(
            args.config,
            method=args.method,
            seed=args.seed,
            scenario=args.scenario,
            device=args.device,
        )
        report = run(config, args.output)
        print(json.dumps(report["evaluation"]["network"], indent=2))
    elif args.command == "suite":
        if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
            parser.error("methods and seeds must be unique")
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("suite output must be new or empty")
        files = []
        for method in args.methods:
            for seed in args.seeds:
                config = Config.load(
                    args.config,
                    method=method,
                    seed=seed,
                    scenario=args.scenario,
                    device=args.device,
                )
                destination = args.output / method / f"seed-{seed}"
                print(f"Running {method}, seed {seed}", flush=True)
                run(config, destination)
                files.append(destination / "metrics.json")
        write_json(args.output / "summary.json", summarize(files))
        print(f"Summary: {args.output / 'summary.json'}")
    elif args.command == "evaluate":
        report = evaluate(args.checkpoint, args.output, args.seed, args.device)
        print(json.dumps(report["network"], indent=2))
    else:
        write_json(args.output, summarize(args.paths, args.metric))
        print(f"Summary: {args.output}")


if __name__ == "__main__":
    main()
