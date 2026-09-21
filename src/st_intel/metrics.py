"""Measured packet outcomes and seed-level uncertainty, without fabricated benchmarks."""

import numpy as np
from scipy.stats import t, ttest_rel

from .environment import TRAFFIC


class Metrics:
    def __init__(self, network):
        self.config = network.config
        self.types = [cell.types.copy() for cell in network.cells]
        self.minimum = [cell.minimum_rates.copy() for cell in network.cells]
        self.limits = [cell.delay_limits.copy() for cell in network.cells]
        self.steps = 0
        self.totals = [
            {
                key: np.zeros(cell.ues)
                for key in (
                    "offered",
                    "offered_packets",
                    "delivered",
                    "hol",
                    "completed",
                    "lost",
                    "dropped",
                    "delay_sum",
                    "reward",
                    "service",
                )
            }
            for cell in network.cells
        ]

    def add(self, transitions):
        self.steps += 1
        for totals, transition in zip(self.totals, transitions):
            for key in totals:
                totals[key] += transition[key]

    def report(self):
        c = self.config
        duration_ms = self.steps * c.tti_ms
        if self.steps == 0:
            raise ValueError("cannot report an empty run")

        def summarize(groups, minimum, limits, bandwidth):
            values = {key: np.concatenate([g[key] for g in groups]) for key in groups[0]}
            minimum, limits = np.concatenate(minimum), np.concatenate(limits)
            good, offered = values["delivered"].sum(), values["offered"].sum()
            completed, lost = values["completed"].sum(), values["lost"].sum()
            if len(minimum) == 0:
                return None
            return dict(
                throughput_mbps=float(good / duration_ms * 1000),
                throughput_achievement_pct=float(100 * good / offered) if offered > 0 else None,
                spectral_efficiency_bps_hz=float(good * 1e9 / duration_ms / bandwidth),
                sla_satisfaction_pct=float(
                    100
                    * np.mean(
                        (values["delivered"] / duration_ms >= minimum)
                        & (values["hol"] / self.steps <= limits)
                    )
                ),
                mean_hol_ms=float(np.mean(values["hol"] / self.steps)),
                delivered_packet_latency_ms=float(values["delay_sum"].sum() / completed)
                if completed
                else None,
                loss_rate=float(lost / (lost + completed)) if lost + completed else 0.0,
                offered_mbit=float(offered),
                delivered_mbit=float(good),
                served_mbit=float(values["service"].sum()),
                dropped_mbit=float(values["dropped"].sum()),
                pending_mbit_end=float(max(0, offered - good - values["dropped"].sum())),
                pending_packets_end=int(values["offered_packets"].sum() - completed - lost),
                mean_reward=float(values["reward"].mean() / self.steps),
                delivered_packets=int(completed),
                lost_packets=int(lost),
                ues=len(minimum),
            )

        bandwidth = c.rrbs * c.rrb_bandwidth_hz
        cells = {
            name: summarize([total], [minimum], [limit], bandwidth)
            for name, total, minimum, limit in zip(
                c.cell_names, self.totals, self.minimum, self.limits
            )
        }
        traffic = {}
        for kind, name in enumerate(TRAFFIC):
            traffic[name] = summarize(
                [
                    {k: v[types == kind] for k, v in totals.items()}
                    for totals, types in zip(self.totals, self.types)
                ],
                [m[t == kind] for m, t in zip(self.minimum, self.types)],
                [m[t == kind] for m, t in zip(self.limits, self.types)],
                bandwidth * len(cells),
            )
        return dict(
            duration_ms=duration_ms,
            network=summarize(self.totals, self.minimum, self.limits, bandwidth * len(cells)),
            cells=cells,
            traffic=traffic,
            metric_notes="Goodput counts successful completed packets; HOL includes empty-queue zeros; packet latency weights delivered packets.",
        )


def interval(values):
    x = np.asarray(values, dtype=float)
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("finite nonempty seed values required")
    mean = float(x.mean())
    if len(x) == 1:
        return dict(n=1, mean=mean, std=None, ci95=None)
    std = float(x.std(ddof=1))
    margin = float(t.ppf(0.975, len(x) - 1) * std / np.sqrt(len(x)))
    return dict(n=len(x), mean=mean, std=std, ci95=[mean - margin, mean + margin])


def paired(first, second):
    seeds = sorted(first.keys() & second.keys())
    if len(seeds) < 2:
        return None
    difference = np.array([first[s] - second[s] for s in seeds])
    if np.all(difference == 0):
        p = 1.0
    elif np.ptp(difference) == 0:
        p = 0.0
    else:
        p = float(ttest_rel([first[s] for s in seeds], [second[s] for s in seeds]).pvalue)
    return dict(
        seeds=seeds,
        difference=interval(difference),
        p_value=p,
        test="two-sided paired t-test, unadjusted",
    )
