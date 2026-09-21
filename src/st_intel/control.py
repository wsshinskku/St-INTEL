"""Bounded windows and independently periodic/event-triggered leader refresh."""

from collections import deque

import numpy as np

from .leader import LeaderInput


class Monitor:
    def __init__(self, config):
        self.config = config
        self.records = deque(maxlen=max(config.stats_window_steps, config.event_window_steps))
        self.last_solve_step = None

    def add(self, transition):
        self.records.append(transition)

    def problem(self, cell):
        c = self.config
        records = list(self.records)[-c.stats_window_steps :]
        if records:
            rates = np.mean([r["rates"] for r in records], axis=0)
            backlog = np.mean([r["backlog"] for r in records], axis=0)
            arrivals = np.mean([r["arrivals"] for r in records], axis=0)
        else:
            rates, backlog, arrivals = cell.rates, cell.backlog(), cell.arrivals / c.tti_ms
        return LeaderInput(
            rates=rates,
            backlog=backlog,
            arrivals=arrivals,
            minimum_rates=cell.minimum_rates,
            delay_limits=cell.delay_limits,
            throughput_target=c.throughput_target_mbps / 1000,
            latency_target=c.latency_target_ms,
            resource_budget=c.rrbs,
        )

    def indicators(self):
        records = list(self.records)[-self.config.event_window_steps :]
        if not records:
            return None
        throughput = np.mean([r["delivered"].sum() / self.config.tti_ms for r in records])
        latency = np.mean([r["hol"].mean() for r in records])
        lost = sum(r["lost"].sum() for r in records)
        completed = sum(r["completed"].sum() for r in records)
        return dict(
            throughput_mbit_per_ms=float(throughput),
            hol_ms=float(latency),
            loss_rate=float(lost / max(1, lost + completed)),
        )

    def reason(self, step, events=True):
        c = self.config
        if self.last_solve_step is None:
            return "initial"
        if step > 0 and step % c.milp_interval_steps == 0:
            return "periodic"
        if not events or step % c.monitor_steps or len(self.records) < c.event_window_steps:
            return None
        if step - self.last_solve_step < c.cooldown_steps:
            return None
        observed = self.indicators()
        target = c.throughput_target_mbps / 1000
        # Literal Eq. (21), including deviations in the overperformance direction.
        if (
            abs(observed["throughput_mbit_per_ms"] - target)
            > c.throughput_tolerance_fraction * target
            or abs(observed["hol_ms"] - c.latency_target_ms)
            > c.latency_tolerance_fraction * c.latency_target_ms
            or observed["loss_rate"] > c.packet_loss_threshold
        ):
            return "event"
        return None

    def solved(self, step):
        self.last_solve_step = step
