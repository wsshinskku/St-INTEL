"""Packet/FIFO analytical cells. No claim of a 3GPP PHY or external RAN emulator."""

from collections import deque
from dataclasses import dataclass

import numpy as np

TRAFFIC = ("eMBB", "URLLC", "mMTC")


@dataclass
class Packet:
    remaining: float  # Mbit
    original: float
    born_ms: float


def feasible_grants(rates, requests, backlog, hol, delay_limits, past_rate, tti_ms=1.0):
    """Deterministic rate/PF/urgency greedy scheduler, respecting request and queue caps."""
    rates = np.asarray(rates, dtype=float)
    requests = np.asarray(requests)
    if rates.ndim != 2 or requests.shape != (rates.shape[1],):
        raise ValueError("rates [rrbs,ues] and per-UE requests required")
    if (
        not np.isfinite(rates).all()
        or not np.isfinite(requests).all()
        or np.any(rates < 0)
        or np.any(requests < 0)
        or np.any(requests != np.floor(requests))
    ):
        raise ValueError("nonnegative rates and integer requests required")
    remaining_requests = requests.astype(int).copy()
    remaining_queue = np.asarray(backlog, dtype=float).copy()
    allocation = np.zeros_like(rates, dtype=int)
    urgency = 1 + np.minimum(np.asarray(hol) / np.asarray(delay_limits), 10)
    for n in range(len(rates)):
        eligible = (remaining_requests > 0) & (remaining_queue > 1e-12) & (rates[n] > 0)
        if not eligible.any():
            continue
        score = rates[n] / np.maximum(past_rate, 1e-7) * urgency
        u = int(np.argmax(np.where(eligible, score, -np.inf)))
        allocation[n, u] = 1
        remaining_requests[u] -= 1
        remaining_queue[u] -= rates[n, u] * tti_ms
    return allocation


class Cell:
    def __init__(self, config, mix, index, seed):
        self.config, self.index, self.seed = config, index, seed
        self.types = np.repeat(np.arange(3), mix)
        self.ues = len(self.types)
        self.minimum_rates = np.asarray(config.minimum_rates_mbps)[self.types] / 1000
        self.delay_limits = np.asarray(config.delay_limits_ms)[self.types]
        self.queues = [deque() for _ in self.types]
        self.past_rate = np.full(self.ues, config.state_rate_scale)
        self.tick = 0
        profile_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 991]))
        self.channel_profile = profile_rng.uniform(0.6, 1.4, self.ues)
        self._prepare_tick()

    def backlog(self):
        return np.array([sum(packet.remaining for packet in queue) for queue in self.queues])

    def hol(self, at_ms=None):
        now = self.tick * self.config.tti_ms if at_ms is None else at_ms
        return np.array(
            [max(0.0, now - queue[0].born_ms) if queue else 0.0 for queue in self.queues]
        )

    def _prepare_tick(self):
        c = self.config
        now = self.tick * c.tti_ms
        self.shock = c.scenario == "unstable" and any(
            start <= now < start + c.shock_duration_ms for start in c.shock_starts_ms
        )
        # RNG keyed by tick: different policies/warm-rollouts cannot perturb future exogenous draws.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.index, self.tick]))
        sigma = c.fading_sigma * (np.sqrt(c.shock_variance_multiplier) if self.shock else 1)
        fade = rng.lognormal(-0.5 * sigma * sigma, sigma, size=(c.rrbs, self.ues))
        efficiency = np.minimum(8.0, c.spectral_efficiency * self.channel_profile[None] * fade)
        self.rates = c.rrb_bandwidth_hz * efficiency / 1e9  # bit/s -> Mbit/ms
        self.loss_draw = rng.random(self.ues)
        self.arrivals = np.zeros(self.ues)
        self.arrival_packets = np.zeros(self.ues, dtype=int)
        self.overflow_packets = np.zeros(self.ues, dtype=int)
        for u, kind in enumerate(self.types):
            size = c.packet_bytes[kind] * 8 / 1e6
            if kind == 0:
                per_tick = c.embb_offered_mbps / 1000 * c.tti_ms / size
                count = int(
                    np.floor((self.tick + 1) * per_tick + 1e-9)
                    - np.floor(self.tick * per_tick + 1e-9)
                )
            elif kind == 1:
                count = int(
                    np.floor((self.tick + 1) * c.tti_ms / c.urllc_period_ms + 1e-9)
                    - np.floor(self.tick * c.tti_ms / c.urllc_period_ms + 1e-9)
                )
            else:
                count = int(rng.poisson(c.mmtc_packets_per_second * c.tti_ms / 1000))
            self.arrivals[u] = count * size
            self.arrival_packets[u] = count
            occupied = sum(p.remaining for p in self.queues[u])
            for _ in range(count):
                if occupied + size > c.queue_limit_mbit + 1e-12:
                    self.overflow_packets[u] += 1
                else:
                    self.queues[u].append(Packet(size, size, now))
                    occupied += size

    def state(self, price):
        c = self.config
        return np.clip(
            np.column_stack(
                (
                    self.rates.mean(0) / c.state_rate_scale,
                    self.backlog() / c.queue_limit_mbit,
                    self.hol() / self.delay_limits,
                    np.full(self.ues, price / c.state_price_scale),
                )
            ),
            0,
            10,
        ).astype(np.float32)

    def step(self, requests, price):
        c = self.config
        requests = np.asarray(requests)
        if not np.isfinite(price) or price < 0:
            raise ValueError("price must be finite and nonnegative")
        if np.any(requests > c.request_cap):
            raise ValueError("request exceeds configured UE action cap")
        backlog_before = self.backlog()
        grants = feasible_grants(
            self.rates,
            requests,
            backlog_before,
            self.hol(),
            self.delay_limits,
            self.past_rate,
            c.tti_ms,
        )
        capacity = (grants * self.rates).sum(0) * c.tti_ms
        served = np.minimum(capacity, backlog_before)
        successful, completed = np.zeros(self.ues), np.zeros(self.ues, dtype=int)
        lost = self.overflow_packets.copy()
        dropped = self.overflow_packets * np.asarray(c.packet_bytes)[self.types] * 8 / 1e6
        delay_sum = np.zeros(self.ues)
        loss_probability = c.shock_loss if self.shock else c.base_packet_loss
        for u, queue in enumerate(self.queues):
            remaining = served[u]
            failed = self.loss_draw[u] < loss_probability
            while queue and remaining > 1e-12:
                packet = queue[0]
                amount = min(packet.remaining, remaining)
                packet.remaining -= amount
                remaining -= amount
                if packet.remaining <= 1e-12:
                    queue.popleft()
                    if failed:
                        lost[u] += 1
                        dropped[u] += packet.original
                    else:
                        completed[u] += 1
                        successful[u] += packet.original
                        delay_sum[u] += (self.tick + 1) * c.tti_ms - packet.born_ms
        # Goodput counts complete successfully delivered packet payload, including prior partial service.
        realized = served / c.tti_ms
        post_hol = self.hol((self.tick + 1) * c.tti_ms)
        violation = (
            np.maximum(0, self.minimum_rates - realized) / self.minimum_rates
            + np.maximum(0, post_hol - self.delay_limits) / self.delay_limits
        )
        reward = np.log1p(realized) - price * requests - c.reward_alpha * violation
        self.past_rate = 0.95 * self.past_rate + 0.05 * realized
        transition = dict(
            rates=self.rates.copy(),
            backlog=backlog_before,
            arrivals=self.arrivals.copy() / c.tti_ms,
            offered=self.arrivals.copy(),
            offered_packets=self.arrival_packets.copy(),
            delivered=successful,
            service=served,
            hol=post_hol,
            reward=reward,
            grants=grants,
            completed=completed,
            lost=lost,
            dropped=dropped,
            delay_sum=delay_sum,
            requests=requests.astype(int),
            shock=self.shock,
        )
        self.tick += 1
        self._prepare_tick()
        return transition


class Network:
    def __init__(self, config, seed):
        self.config = config
        self.cells = [Cell(config, mix, i, seed) for i, mix in enumerate(config.cell_mix)]

    def states(self, prices):
        if len(prices) != len(self.cells):
            raise ValueError("one scalar price per cell required")
        return [cell.state(prices[i]) for i, cell in enumerate(self.cells)]

    def step(self, requests, prices):
        if len(requests) != len(self.cells) or len(prices) != len(self.cells):
            raise ValueError("one request array and price per cell required")
        return [
            cell.step(action, prices[i])
            for i, (cell, action) in enumerate(zip(self.cells, requests))
        ]
