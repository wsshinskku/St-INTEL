"""Units and configuration for the St-INTEL simulation environment."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

METHODS = ("st-intel", "ddqn", "fl-rl", "milp", "no-price", "no-warmstart", "no-fl", "no-events")


@dataclass
class Config:
    seed: int = 1
    method: str = "st-intel"
    scenario: str = "stable"
    device: str = "cpu"
    threads: int = 2
    # Each entry is [eMBB, URLLC, mMTC] UE counts for one cell.
    cell_mix: list = field(default_factory=lambda: [[2, 1, 1], [1, 2, 1]])
    cell_names: list = field(default_factory=lambda: ["Commercial", "Education"])
    rrbs: int = 12
    request_cap: int = 6
    tti_ms: float = 1.0
    train_steps: int = 160
    evaluation_steps: int = 120
    episode_steps: int = 20
    cell_fl_steps: int = 40
    global_fl_steps: int = 80
    monitor_steps: int = 20
    milp_interval_steps: int = 80
    cooldown_steps: int = 40
    stats_window_steps: int = 40
    event_window_steps: int = 20
    warm_steps: int = 16
    train_every: int = 4
    replay_capacity: int = 2048
    batch_size: int = 16
    hidden_dim: int = 32
    learning_rate: float = 0.0001
    gamma: float = 0.99
    target_sync_steps: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 160
    fedprox_mu: float = 0.1
    sla_weight: float = 10.0
    intent_weight: float = 10.0
    reward_alpha: float = 1.0
    throughput_target_mbps: float = 5.0
    latency_target_ms: float = 10.0
    minimum_rates_mbps: list = field(default_factory=lambda: [2.0, 1.5, 0.002])
    delay_limits_ms: list = field(default_factory=lambda: [30.0, 5.0, 100.0])
    embb_offered_mbps: float = 2.0
    urllc_period_ms: float = 0.125
    mmtc_packets_per_second: float = 1.0
    packet_bytes: list = field(default_factory=lambda: [1500, 32, 256])
    queue_limit_mbit: float = 2.0
    rrb_bandwidth_hz: float = 180000.0
    spectral_efficiency: float = 4.0
    fading_sigma: float = 0.2
    base_packet_loss: float = 0.0
    throughput_tolerance_fraction: float = 0.05
    latency_tolerance_fraction: float = 0.1
    packet_loss_threshold: float = 0.01
    shock_starts_ms: list = field(default_factory=lambda: [40.0, 100.0])
    shock_duration_ms: float = 20.0
    shock_loss: float = 0.2
    shock_variance_multiplier: float = 1.5
    state_rate_scale: float = 0.001
    state_price_scale: float = 0.1
    milp_time_limit: float = 5.0
    milp_gap: float = 0.01
    arrival_floor: float = 1e-9

    def validate(self):
        if self.method not in METHODS or self.scenario not in ("stable", "unstable"):
            raise ValueError("invalid method or scenario")
        if not self.cell_mix or len(self.cell_names) != len(self.cell_mix):
            raise ValueError("one name per nonempty cell required")
        if len(set(self.cell_names)) != len(self.cell_names):
            raise ValueError("cell names must be unique")
        for mix in self.cell_mix:
            if len(mix) != 3 or any(not isinstance(v, int) or v < 0 for v in mix) or sum(mix) < 1:
                raise ValueError("cell_mix entries must contain three nonnegative integer counts")
        for key in (
            "threads",
            "rrbs",
            "request_cap",
            "train_steps",
            "evaluation_steps",
            "episode_steps",
            "cell_fl_steps",
            "global_fl_steps",
            "monitor_steps",
            "milp_interval_steps",
            "cooldown_steps",
            "stats_window_steps",
            "event_window_steps",
            "train_every",
            "replay_capacity",
            "batch_size",
            "hidden_dim",
            "target_sync_steps",
            "epsilon_decay_steps",
        ):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if (
            not isinstance(self.warm_steps, int)
            or self.warm_steps < 0
            or self.request_cap > self.rrbs
            or self.batch_size > self.replay_capacity
        ):
            raise ValueError("invalid warm steps, request cap, or replay batch size")
        for key in (
            "tti_ms",
            "learning_rate",
            "throughput_target_mbps",
            "latency_target_ms",
            "queue_limit_mbit",
            "rrb_bandwidth_hz",
            "spectral_efficiency",
            "state_rate_scale",
            "state_price_scale",
            "milp_time_limit",
            "arrival_floor",
            "urllc_period_ms",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in (
            "gamma",
            "epsilon_start",
            "epsilon_end",
            "base_packet_loss",
            "shock_loss",
            "packet_loss_threshold",
            "milp_gap",
        ):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0,1]")
        for key in ("minimum_rates_mbps", "delay_limits_ms", "packet_bytes"):
            values = getattr(self, key)
            if len(values) != 3 or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"{key} requires three finite positive values")
        if self.milp_gap >= 1:
            raise ValueError("milp_gap must be less than one")
        for key in (
            "fedprox_mu",
            "sla_weight",
            "intent_weight",
            "reward_alpha",
            "fading_sigma",
            "throughput_tolerance_fraction",
            "latency_tolerance_fraction",
            "embb_offered_mbps",
            "mmtc_packets_per_second",
            "shock_duration_ms",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if (
            not isinstance(self.seed, int)
            or self.seed < 0
            or not math.isfinite(self.shock_variance_multiplier)
            or self.shock_variance_multiplier < 1
            or any(not math.isfinite(v) or v < 0 for v in self.shock_starts_ms)
        ):
            raise ValueError("invalid seed or shock settings")
        return self

    def as_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=None, **overrides):
        values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values).validate()
