"""Integration checks for measured outcomes and isolated training/evaluation."""

import copy
import json

import numpy as np
import pytest
import torch

from st_intel.config import Config
from st_intel.environment import Network
from st_intel.metrics import Metrics, interval, paired
from st_intel.runner import evaluate, make_agents, run, run_phase, warm_replay


def tiny_config(**changes):
    values = dict(
        cell_mix=[[0, 1, 0]],
        cell_names=["tiny"],
        rrbs=2,
        request_cap=2,
        threads=1,
        train_steps=6,
        evaluation_steps=4,
        episode_steps=2,
        cell_fl_steps=2,
        global_fl_steps=4,
        monitor_steps=2,
        milp_interval_steps=4,
        cooldown_steps=2,
        stats_window_steps=2,
        event_window_steps=2,
        warm_steps=2,
        train_every=2,
        replay_capacity=64,
        batch_size=2,
        hidden_dim=8,
        target_sync_steps=2,
        epsilon_decay_steps=6,
    )
    values.update(changes)
    return Config(**values).validate()


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items() if key != "wall_seconds"}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


def transition(**values):
    record = {
        key: np.zeros(2)
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
    record.update({key: np.asarray(value, dtype=float) for key, value in values.items()})
    return record


def test_metrics_use_run_average_ue_sla_and_distinct_latency_denominators():
    settings = tiny_config(
        cell_mix=[[1, 1, 0]],
        tti_ms=2,
        minimum_rates_mbps=[1, 1, 0.002],
        delay_limits_ms=[2, 2, 100],
    )
    metrics = Metrics(Network(settings, seed=1))
    metrics.add(
        [
            transition(
                offered=[0.006, 0.008],
                offered_packets=[3, 2],
                delivered=[0.004, 0],
                hol=[0, 4],
                completed=[2, 0],
                lost=[1, 0],
                dropped=[0.002, 0],
                delay_sum=[4, 0],
                reward=[1, 3],
            )
        ]
    )
    metrics.add(
        [
            transition(
                offered=[0, 0],
                delivered=[0, 0.004],
                hol=[2, 4],
                completed=[0, 1],
                lost=[0, 1],
                dropped=[0, 0.004],
                delay_sum=[0, 7],
                reward=[3, 5],
            )
        ]
    )
    report = metrics.report()
    network = report["network"]
    assert report["duration_ms"] == 4
    assert network["throughput_mbps"] == pytest.approx(2)
    assert network["throughput_achievement_pct"] == pytest.approx(100 * 0.008 / 0.014)
    assert network["spectral_efficiency_bps_hz"] == pytest.approx(
        2e6 / (settings.rrbs * settings.rrb_bandwidth_hz)
    )
    # Both satisfy average throughput; only the first UE satisfies average HOL.
    assert network["sla_satisfaction_pct"] == 50
    assert network["mean_hol_ms"] == pytest.approx(2.5)
    assert network["delivered_packet_latency_ms"] == pytest.approx(11 / 3)
    assert network["loss_rate"] == pytest.approx(2 / 5)
    assert network["mean_reward"] == 3
    assert network["delivered_packets"] == 3
    assert network["lost_packets"] == 2
    assert network["dropped_mbit"] == pytest.approx(0.006)
    assert network["pending_mbit_end"] == pytest.approx(0)
    assert network["pending_packets_end"] == 0
    assert report["traffic"]["eMBB"]["sla_satisfaction_pct"] == 100
    assert report["traffic"]["URLLC"]["sla_satisfaction_pct"] == 0
    assert report["traffic"]["mMTC"] is None


def test_metrics_distinguish_undefined_achievement_and_packet_latency_from_zero():
    settings = tiny_config(cell_mix=[[1, 1, 0]])
    metrics = Metrics(Network(settings, seed=1))
    with pytest.raises(ValueError):
        metrics.report()
    metrics.add([transition()])
    network = metrics.report()["network"]
    assert network["throughput_mbps"] == 0
    assert network["throughput_achievement_pct"] is None
    assert network["delivered_packet_latency_ms"] is None
    assert network["sla_satisfaction_pct"] == 0


def test_real_environment_metrics_match_packet_conservation_and_network_bandwidth():
    settings = tiny_config(
        cell_mix=[[1, 1, 0], [0, 1, 1]],
        cell_names=["first", "second"],
        embb_offered_mbps=12,
        mmtc_packets_per_second=2000,
        queue_limit_mbit=0.012,
        base_packet_loss=0.2,
    )
    network = Network(settings, seed=1)
    metrics = Metrics(network)
    offered = delivered = dropped = served = 0.0
    for _ in range(10):
        results = network.step([[2, 2], [2, 2]], [0, 0])
        metrics.add(results)
        offered += sum(result["offered"].sum() for result in results)
        delivered += sum(result["delivered"].sum() for result in results)
        dropped += sum(result["dropped"].sum() for result in results)
        served += sum(result["service"].sum() for result in results)
    pending = 0.0
    for cell in network.cells:
        next_sizes = np.asarray(settings.packet_bytes)[cell.types] * 8 / 1e6
        next_accepted = cell.arrivals - cell.overflow_packets * next_sizes
        pending += sum(packet.original for queue in cell.queues for packet in queue)
        pending -= next_accepted.sum()
    assert offered == pytest.approx(delivered + dropped + pending)
    report = metrics.report()
    assert report["network"]["offered_mbit"] == pytest.approx(offered)
    assert report["network"]["delivered_mbit"] == pytest.approx(delivered)
    assert report["network"]["dropped_mbit"] == pytest.approx(dropped)
    assert report["network"]["served_mbit"] == pytest.approx(served)
    assert report["network"]["pending_mbit_end"] == pytest.approx(pending)
    pending_packets = sum(len(queue) for cell in network.cells for queue in cell.queues)
    pending_packets -= sum(
        (cell.arrival_packets - cell.overflow_packets).sum() for cell in network.cells
    )
    assert report["network"]["pending_packets_end"] == pending_packets
    assert sum(cell["throughput_mbps"] for cell in report["cells"].values()) == pytest.approx(
        report["network"]["throughput_mbps"]
    )
    assert report["network"]["spectral_efficiency_bps_hz"] == pytest.approx(
        delivered * 1e9 / 10 / (2 * settings.rrbs * settings.rrb_bandwidth_hz)
    )


def test_warm_teacher_generates_real_transitions_without_advancing_live_cell():
    settings = tiny_config()
    cell = Network(settings, seed=1).cells[0]
    expected_teacher = copy.deepcopy(cell)
    agents = make_agents(settings)[0]
    before = copy.deepcopy(cell)
    expected = []
    for _ in range(3):
        state = expected_teacher.state(0.1)
        outcome = expected_teacher.step([2], 0.1)
        expected.append((state[0], outcome["reward"][0], expected_teacher.state(0.1)[0]))
    warm_replay(cell, agents, np.array([2]), 0.1, 3)
    assert cell.tick == before.tick == 0
    np.testing.assert_array_equal(cell.rates, before.rates)
    np.testing.assert_array_equal(cell.arrivals, before.arrivals)
    np.testing.assert_array_equal(cell.backlog(), before.backlog())
    assert [list(queue) for queue in cell.queues] == [list(queue) for queue in before.queues]
    replay = agents[0].replay
    assert replay.size == 3
    for index, (state, reward, next_state) in enumerate(expected):
        np.testing.assert_array_equal(replay.states[index], state)
        np.testing.assert_array_equal(replay.next_states[index], next_state)
        assert replay.actions[index] == 2
        assert replay.rewards[index] == pytest.approx(reward)
        assert replay.done[index] == 0


def test_frozen_evaluation_leaves_models_targets_anchors_replay_and_rng_unchanged(tmp_path):
    settings = tiny_config()
    agents = make_agents(settings)
    agent = agents[0][0]
    agent.replay.add(np.ones(4), 1, 2.0, np.zeros(4), False)
    agent.replay.add(np.zeros(4), 0, 1.0, np.ones(4), False)
    assert agent.update() is not None
    model = agent.cpu_state()
    target = copy.deepcopy(agent.target.state_dict())
    reference = [tensor.clone() for tensor in agent.reference]
    replay = copy.deepcopy(agent.replay)
    agent_rng = copy.deepcopy(agent.rng.bit_generator.state)
    updates = agent.updates
    report = run_phase(settings, agents, 4, 100001, False, tmp_path / "frozen")
    for name, tensor in agent.online.state_dict().items():
        assert torch.equal(tensor.cpu(), model[name])
    for name, tensor in agent.target.state_dict().items():
        assert torch.equal(tensor, target[name])
    assert all(torch.equal(a, b) for a, b in zip(reference, agent.reference))
    assert agent.updates == updates
    assert agent.rng.bit_generator.state == agent_rng
    assert agent.replay.size == replay.size
    assert agent.replay.position == replay.position
    assert agent.replay.rng.bit_generator.state == replay.rng.bit_generator.state
    for name in ("states", "next_states", "actions", "rewards", "done"):
        np.testing.assert_array_equal(
            getattr(agent.replay, name)[: replay.size], getattr(replay, name)[: replay.size]
        )
    assert report["phase"] == "frozen_policy_evaluation"
    assert report["communication_payload_bytes"]["model_upload"] == 0
    assert json.loads((tmp_path / "frozen" / "federation.json").read_text()) == []


def test_run_checkpoint_roundtrip_reproduces_disjoint_frozen_evaluation(tmp_path):
    settings = tiny_config()
    output = tmp_path / "experiment"
    report = run(settings, output)
    restored = evaluate(output / "final.pt", tmp_path / "restored")
    assert report["training"]["environment_seed"] == settings.seed
    assert report["evaluation"]["environment_seed"] == settings.seed + 100000
    assert without_timing(restored) == without_timing(report["evaluation"])
    assert json.loads((output / "metrics.json").read_text()) == report
    checkpoint = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    assert checkpoint["format_version"] == 1
    assert len(checkpoint["models"]) == len(settings.cell_mix)
    assert all(
        torch.isfinite(tensor).all()
        for cell in checkpoint["models"]
        for model in cell
        for tensor in model.values()
    )
    with pytest.raises(FileExistsError):
        run(settings, output)
    with pytest.raises(FileExistsError):
        evaluate(output / "final.pt", tmp_path / "restored")


@pytest.mark.parametrize(
    "method", ["ddqn", "fl-rl", "milp", "no-price", "no-warmstart", "no-fl", "no-events"]
)
def test_ablations_change_actual_training_components(method, tmp_path):
    settings = tiny_config(method=method)
    agents = make_agents(settings)
    destination = tmp_path / method
    report = run_phase(settings, agents, settings.train_steps, settings.seed, True, destination)
    solves = json.loads((destination / "leader.json").read_text())
    federation = json.loads((destination / "federation.json").read_text())
    if method in ("ddqn", "fl-rl"):
        assert solves == []
    else:
        assert solves
    if method in ("ddqn", "no-fl", "milp"):
        assert federation == []
        assert report["communication_payload_bytes"]["model_upload"] == 0
    else:
        assert {entry["level"] for entry in federation} == {"cell", "global"}
        assert report["communication_payload_bytes"]["model_upload"] > 0
    if method == "no-price":
        assert all(entry["broadcast_price"] == 0 for entry in solves)
    if method in ("no-events", "milp"):
        assert all(entry["reason"] != "event" for entry in solves)
    if method == "milp":
        assert agents[0][0].replay is None
        assert agents[0][0].updates == 0
    elif method in ("ddqn", "fl-rl", "no-warmstart"):
        assert agents[0][0].replay.size == settings.train_steps
    else:
        assert agents[0][0].replay.size == settings.train_steps + settings.warm_steps * len(solves)


def test_seed_intervals_use_sample_standard_deviation_and_student_t():
    result = interval([1, 2, 3, 4, 5])
    assert result["mean"] == 3
    assert result["std"] == pytest.approx(np.sqrt(2.5))
    # t_(.975, df=4) = 2.776445105..., rather than the large-sample 1.96 value.
    margin = 2.7764451051977987 * np.sqrt(2.5) / np.sqrt(5)
    assert result["ci95"] == pytest.approx([3 - margin, 3 + margin])
    assert interval([4])["ci95"] is None
    for values in ([], [np.nan], [1, np.inf]):
        with pytest.raises(ValueError):
            interval(values)


def test_paired_comparison_joins_matching_seeds_and_handles_equal_outcomes():
    report = paired({1: 5, 2: 7, 3: 999}, {1: 5, 2: 7, 4: -999})
    assert report["seeds"] == [1, 2]
    assert report["difference"]["mean"] == 0
    assert report["p_value"] == 1
    assert paired({1: 5}, {1: 5, 2: 7}) is None
