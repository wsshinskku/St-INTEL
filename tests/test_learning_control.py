"""Behavioral checks for DDQN/FedProx and the independently timed leader loop."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from st_intel.config import Config
from st_intel.control import Monitor
from st_intel.learning import Agent, Replay, average_models, ddqn_targets, relative_model_change


def tiny_config(**changes):
    values = dict(
        cell_mix=[[1, 1, 0]],
        cell_names=["Test"],
        rrbs=2,
        request_cap=1,
        hidden_dim=2,
        batch_size=1,
        replay_capacity=8,
        learning_rate=0.1,
        target_sync_steps=2,
        fedprox_mu=0.4,
        tti_ms=2.5,
        throughput_target_mbps=10.0,
        latency_target_ms=10.0,
        stats_window_steps=3,
        event_window_steps=2,
        monitor_steps=2,
        milp_interval_steps=12,
        cooldown_steps=5,
    )
    values.update(changes)
    return Config(**values).validate()


def zero_agent(config=None, seed=3):
    agent = Agent(config or tiny_config(), seed=seed)
    agent.set_shared({key: torch.zeros_like(value) for key, value in agent.cpu_state().items()})
    return agent


def transition(scale=1.0):
    # Rates/arrivals already use Mbit/ms; delivered is Mbit per 2.5 ms TTI.
    return {
        "rates": np.array([[0.001, 0.002], [0.003, 0.004]]) * scale,
        "backlog": np.array([0.1, 0.2]) * scale,
        "arrivals": np.array([0.002, 0.003]) * scale,
        "delivered": np.array([0.0125, 0.0125]) * scale,
        "hol": np.array([10.0, 10.0]),
        "lost": np.zeros(2),
        "completed": np.array([5, 5]),
    }


def test_double_q_uses_online_argmax_target_value_and_terminal_mask():
    # The two networks disagree about the best action in every row.
    online = torch.tensor([[1.0, 9.0, 3.0], [8.0, 2.0, 1.0], [0.0, 1.0, 7.0]])
    target = torch.tensor([[100.0, 4.0, 20.0], [3.0, 200.0, 5.0], [9.0, 8.0, 900.0]])
    result = ddqn_targets(
        online, target, torch.tensor([2.0, -1.0, 6.0]), torch.tensor([0.0, 0.0, 1.0]), gamma=0.5
    )
    torch.testing.assert_close(result, torch.tensor([4.0, 0.5, 6.0]))


def test_replay_ring_overwrites_oldest_and_samples_reproducibly():
    buffers = [Replay(capacity=3, seed=42), Replay(capacity=3, seed=42)]
    for buffer in buffers:
        for index in range(5):
            state = np.full(4, index, dtype=np.float32)
            buffer.add(state, index, index + 0.5, state + 1, done=index % 2)
        assert buffer.size == 3
        assert buffer.position == 2
        assert set(buffer.actions) == {2, 3, 4}
    for _ in range(3):
        batch_a, batch_b = buffers[0].sample(2, "cpu"), buffers[1].sample(2, "cpu")
        for actual, repeated in zip(batch_a, batch_b):
            torch.testing.assert_close(actual, repeated)
        states, actions, rewards, next_states, done = batch_a
        torch.testing.assert_close(states[:, 0], actions.float())
        torch.testing.assert_close(rewards, actions.float() + 0.5)
        torch.testing.assert_close(next_states, states + 1)
        torch.testing.assert_close(done, (actions % 2).float())
    with pytest.raises(ValueError, match="insufficient"):
        buffers[0].sample(4, "cpu")


def test_replay_retains_transition_after_callers_mutate_their_input():
    buffer = Replay(2, 0)
    state = np.arange(4, dtype=np.float32)
    next_state = state + 1
    buffer.add(state, 0, 1.0, next_state)
    state[:] = -100
    next_state[:] = -100
    states, _, _, next_states, _ = buffer.sample(1, "cpu")
    torch.testing.assert_close(states[0], torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(next_states[0], torch.arange(1, 5, dtype=torch.float32))


@pytest.mark.parametrize(
    "proximal, expected_bias, expected_loss", [(True, 1.92, 0.8), (False, 2.0, 0.0)]
)
def test_fedprox_moves_an_unselected_action_toward_broadcast_reference(
    proximal, expected_bias, expected_loss
):
    agent = zero_agent(tiny_config(target_sync_steps=100))
    # No TD gradient can reach action 1: only action 0 is sampled, with a zero
    # terminal target and zero prediction. Any movement of this bias therefore
    # comes from the proximal term mu/2 * ||theta - reference||^2.
    with torch.no_grad():
        agent.online.net[-1].bias[1] = 2.0
    agent.optimizer = torch.optim.SGD(agent.online.parameters(), lr=0.1)
    agent.replay.add(np.zeros(4, np.float32), 0, 0, np.zeros(4, np.float32), done=True)
    loss = agent.update(proximal=proximal)
    assert loss == pytest.approx(expected_loss, abs=1e-7)
    assert float(agent.online.net[-1].bias[1].detach()) == pytest.approx(expected_bias)
    assert all(torch.count_nonzero(reference) == 0 for reference in agent.reference)


def test_target_is_fixed_until_configured_training_update_then_synchronized():
    agent = zero_agent()
    before = {key: value.clone() for key, value in agent.target.state_dict().items()}
    agent.replay.add(np.zeros(4, np.float32), 0, 1, np.zeros(4, np.float32), done=True)
    agent.update(proximal=False)
    assert agent.updates == 1
    assert agent.online.net[-1].bias[0].item() > 0
    for key, value in agent.target.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert all(parameter.grad is None for parameter in agent.target.parameters())
    agent.update(proximal=False)
    assert agent.updates == 2
    for key, value in agent.target.state_dict().items():
        torch.testing.assert_close(value, agent.online.state_dict()[key])


def test_model_averaging_weights_every_parameter_including_output_head():
    agents = [zero_agent(seed=1), zero_agent(seed=2)]
    for multiplier, agent in zip((1.0, 3.0), agents):
        state = {
            key: torch.full_like(value, multiplier * (index + 1))
            for index, (key, value) in enumerate(agent.cpu_state().items())
        }
        agent.set_shared(state)
    averaged = average_models(agents, weights=[1.0, 3.0])
    assert set(averaged) == set(agents[0].online.state_dict())
    for index, (key, value) in enumerate(averaged.items()):
        torch.testing.assert_close(value, torch.full_like(value, 2.5 * (index + 1)))
        torch.testing.assert_close(
            agents[0].online.state_dict()[key], torch.full_like(value, index + 1)
        )
    # Results are detached snapshots rather than aliases to a client's head.
    averaged["net.4.bias"].zero_()
    assert torch.count_nonzero(agents[0].online.net[-1].bias) > 0


@pytest.mark.parametrize("weights", [[0, 1], [-1, 1], [1], [np.nan, 1]])
def test_invalid_federated_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="weight"):
        average_models([zero_agent(seed=1), zero_agent(seed=2)], weights)


def test_broadcast_replaces_both_networks_anchors_fedprox_and_resets_optimizer():
    agent = zero_agent()
    agent.replay.add(np.zeros(4, np.float32), 0, 1, np.zeros(4, np.float32), done=True)
    agent.update()
    assert agent.optimizer.state
    shared = {key: torch.full_like(value, 0.25) for key, value in agent.cpu_state().items()}
    agent.set_shared(shared)
    assert not agent.optimizer.state
    for online, target, reference in zip(
        agent.online.parameters(), agent.target.parameters(), agent.reference
    ):
        torch.testing.assert_close(online, torch.full_like(online, 0.25))
        torch.testing.assert_close(target, online)
        torch.testing.assert_close(reference, online)
        assert not reference.requires_grad
        assert reference.data_ptr() != online.data_ptr()
        assert target.data_ptr() != online.data_ptr()
    with torch.no_grad():
        agent.online.net[-1].bias.add_(1)
    torch.testing.assert_close(agent.reference[-1], torch.full_like(agent.reference[-1], 0.25))
    assert agent.replay.size == 1  # Federation keeps private experience.


def test_relative_model_change_is_norm_ratio_and_handles_zero_reference():
    old = {"weight": torch.tensor([3.0, 4.0]), "bias": torch.tensor([0.0])}
    new = {"weight": torch.tensor([3.0, 4.0]), "bias": torch.tensor([2.0])}
    assert relative_model_change(old, new) == pytest.approx(0.4)
    assert relative_model_change({"p": torch.zeros(1)}, {"p": torch.zeros(1)}) == 0
    assert np.isfinite(relative_model_change({"p": torch.zeros(1)}, {"p": torch.ones(1)}))


@pytest.mark.parametrize(
    "change",
    [
        {"delivered": np.array([0.025, 0.025])},  # Throughput overperformance.
        {"delivered": np.zeros(2)},  # Throughput shortfall.
        {"hol": np.zeros(2)},  # Latency better than target.
        {"hol": np.array([20.0, 20.0])},  # Latency worse than target.
        {"lost": np.array([1, 1])},  # Packet loss threshold.
    ],
)
def test_monitor_uses_literal_absolute_intent_deviations(change):
    monitor = Monitor(tiny_config())
    assert monitor.reason(0) == "initial"
    monitor.solved(0)
    record = {**transition(), **change}
    monitor.add(record)
    assert monitor.reason(6) is None  # Wait for a full observation window.
    monitor.add(record)
    assert monitor.reason(2) is None  # Cooldown still in effect.
    assert monitor.reason(6) == "event"
    assert monitor.reason(7) is None  # Check only at monitoring cadence.
    assert monitor.reason(6, events=False) is None


def test_periodic_refresh_is_independent_of_event_toggle_and_cooldown():
    monitor = Monitor(tiny_config())
    monitor.solved(11)
    # A recent solve and an empty event window do not suppress periodic work.
    assert monitor.reason(12, events=False) == "periodic"
    assert monitor.reason(12, events=True) == "periodic"


def test_new_solve_restarts_event_cooldown():
    monitor = Monitor(tiny_config(milp_interval_steps=100))
    monitor.solved(0)
    for _ in range(2):
        monitor.add(transition(scale=2.0))
    assert monitor.reason(6) == "event"
    monitor.solved(6)
    assert monitor.reason(10) is None
    assert monitor.reason(12) == "event"


def test_on_target_observations_do_not_trigger_and_loss_is_count_weighted():
    monitor = Monitor(tiny_config())
    monitor.solved(0)
    for _ in range(2):
        monitor.add(transition())
    assert monitor.reason(6) is None
    monitor.add({**transition(), "lost": np.array([1, 0]), "completed": np.array([99, 0])})
    monitor.add({**transition(), "lost": np.array([1, 0]), "completed": np.array([0, 0])})
    indicators = monitor.indicators()
    assert indicators["loss_rate"] == pytest.approx(2 / 101)
    assert indicators["throughput_mbit_per_ms"] == pytest.approx(0.01)


def test_epoch_averages_keep_arrival_rate_units_and_bound_history():
    config = tiny_config()
    monitor = Monitor(config)
    cell = SimpleNamespace(
        rates=np.ones((2, 2)) * 0.01,
        backlog=lambda: np.array([0.03, 0.04]),
        arrivals=np.array([0.0125, 0.025]),  # Mbit arriving in this TTI.
        minimum_rates=np.array([0.002, 0.0015]),
        delay_limits=np.array([30.0, 5.0]),
    )
    first = monitor.problem(cell)
    np.testing.assert_allclose(first.arrivals, [0.005, 0.01])
    assert first.throughput_target == pytest.approx(0.01)
    for scale in (1, 2, 3, 4):
        monitor.add(transition(scale=scale))
    assert len(monitor.records) == 3
    epoch = monitor.problem(cell)
    np.testing.assert_allclose(epoch.rates, transition(scale=3)["rates"])
    np.testing.assert_allclose(epoch.backlog, [0.3, 0.6])
    np.testing.assert_allclose(epoch.arrivals, [0.006, 0.009])
    np.testing.assert_array_equal(epoch.minimum_rates, cell.minimum_rates)
    assert epoch.latency_target == config.latency_target_ms
    assert epoch.resource_budget == config.rrbs
    # The event window has two records while the optimization window has three.
    assert monitor.indicators()["throughput_mbit_per_ms"] == pytest.approx(0.035)


def test_config_json_roundtrip_and_explicit_override(tmp_path):
    config = tiny_config(seed=19)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    restored = Config.load(path, seed=20, method="no-events", scenario=None)
    assert restored.seed == 20
    assert restored.method == "no-events"
    assert restored.scenario == config.scenario
    assert restored.cell_mix == config.cell_mix
    assert restored.tti_ms == 2.5


@pytest.mark.parametrize(
    "changes",
    [
        {"request_cap": 3},
        {"batch_size": 9},
        {"tti_ms": 0},
        {"cell_mix": [[0, 0, 0]]},
        {"event_window_steps": 0},
        {"scenario": "unknown"},
    ],
)
def test_config_rejects_inconsistent_learning_and_control_settings(changes):
    with pytest.raises(ValueError):
        tiny_config(**changes)
