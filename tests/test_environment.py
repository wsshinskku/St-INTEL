"""Physical, temporal, and reproducibility invariants of the analytical simulator."""

from collections import deque
from dataclasses import replace

import numpy as np
import pytest

from st_intel.config import Config
from st_intel.environment import Cell, Network, Packet, feasible_grants


def config(**changes):
    values = {
        "cell_mix": [[1, 0, 0]],
        "cell_names": ["test"],
        "rrbs": 2,
        "request_cap": 2,
        "embb_offered_mbps": 0.0,
        "mmtc_packets_per_second": 0.0,
    }
    values.update(changes)
    return Config(**values).validate()


def cell_for(settings):
    return Cell(settings, settings.cell_mix[0], index=0, seed=settings.seed)


def pending_payload(cell):
    # Partially transmitted packets are not delivered until completion.
    return np.array([sum(packet.original for packet in q) for q in cell.queues])


def test_scheduler_exclusivity_request_caps_and_zero_backlog():
    rng = np.random.default_rng(7)
    for _ in range(30):
        rates = rng.uniform(0.0001, 0.001, (12, 5))
        requests = rng.integers(0, 7, 5)
        backlog = rng.uniform(0.001, 0.02, 5)
        backlog[1] = 0
        grants = feasible_grants(rates, requests, backlog, np.zeros(5), np.ones(5), np.ones(5))
        assert np.isin(grants, [0, 1]).all()
        assert (grants.sum(axis=1) <= 1).all()
        assert (grants.sum(axis=0) <= requests).all()
        assert grants[:, 1].sum() == 0


def test_scheduler_stops_granting_once_queue_capacity_is_covered():
    rates = np.full((6, 1), 0.002)
    grants = feasible_grants(rates, [6], [0.003], [0], [10], [1], tti_ms=1)
    assert grants.sum() == 2


def test_scheduler_zero_rates_and_requests_do_not_allocate():
    for rates, requests in [(np.zeros((3, 2)), [3, 3]), (np.ones((3, 2)), [0, 0])]:
        grants = feasible_grants(rates, requests, [1, 1], [0, 0], [1, 1], [1, 1])
        assert not grants.any()


@pytest.mark.parametrize("actions", [[-1], [0.5], [np.nan], [np.inf], [], [0, 0]])
def test_scheduler_rejects_invalid_requests(actions):
    with pytest.raises(ValueError):
        feasible_grants(np.ones((2, 1)), actions, [1], [0], [1], [1])


@pytest.mark.parametrize("actions", [[-1], [0.5], [np.nan], [np.inf], [3], [], [0, 0]])
def test_cell_rejects_invalid_requests_without_advancing(actions):
    cell = cell_for(config())
    before = cell.backlog().copy()
    with pytest.raises(ValueError):
        cell.step(actions, price=0)
    assert cell.tick == 0
    np.testing.assert_array_equal(cell.backlog(), before)


def test_zero_request_preserves_fifo_payload_and_advances_hol():
    cell = cell_for(config())
    cell.queues[0].append(Packet(0.001, 0.001, -3))
    result = cell.step([0], price=0)
    np.testing.assert_array_equal(result["grants"], 0)
    assert result["service"][0] == 0
    assert result["delivered"][0] == 0
    assert cell.queues[0][0].remaining == 0.001
    assert result["hol"][0] == 4


def test_fifo_partial_service_completion_and_timestamp_delay():
    cell = cell_for(config())
    cell.queues[0] = deque([Packet(0.0015, 0.0015, -3), Packet(0.001, 0.001, -1)])
    cell.rates[:] = 0.001
    first = cell.step([1], price=0)
    assert first["service"][0] == pytest.approx(0.001)
    assert first["delivered"][0] == 0
    assert first["hol"][0] == 4
    assert len(cell.queues[0]) == 2
    assert cell.queues[0][0].remaining == pytest.approx(0.0005)
    cell.rates[:] = 0.001
    second = cell.step([1], price=0)
    assert second["completed"][0] == 1
    assert second["delivered"][0] == pytest.approx(0.0015)
    assert second["service"][0] == pytest.approx(0.001)
    # Equal physical service should not create a reward spike on packet completion.
    assert second["reward"][0] == pytest.approx(first["reward"][0])
    assert second["delay_sum"][0] == 5
    assert second["hol"][0] == 3
    assert len(cell.queues[0]) == 1
    assert cell.queues[0][0].born_ms == -1
    assert cell.queues[0][0].remaining == pytest.approx(0.0005)


def test_service_cannot_exceed_granted_capacity_or_queued_payload():
    cell = cell_for(config(tti_ms=2))
    cell.queues[0].append(Packet(0.001, 0.001, 0))
    cell.rates[:] = 0.002
    result = cell.step([2], price=0)
    capacity = (result["grants"] * result["rates"]).sum(0) * cell.config.tti_ms
    assert np.all(result["service"] <= capacity + 1e-12)
    assert np.all(result["service"] <= result["backlog"] + 1e-12)
    assert result["service"][0] == pytest.approx(0.001)
    assert result["delivered"][0] == pytest.approx(0.001)
    assert result["hol"][0] == 0


def test_failed_complete_packet_is_dropped_once_and_never_delivered():
    cell = cell_for(config(base_packet_loss=1.0))
    cell.queues[0].append(Packet(0.001, 0.001, 0))
    cell.rates[:] = 0.002
    result = cell.step([1], price=0)
    assert result["service"][0] == pytest.approx(0.001)
    assert result["lost"][0] == 1
    assert result["dropped"][0] == pytest.approx(0.001)
    assert result["delivered"][0] == 0
    assert result["completed"][0] == 0
    assert not cell.queues[0]


def test_offered_payload_conservation_with_overflow_loss_and_partial_service():
    settings = config(
        cell_mix=[[1, 1, 1]],
        rrbs=4,
        request_cap=4,
        embb_offered_mbps=12,
        mmtc_packets_per_second=2000,
        queue_limit_mbit=0.012,
        base_packet_loss=0.4,
    )
    cell = cell_for(settings)
    cumulative_offered = np.zeros(3)
    cumulative_delivered = np.zeros(3)
    cumulative_dropped = np.zeros(3)
    cumulative_packets = np.zeros(3, dtype=int)
    completed_packets = np.zeros(3, dtype=int)
    lost_packets = np.zeros(3, dtype=int)
    for tick in range(50):
        result = cell.step([4 if tick % 3 == i else 0 for i in range(3)], price=0)
        cumulative_offered += result["offered"]
        cumulative_delivered += result["delivered"]
        cumulative_dropped += result["dropped"]
        cumulative_packets += result["offered_packets"]
        completed_packets += result["completed"]
        lost_packets += result["lost"]
        # step() has already prepared the next tick; exclude those newly enqueued arrivals.
        packet_sizes = np.asarray(settings.packet_bytes)[cell.types] * 8 / 1e6
        next_accepted = cell.arrivals - cell.overflow_packets * packet_sizes
        pending = pending_payload(cell) - next_accepted
        np.testing.assert_allclose(
            cumulative_offered,
            cumulative_delivered + cumulative_dropped + pending,
            atol=1e-12,
            rtol=1e-12,
        )
        pending_count = np.array([len(q) for q in cell.queues])
        pending_count -= cell.arrival_packets - cell.overflow_packets
        np.testing.assert_array_equal(
            cumulative_packets, completed_packets + lost_packets + pending_count
        )
    assert cumulative_delivered.sum() > 0
    assert cumulative_dropped.sum() > 0


def test_price_penalty_depends_on_request_not_grant_count():
    settings = config()
    first, second = cell_for(settings), cell_for(settings)
    # Empty queues guarantee identical zero grants, but different requested resource costs.
    result0, result2 = first.step([0], 0.3), second.step([2], 0.3)
    assert result0["grants"].sum() == result2["grants"].sum() == 0
    assert result0["reward"][0] - result2["reward"][0] == pytest.approx(0.6)


def test_keyed_exogenous_trajectory_is_independent_of_policy_and_extra_cells():
    settings = config(
        cell_mix=[[1, 1, 1]],
        embb_offered_mbps=5,
        mmtc_packets_per_second=2000,
        scenario="unstable",
        shock_starts_ms=[3],
        shock_duration_ms=4,
    )
    idle, busy = cell_for(settings), cell_for(settings)
    unrelated = Cell(settings, settings.cell_mix[0], index=0, seed=settings.seed)
    for _ in range(20):
        unrelated.step([2, 2, 2], 0)
    for _ in range(12):
        np.testing.assert_array_equal(idle.rates, busy.rates)
        np.testing.assert_array_equal(idle.arrivals, busy.arrivals)
        np.testing.assert_array_equal(idle.loss_draw, busy.loss_draw)
        assert idle.shock == busy.shock
        idle.step([0, 0, 0], 0)
        busy.step([2, 2, 2], 0.3)
    assert idle.backlog().sum() > busy.backlog().sum()


def test_seed_and_cell_index_produce_distinct_channel_profiles():
    settings = config()
    first = Cell(settings, [1, 0, 0], index=0, seed=1)
    other_seed = Cell(settings, [1, 0, 0], index=0, seed=2)
    other_cell = Cell(settings, [1, 0, 0], index=1, seed=1)
    assert not np.array_equal(first.rates, other_seed.rates)
    assert not np.array_equal(first.rates, other_cell.rates)


def test_shocks_use_half_open_intervals_and_do_not_apply_to_stable_scenario():
    settings = config(scenario="unstable", shock_starts_ms=[2, 6], shock_duration_ms=2)
    unstable = cell_for(settings)
    stable = cell_for(replace(settings, scenario="stable"))
    observed = []
    for _ in range(9):
        observed.append(unstable.shock)
        assert not stable.shock
        unstable.step([0], 0)
        stable.step([0], 0)
    assert observed == [False, False, True, True, False, False, True, True, False]


def test_shock_multiplier_scales_fading_variance_not_standard_deviation():
    settings = config(
        scenario="unstable",
        shock_starts_ms=[0],
        shock_duration_ms=2,
        shock_variance_multiplier=4,
        fading_sigma=0.2,
        spectral_efficiency=0.1,
        rrbs=20,
    )
    stable = cell_for(replace(settings, scenario="stable"))
    shocked = cell_for(settings)
    # Both use identical underlying standard-normal variates; clipping is inactive here.
    baseline = settings.rrb_bandwidth_hz * settings.spectral_efficiency / 1e9
    denominator = baseline * stable.channel_profile[None, :]
    stable_log_fade = np.log(stable.rates / denominator)
    shock_log_fade = np.log(shocked.rates / denominator)
    sigma = settings.fading_sigma
    expected = 2 * (stable_log_fade + 0.5 * sigma**2) - 0.5 * (2 * sigma) ** 2
    np.testing.assert_allclose(shock_log_fade, expected, atol=1e-14)


def test_nonunit_tti_arrival_rates_and_periodic_packet_counts():
    settings = config(cell_mix=[[0, 1, 0]], tti_ms=0.5)
    cell = cell_for(settings)
    result = cell.step([0], 0)
    assert result["offered_packets"][0] == 4
    assert result["offered"][0] == pytest.approx(4 * 32 * 8 / 1e6)
    assert result["arrivals"][0] == pytest.approx(2.048 / 1000)


@pytest.mark.parametrize(
    "actions,prices", [([[0]], [0, 0]), ([[0], [0], [0]], [0, 0]), ([[0], [0]], [0])]
)
def test_network_rejects_wrong_cell_counts_before_advancing(actions, prices):
    settings = config(cell_mix=[[1, 0, 0], [1, 0, 0]], cell_names=["first", "second"])
    network = Network(settings, seed=1)
    with pytest.raises(ValueError):
        network.step(actions, prices)
    assert all(cell.tick == 0 for cell in network.cells)


@pytest.mark.parametrize(
    "change",
    [
        {"warm_steps": 1.5},
        {"seed": 1.5},
        {"shock_starts_ms": [np.nan]},
        {"shock_starts_ms": [np.inf]},
        {"shock_variance_multiplier": np.nan},
        {"shock_variance_multiplier": np.inf},
    ],
)
def test_config_rejects_invalid_temporal_and_random_settings(change):
    settings = replace(config(), **change)
    with pytest.raises(ValueError):
        settings.validate()
