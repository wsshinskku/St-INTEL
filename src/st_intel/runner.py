"""Multi-timescale experiment driver with disjoint training and frozen-policy evaluation."""

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .control import Monitor
from .environment import Network
from .leader import solve_leader
from .learning import Agent, QNetwork, average_models, relative_model_change
from .metrics import Metrics


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def features(method):
    return dict(
        leader=method not in ("ddqn", "fl-rl"),
        pricing=method not in ("ddqn", "fl-rl", "no-price"),
        warm=method not in ("ddqn", "fl-rl", "no-warmstart", "milp"),
        federation=method not in ("ddqn", "no-fl", "milp"),
        events=method not in ("ddqn", "fl-rl", "milp", "no-events"),
        learning=method != "milp",
    )


def make_agents(config, state=None, training=True):
    torch.manual_seed(config.seed)
    initial = QNetwork(config.request_cap + 1, config.hidden_dim).state_dict()
    return [
        [
            Agent(
                config,
                config.seed + 1000 * c + u,
                initial=initial if state is None else state[c][u],
                allocate_replay=training and config.method != "milp",
            )
            for u in range(sum(mix))
        ]
        for c, mix in enumerate(config.cell_mix)
    ]


def warm_replay(cell, agents, requests, price, steps):
    """Roll the teacher on a copied simulator; never relabel incompatible observed actions."""
    teacher = copy.deepcopy(cell)
    for _ in range(steps):
        state = teacher.state(price)
        result = teacher.step(requests, price)
        next_state = teacher.state(price)
        for u, agent in enumerate(agents):
            agent.replay.add(state[u], requests[u], result["reward"][u], next_state[u], False)


def run_phase(config, agents, steps, seed, training, output):
    f = features(config.method)
    env = Network(config, seed)
    monitors = [Monitor(config) for _ in env.cells]
    metrics = Metrics(env)
    prices = np.zeros(len(env.cells))
    requests = [np.zeros(cell.ues, dtype=int) for cell in env.cells]
    solve_log, federation_log, episode_log = [], [], []
    byte_counts = dict(
        model_upload=0,
        model_download=0,
        global_model_upload=0,
        global_model_download=0,
        scalar_price=0,
        epoch_statistics=0,
        scheduling_requests=0,
        grant_counts=0,
    )
    model_bytes = sum(p.numel() * p.element_size() for p in agents[0][0].online.parameters())
    teacher_clipped = 0
    episode_reward = []
    prior_global = average_models([a for cell in agents for a in cell])
    started = time.perf_counter()
    for step in range(steps):
        for c, (cell, monitor) in enumerate(zip(env.cells, monitors)):
            reason = monitor.reason(step, f["events"]) if f["leader"] else None
            if reason:
                solution = solve_leader(
                    monitor.problem(cell),
                    config.sla_weight,
                    config.intent_weight,
                    config.arrival_floor,
                    config.milp_time_limit,
                    config.milp_gap,
                )
                requests[c] = np.clip(solution.requests, 0, config.request_cap).astype(int)
                teacher_clipped += int(np.count_nonzero(solution.requests > config.request_cap))
                prices[c] = solution.price if f["pricing"] else 0.0
                monitor.solved(step)
                solve_log.append(
                    dict(
                        step=step,
                        cell=config.cell_names[c],
                        reason=reason,
                        broadcast_price=float(prices[c]),
                        raw_mean_rrb_price=float(solution.price),
                        teacher_clipped_ues=int(
                            np.count_nonzero(solution.requests > config.request_cap)
                        ),
                        objective=float(solution.objective),
                        throughput_slack=float(solution.throughput_slack),
                        latency_slack=float(solution.latency_slack),
                        diagnostics=solution.diagnostics,
                    )
                )
                byte_counts["scalar_price"] += 8 * cell.ues
                byte_counts["epoch_statistics"] += 8 * (config.rrbs * cell.ues + 2 * cell.ues)
                if training and f["warm"]:
                    warm_replay(cell, agents[c], requests[c], prices[c], config.warm_steps)
        states = env.states(prices)
        if f["learning"]:
            actions = [
                np.array(
                    [
                        agent.action(states[c][u], step, greedy=not training)
                        for u, agent in enumerate(cell)
                    ]
                )
                for c, cell in enumerate(agents)
            ]
        else:
            actions = [r.copy() for r in requests]
        transitions = env.step(actions, prices)
        next_states = env.states(prices)
        metrics.add(transitions)
        episode_reward.append(float(np.mean(np.concatenate([t["reward"] for t in transitions]))))
        for c, transition in enumerate(transitions):
            monitors[c].add(transition)
            byte_counts["scheduling_requests"] += 2 * env.cells[c].ues
            byte_counts["grant_counts"] += 2 * env.cells[c].ues
            if training and f["learning"]:
                for u, agent in enumerate(agents[c]):
                    agent.replay.add(
                        states[c][u],
                        actions[c][u],
                        transition["reward"][u],
                        next_states[c][u],
                        step + 1 == steps,
                    )
                    if (step + 1) % config.train_every == 0:
                        agent.update(proximal=f["federation"])
        if training and f["federation"] and (step + 1) % config.cell_fl_steps == 0:
            for cell_agents in agents:
                shared = average_models(cell_agents)
                for agent in cell_agents:
                    agent.set_shared(shared)
                byte_counts["model_upload"] += model_bytes * len(cell_agents)
                byte_counts["model_download"] += model_bytes * len(cell_agents)
            federation_log.append(dict(step=step + 1, level="cell"))
        if training and f["federation"] and (step + 1) % config.global_fl_steps == 0:
            all_agents = [agent for cell_agents in agents for agent in cell_agents]
            # Mean within cell, then weight cells by UE count: equivalent to equal UE weights.
            shared = average_models(all_agents)
            change = relative_model_change(prior_global, shared)
            for agent in all_agents:
                agent.set_shared(shared)
            prior_global = shared
            byte_counts["global_model_upload"] += model_bytes * len(agents)
            byte_counts["global_model_download"] += model_bytes * len(agents)
            federation_log.append(dict(step=step + 1, level="global", relative_change=change))
        if (step + 1) % config.episode_steps == 0 or step + 1 == steps:
            episode_log.append(
                dict(
                    step=step + 1,
                    episode=len(episode_log) + 1,
                    mean_reward=float(np.mean(episode_reward)),
                )
            )
            episode_reward = []
    report = metrics.report()
    report.update(
        phase="training" if training else "frozen_policy_evaluation",
        environment_seed=seed,
        wall_seconds=time.perf_counter() - started,
        leader_solves=len(solve_log),
        event_solves=sum(r["reason"] == "event" for r in solve_log),
        teacher_clipped_ues=teacher_clipped,
        communication_payload_bytes=byte_counts,
        communication_note="Logical payload accounting only: float32 model tensors, float64 prices/statistics, uint16 request/grant counts; no transport headers or measured fronthaul claim.",
        global_convergence_round=next(
            (
                i + 1
                for i, r in enumerate([x for x in federation_log if x["level"] == "global"])
                if r["relative_change"] < 0.01
            ),
            None,
        ),
        rl_convergence_episodes=None,
        convergence_note="Global: first <1% relative-change crossing, not a stability guarantee. RL convergence is not inferred from a short trace.",
    )
    write_json(Path(output) / "metrics.json", report)
    write_json(Path(output) / "leader.json", solve_log)
    write_json(Path(output) / "federation.json", federation_log)
    write_json(Path(output) / "episodes.json", episode_log)
    return report


def configure_runtime(config):
    config.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")


def run(config, output):
    configure_runtime(config)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("experiment output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.as_dict())
    agents = make_agents(config)
    training = run_phase(config, agents, config.train_steps, config.seed, True, output / "train")
    state = [[agent.cpu_state() for agent in cell] for cell in agents]
    torch.save(dict(format_version=1, config=config.as_dict(), models=state), output / "final.pt")
    evaluation_agents = make_agents(config, state, training=False)
    evaluation = run_phase(
        config,
        evaluation_agents,
        config.evaluation_steps,
        config.seed + 100000,
        False,
        output / "evaluation",
    )
    report = dict(
        method=config.method,
        seed=config.seed,
        scenario=config.scenario,
        config=config.as_dict(),
        training=training,
        evaluation=evaluation,
        provenance="Computed analytical-simulator reference results, not manuscript emulation scores.",
    )
    write_json(output / "metrics.json", report)
    return report


def evaluate(checkpoint, output, seed=None, device=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    values = payload["config"].copy()
    if device is not None:
        values["device"] = device
    config = Config(**values)
    configure_runtime(config)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("evaluation output must be new or empty")
    agents = make_agents(config, payload["models"], training=False)
    return run_phase(
        config,
        agents,
        config.evaluation_steps,
        config.seed + 100000 if seed is None else seed,
        False,
        output,
    )
