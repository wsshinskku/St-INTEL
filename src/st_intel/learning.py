"""DDQN experience, target selection and FedProx with independent UE replay."""

import copy

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, actions, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, actions),
        )

    def forward(self, state):
        return self.net(state)


class Replay:
    def __init__(self, capacity, seed):
        self.capacity, self.size, self.position = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((capacity, 4), dtype=np.float32)
        self.next_states = np.empty((capacity, 4), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.done = np.empty(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done=False):
        index = self.position
        self.states[index], self.next_states[index] = state, next_state
        self.actions[index], self.rewards[index], self.done[index] = action, reward, done
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        if self.size < batch_size:
            raise ValueError("insufficient replay samples")
        indices = self.rng.choice(self.size, batch_size, replace=False)
        return tuple(
            torch.as_tensor(array[indices], device=device)
            for array in (self.states, self.actions, self.rewards, self.next_states, self.done)
        )


def ddqn_targets(online_next, target_next, rewards, done, gamma):
    """Online chooses the action; target evaluates that action (Eq. 17)."""
    actions = online_next.argmax(dim=1, keepdim=True)
    return rewards + gamma * (1 - done) * target_next.gather(1, actions).squeeze(1)


class Agent:
    def __init__(self, config, seed, initial=None, allocate_replay=True):
        self.config = config
        self.online = QNetwork(config.request_cap + 1, config.hidden_dim).to(config.device)
        if initial is not None:
            self.online.load_state_dict(initial)
        self.target = copy.deepcopy(self.online).eval()
        self.reference = [p.detach().clone() for p in self.online.parameters()]
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = Replay(config.replay_capacity, seed) if allocate_replay else None
        self.rng = np.random.default_rng(seed + 10000)
        self.updates = 0

    def action(self, state, step, greedy=False):
        c = self.config
        epsilon = c.epsilon_end + (c.epsilon_start - c.epsilon_end) * max(
            0, 1 - step / c.epsilon_decay_steps
        )
        if not greedy and self.rng.random() < epsilon:
            return int(self.rng.integers(c.request_cap + 1))
        with torch.no_grad():
            return int(self.online(torch.as_tensor(state, device=c.device)).argmax().item())

    def update(self, proximal=True):
        c = self.config
        if self.replay is None or self.replay.size < c.batch_size:
            return None
        state, action, reward, next_state, done = self.replay.sample(c.batch_size, c.device)
        prediction = self.online(state).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            target = ddqn_targets(
                self.online(next_state), self.target(next_state), reward, done, c.gamma
            )
        loss = torch.nn.functional.smooth_l1_loss(prediction, target)
        if proximal:
            loss = loss + 0.5 * c.fedprox_mu * sum(
                (p - ref).square().sum() for p, ref in zip(self.online.parameters(), self.reference)
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite DDQN loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % c.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu())

    def set_shared(self, state):
        self.online.load_state_dict(state)
        self.target.load_state_dict(state)
        self.reference = [p.detach().clone() for p in self.online.parameters()]
        # Momentum associated with overwritten parameters is discarded at federation boundaries.
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)

    def cpu_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.online.state_dict().items()}


def average_models(agents, weights=None):
    if not agents:
        raise ValueError("at least one agent required")
    weights = np.ones(len(agents)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(agents),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("one finite positive weight per agent required")
    weights = weights / weights.sum()
    states = [agent.cpu_state() for agent in agents]
    return {k: sum(float(w) * state[k] for w, state in zip(weights, states)) for k in states[0]}


def relative_model_change(old, new):
    numerator = sum((new[k] - old[k]).double().square().sum() for k in old).sqrt()
    denominator = sum(old[k].double().square().sum() for k in old).sqrt().clamp_min(1e-12)
    return float(numerator / denominator)
