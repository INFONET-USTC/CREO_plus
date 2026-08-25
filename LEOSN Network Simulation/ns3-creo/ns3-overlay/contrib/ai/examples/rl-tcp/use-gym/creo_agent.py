import math
import os
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pre_process import daubechies_denoise


DEFAULT_ACTION_SPACE = [0.61, 0.85, 0.95, 1.0, 1.03, 1.27, 1.67]


@dataclass
class CreoTransition:
    state: dict
    action: int
    reward: float
    next_state: dict
    done: bool


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class CreoFeatureBuilder:
    """Build CREO+ temporal state from the 20-field ns-3 observation."""

    def __init__(self, capacity_window=64, history_len=10):
        self.capacity_window = capacity_window
        self.history_len = history_len
        self.capacity_hist = deque(maxlen=capacity_window)
        self.metric_hist = deque(maxlen=history_len)
        self.last_rtt = None
        self.last_action = 1.0
        self.min_rtt = None
        self.best_tp = 1.0
        self.last_send_rate_mbps = 0.0
        self.control_steps = 0

    @staticmethod
    def _safe_ratio(value, denom):
        return float(value) / float(denom) if denom else 0.0

    @staticmethod
    def _moving_average(values, window):
        if len(values) == 0:
            return values
        window = max(1, min(window, len(values)))
        kernel = np.ones(window, dtype=np.float32) / float(window)
        return np.convolve(values, kernel, mode="same").astype(np.float32)

    def make(self, obs, action_multiplier=None, target_rate_bps=None):
        obs = np.asarray(obs, dtype=np.float32)
        segment_size = max(float(obs[6]), 1.0)
        c_wnd_packets = self._safe_ratio(obs[5], segment_size)
        throughput_mbps = float(obs[15]) / 1e6
        sampled_capacity_mbps = max(float(obs[17]) / 1e6, throughput_mbps, 1e-6)
        rtt_ms = max(float(obs[11]) / 1000.0, 1e-3)

        self.best_tp = max(self.best_tp, throughput_mbps, sampled_capacity_mbps)
        if self.min_rtt is None or rtt_ms < self.min_rtt:
            self.min_rtt = rtt_ms
        rtt_grad = 0.0 if self.last_rtt is None else rtt_ms - self.last_rtt
        self.last_rtt = rtt_ms

        loss_rate = float(obs[18]) / 1e6
        inflight_packets = self._safe_ratio(obs[8], segment_size)
        bdp_packets = max(float(obs[16]), 1.0)
        if action_multiplier is not None:
            self.last_action = float(action_multiplier)
        if target_rate_bps is not None:
            self.last_send_rate_mbps = max(float(target_rate_bps) / 1e6, 0.0)
        elif self.last_send_rate_mbps <= 0.0:
            self.last_send_rate_mbps = max(throughput_mbps, sampled_capacity_mbps)

        metrics = np.array(
            [
                loss_rate,
                throughput_mbps / max(self.best_tp, 1e-6),
                sampled_capacity_mbps / max(self.best_tp, 1e-6),
                rtt_ms / max(self.min_rtt, 1e-3),
                self.min_rtt / 1000.0,
                rtt_grad / 100.0,
                inflight_packets / max(bdp_packets, 1.0),
                self.last_send_rate_mbps / max(self.best_tp, 1e-6),
                self.last_action,
            ],
            dtype=np.float32,
        )
        metrics = np.clip(
            metrics,
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.1], dtype=np.float32),
            np.array([1.0, 2.0, 2.0, 10.0, 10.0, 10.0, 10.0, 4.0, 3.0], dtype=np.float32),
        )
        self.metric_hist.append(metrics)
        self.capacity_hist.append(sampled_capacity_mbps)

        cap = np.asarray(self.capacity_hist, dtype=np.float32)
        cap = daubechies_denoise(cap)
        trend = self._moving_average(cap, max(3, len(cap) // 8))
        fluct = cap - trend

        cap = self._pad_left(cap, self.capacity_window)
        trend = self._pad_left(trend, self.capacity_window)
        fluct = self._pad_left(fluct, self.capacity_window)
        metric_hist = self._pad_left(np.asarray(self.metric_hist, dtype=np.float32), self.history_len, axis=0)

        scale = max(self.best_tp, 1e-6)
        return {
            "capacity": np.clip(cap / scale, 0.0, 2.0),
            "trend": np.clip(trend / scale, 0.0, 2.0),
            "fluct": np.clip(fluct / scale, -2.0, 2.0),
            "metrics": metric_hist,
        }

    @staticmethod
    def _pad_left(values, length, axis=None):
        arr = np.asarray(values, dtype=np.float32)
        if axis is None:
            if arr.shape[0] >= length:
                return arr[-length:]
            pad = np.zeros(length - arr.shape[0], dtype=np.float32)
            return np.concatenate([pad, arr])
        if arr.shape[0] >= length:
            return arr[-length:]
        pad_shape = list(arr.shape)
        pad_shape[0] = length - arr.shape[0]
        pad = np.zeros(pad_shape, dtype=np.float32)
        return np.concatenate([pad, arr], axis=0)


class CreoNet(nn.Module):
    def __init__(self, action_dim, capacity_window=64, history_len=10, metric_dim=9, hidden=96):
        super().__init__()
        self.cap_lstm = nn.LSTM(1, hidden, batch_first=True)
        self.trend_lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fluct_lstm = nn.LSTM(1, hidden, batch_first=True)
        self.metric_cnn = nn.Sequential(
            nn.Conv1d(metric_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        fused = hidden * 3 + 64 + history_len * metric_dim
        self.shared = nn.Sequential(nn.Linear(fused, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.actor = nn.Linear(128, action_dim)
        self.q1 = nn.Linear(128, action_dim)
        self.q2 = nn.Linear(128, action_dim)

    def encode(self, batch):
        cap = batch["capacity"].unsqueeze(-1)
        trend = batch["trend"].unsqueeze(-1)
        fluct = batch["fluct"].unsqueeze(-1)
        metrics = batch["metrics"]

        _, (hc, _) = self.cap_lstm(cap)
        _, (ht, _) = self.trend_lstm(trend)
        _, (hf, _) = self.fluct_lstm(fluct)
        metric_cnn = self.metric_cnn(metrics.transpose(1, 2)).squeeze(-1)
        metric_flat = metrics.flatten(start_dim=1)
        x = torch.cat([hc[-1], ht[-1], hf[-1], metric_cnn, metric_flat], dim=1)
        return self.shared(x)

    def forward(self, batch):
        z = self.encode(batch)
        return self.actor(z), self.q1(z), self.q2(z)


class CreoAgent:
    def __init__(
        self,
        action_space=None,
        capacity_window=64,
        history_len=10,
        mini_window_segments=15,
        replay_size=20000,
        batch_size=64,
        gamma=0.98,
        tau=0.005,
        lr=3e-4,
        device=None,
        eval_mode=False,
    ):
        self.action_space = list(action_space or DEFAULT_ACTION_SPACE)
        self.capacity_window = capacity_window
        self.history_len = history_len
        self.mini_window_segments = mini_window_segments
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.eval_mode = eval_mode
        self.builder = CreoFeatureBuilder(capacity_window, history_len)
        self.replay = ReplayBuffer(replay_size)
        self.net = CreoNet(len(self.action_space), capacity_window, history_len).to(self.device)
        self.target = CreoNet(len(self.action_space), capacity_window, history_len).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = math.log(len(self.action_space)) * 0.7
        self.last_state = None
        self.last_action_idx = None
        self.learn_steps = 0

    def new_feature_builder(self):
        return CreoFeatureBuilder(self.capacity_window, self.history_len)

    def observe(self, obs, action_multiplier=None, target_rate_bps=None, builder=None):
        feature_builder = builder or self.builder
        return feature_builder.make(obs, action_multiplier, target_rate_bps)

    def choose_action(self, state, deterministic=False):
        batch = self._collate([state])
        with torch.no_grad():
            logits, _, _ = self.net(batch)
            probs = torch.softmax(logits, dim=-1)
            if deterministic or self.eval_mode:
                action_idx = int(torch.argmax(probs, dim=-1).item())
            else:
                action_idx = int(torch.distributions.Categorical(probs=probs).sample().item())
        return action_idx, self.action_space[action_idx]

    def neutral_action(self):
        action_idx = min(
            range(len(self.action_space)),
            key=lambda index: abs(self.action_space[index] - 1.0),
        )
        return action_idx, self.action_space[action_idx]

    def action_to_control(self, obs, multiplier, builder=None):
        feature_builder = builder or self.builder
        obs = np.asarray(obs, dtype=np.float32)
        segment_size = max(int(obs[6]), 1)
        observed_rate_bps = max(float(obs[17]), float(obs[15]), 1.0)
        previous_rate_bps = feature_builder.last_send_rate_mbps * 1e6
        min_rtt_seconds = max(float(obs[12]) / 1e6, 1e-3)
        cwnd_rate_bps = max(float(obs[5]) * 8.0 / min_rtt_seconds, 1.0)
        if feature_builder.control_steps < 3:
            # The initial burst samples establish the connected phase's first
            # sending-rate baseline before multiplicative DRL control begins.
            rate_base_bps = max(previous_rate_bps, observed_rate_bps, cwnd_rate_bps)
        else:
            rate_base_bps = previous_rate_bps
        feature_builder.control_steps += 1
        target_rate_bps = int(np.clip(multiplier * rate_base_bps, 1e5, 1e9))
        target_bdp_bytes = target_rate_bps * min_rtt_seconds / 8.0
        minimum_burst_bytes = self.mini_window_segments * segment_size
        cwnd = int(np.clip(target_bdp_bytes, minimum_burst_bytes, (1 << 31) - 1))
        return cwnd, target_rate_bps

    def action_to_cwnd(self, obs, multiplier):
        return self.action_to_control(obs, multiplier)[0]

    def remember(self, state, action_idx, reward, next_state, done):
        self.replay.push(CreoTransition(state, action_idx, reward, next_state, done))

    def learn(self):
        if self.eval_mode or len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)
        states = self._collate([t.state for t in batch])
        next_states = self._collate([t.next_state for t in batch])
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        logits, q1, q2 = self.net(states)
        q1_a = q1.gather(1, actions)
        q2_a = q2.gather(1, actions)

        with torch.no_grad():
            next_logits, tq1, tq2 = self.target(next_states)
            next_probs = torch.softmax(next_logits, dim=-1)
            next_log_probs = torch.log_softmax(next_logits, dim=-1)
            next_q = torch.min(tq1, tq2)
            alpha = self.log_alpha.exp()
            next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + (1.0 - dones) * self.gamma * next_v

        q_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        min_q = torch.min(q1, q2)
        alpha = self.log_alpha.exp().detach()
        actor_loss = (probs * (alpha * log_probs - min_q)).sum(dim=1).mean()

        self.optim.zero_grad()
        (q_loss + actor_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.optim.step()

        entropy = -(probs.detach() * log_probs.detach()).sum(dim=1).mean()
        alpha_loss = self.log_alpha.exp() * (entropy - self.target_entropy)
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self._soft_update()
        self.learn_steps += 1
        return {
            "q_loss": float(q_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
        }

    def _soft_update(self):
        with torch.no_grad():
            for target_param, param in zip(self.target.parameters(), self.net.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def _collate(self, states):
        return {
            key: torch.tensor(np.stack([s[key] for s in states]), dtype=torch.float32, device=self.device)
            for key in ("capacity", "trend", "fluct", "metrics")
        }

    def save(self, path, metadata=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optim.state_dict(),
                "alpha_optimizer": self.alpha_optim.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "action_space": self.action_space,
                "learn_steps": self.learn_steps,
                "capacity_window": self.capacity_window,
                "history_len": self.history_len,
                "mini_window_segments": self.mini_window_segments,
                "gamma": self.gamma,
                "metadata": metadata or {},
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.action_space = checkpoint.get("action_space", self.action_space)
        self.net.load_state_dict(checkpoint["net"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["net"]))
        if "optimizer" in checkpoint and not self.eval_mode:
            self.optim.load_state_dict(checkpoint["optimizer"])
        if "alpha_optimizer" in checkpoint and not self.eval_mode:
            self.alpha_optim.load_state_dict(checkpoint["alpha_optimizer"])
        if "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"].to(self.device).detach().clone().requires_grad_(True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.optim.param_groups[0]["lr"])
        self.learn_steps = checkpoint.get("learn_steps", 0)


def creo_reward(obs, theta=0.2, default=0.0):
    obs = np.asarray(obs, dtype=np.float32)
    throughput = max(float(obs[15]), 0.0)
    tpmax = max(float(obs[19]), 1.0) if obs.size > 19 else max(float(obs[17]), throughput, 1.0)
    rtt = max(float(obs[11]), 1.0)
    min_rtt = max(float(obs[12]), 1.0)
    loss = max(float(obs[18]) / 1e6, 0.0)
    reward = (throughput / tpmax - theta * loss) / ((rtt / min_rtt) ** 2)
    if not np.isfinite(reward):
        return default
    return float(np.clip(reward, -2.0, 2.0))


def set_seed(seed, torch_threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
