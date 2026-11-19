"""Double-DQN agent used for all NPC training variants."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import DQNetwork, DuelingDQNetwork


class DQNAgent:
    def __init__(self, cfg, obs_dim, n_actions, device="cpu"):
        self.cfg = cfg
        self.rl_cfg = cfg["rl"]
        self.device = torch.device(device)
        self.n_actions = n_actions

        net_cls = DuelingDQNetwork if self.rl_cfg.get("network", "dqn") == "dueling" else DQNetwork
        self.online = net_cls(obs_dim, n_actions, self.rl_cfg["hidden_sizes"]).to(self.device)
        self.target = net_cls(obs_dim, n_actions, self.rl_cfg["hidden_sizes"]).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(
            self.online.parameters(),
            lr=float(self.rl_cfg["lr"]),
            weight_decay=float(self.rl_cfg.get("weight_decay", 0.0)),
        )
        self.loss_fn = nn.SmoothL1Loss()

        self._update_counter = 0
        self.last_stats = {}

        self.eps_start = float(self.rl_cfg["epsilon_start"])
        self.eps_end = float(self.rl_cfg["epsilon_end"])
        self.eps_decay = float(self.rl_cfg["epsilon_decay_steps"])

        self.np_rng = __import__("numpy").random.default_rng(int(cfg["seed"]))

    # ------------------------------------------------------------------ #
    def epsilon(self, step):
        frac = min(1.0, step / max(self.eps_decay, 1.0))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, obs, step, epsilon=None):
        eps = self.epsilon(step) if epsilon is None else epsilon
        if self.np_rng.random() < eps:
            return int(self.np_rng.integers(0, self.n_actions))
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online(x)
        return int(q.argmax(dim=-1).item())

    @torch.no_grad()
    def greedy(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online(x)
        return int(q.argmax(dim=-1).item())

    # ------------------------------------------------------------------ #
    def update(self, transitions):
        batch = [list(t) for t in transitions]
        s = torch.as_tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        a = torch.as_tensor([b[1] for b in batch], dtype=torch.long, device=self.device).unsqueeze(-1)
        r = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)
        s2 = torch.as_tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        d = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(-1)

        q = self.online(s).gather(1, a)
        with torch.no_grad():
            a2 = self.online(s2).argmax(dim=-1, keepdim=True)
            q2 = self.target(s2).gather(1, a2)
            target_q = r + float(self.rl_cfg["gamma"]) * q2 * (1.0 - d)
            target_clip = float(self.rl_cfg.get("target_clip", 0.0) or 0.0)
            if target_clip > 0.0:
                target_q = target_q.clamp(-target_clip, target_clip)

        loss = self.loss_fn(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(self.online.parameters(), 10.0))
        self.optimizer.step()

        self._update_counter += 1
        freq = int(self.rl_cfg["target_update_freq"])
        if freq > 0 and self._update_counter % freq == 0:
            self.target.load_state_dict(self.online.state_dict())
        tau = float(self.rl_cfg["tau"])
        if tau > 0 and tau < 1.0:
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)

        self.last_stats = {
            "update": self._update_counter,
            "loss": float(loss.item()),
            "q_mean_abs": float(q.detach().abs().mean().item()),
            "q_max_abs": float(q.detach().abs().max().item()),
            "target_mean_abs": float(target_q.detach().abs().mean().item()),
            "target_max_abs": float(target_q.detach().abs().max().item()),
            "grad_norm": grad_norm,
        }

        return loss.item()

    def save(self, path):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ck["online"])
        self.target.load_state_dict(ck["target"])