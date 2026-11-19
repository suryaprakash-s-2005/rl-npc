"""Neural network architectures for the RL agents."""

import torch.nn as nn


def build_mlp(in_dim, hidden_sizes, out_dim):
    layers = []
    sizes = [in_dim] + list(hidden_sizes)
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers += [nn.Linear(sizes[-1], out_dim)]
    return nn.Sequential(*layers)


class DQNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes=(128, 128)):
        super().__init__()
        self.net = build_mlp(in_dim, hidden_sizes, out_dim)

    def forward(self, x):
        return self.net(x)


class DuelingDQNetwork(nn.Module):
    """Dueling architecture: separate value and advantage streams."""

    def __init__(self, in_dim, out_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        sizes = [in_dim] + list(hidden_sizes)
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        last = sizes[-1]
        self.value = nn.Sequential(
            nn.Linear(last, last), nn.ReLU(), nn.Linear(last, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(last, last), nn.ReLU(), nn.Linear(last, out_dim)
        )

    def forward(self, x):
        h = self.trunk(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=-1, keepdim=True)
