"""Strategy classifier (lightweight temporal MLP).

Input  : context_len consecutive 5-feature telemetry vectors flattened
         (context_len * 5 dims).
Output : softmax over 4 strategy classes.
"""

import collections

import torch
import torch.nn as nn

from ..env.features import feature_dim


class StrategyClassifier(nn.Module):
    def __init__(self, cfg, in_dim=None, out_dim=None):
        super().__init__()
        d = cfg["detector"]
        self.context_len = int(d.get("context_len", 8))
        in_dim = in_dim or (self.context_len * feature_dim())
        out_dim = out_dim or len(cfg["strategy"]["labels"])
        layers = []
        sizes = [in_dim] + list(d["hidden_sizes"])
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers += [nn.Linear(sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)

    def predict(self, feature_history):
        """feature_history: list of 5-feature arrays (most recent last)."""
        if len(feature_history) < self.context_len:
            raise ValueError("not enough history for classification")
        ctx = feature_history[-self.context_len :]
        x = torch.as_tensor(torch.cat([torch.as_tensor(v) for v in ctx]).float()).unsqueeze(0)
        with torch.no_grad():
            logits = self.net(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        label = int(probs.argmax())
        return label, probs


class FeatureHistory:
    """Ring buffer of 5-feature vectors, with ready flag."""

    def __init__(self, context_len):
        self.context_len = context_len
        self.history = collections.deque(maxlen=context_len)

    def add(self, feat):
        self.history.append(feat)

    def ready(self):
        return len(self.history) == self.context_len

    def clear(self):
        self.history.clear()

    def as_context(self):
        return list(self.history)