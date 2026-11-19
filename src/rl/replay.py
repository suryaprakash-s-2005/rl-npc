"""Replay memories, including the core novelty:

* StrategyPartitionedReplayMemory
    - partitions transitions by the *detected* player-strategy id
    - at sampling time, boosts the partition matching the *current* strategy
      (strategy-aware prioritization) and dampens stale partitions
      (outdated-experience suppression) to reduce negative transfer
    - cross-partition sharing weight is derived from the similarity of the
      underlying strategy parameter vectors

Also provides a standard single ReplayMemory used as the baseline.
"""

import collections

import numpy as np

from ..env.strategy import LABEL_TO_ID, params_for_label


class ReplayMemory:
    """Standard uniform replay buffer.

    A dedicated RNG avoids drawing from NumPy's global stream.  The
    ``legacy_global_rng`` flag is retained only for the F1 audit check that
    reconstructs the original confounded sampling stream.
    """

    def __init__(self, capacity, seed=0, legacy_global_rng=False):
        self.capacity = int(capacity)
        self.buffer = collections.deque(maxlen=self.capacity)
        self._rng = np.random.default_rng(int(seed))
        self.legacy_global_rng = bool(legacy_global_rng)

    def add(self, transition, strategy_id=None):
        self.buffer.append(transition)

    def sample(self, batch_size, **kwargs):
        batch_size = min(batch_size, len(self.buffer))
        if batch_size <= 0:
            return []
        if self.legacy_global_rng:
            idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        else:
            idx = self._rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


def strategy_similarity_matrix(cfg):
    """NxN similarity between strategies from their parameter vectors.

    sim = exp(-normalized distance). Identical -> 1.0.
    """
    labels = cfg["strategy"]["labels"]
    n = len(labels)
    P = np.asarray([params_for_label(cfg, lb) for lb in labels], dtype=np.float64)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = Pn @ Pn.T
    return S


class StrategyPartitionedReplayMemory:
    """Partitioned replay with strategy-aware prioritization."""

    def __init__(self, cfg, capacity, n_strategies, seed=None, legacy_global_rng=False):
        self.cfg = cfg
        self.capacity = int(capacity)
        self.n_strategies = n_strategies
        self.partitions = [
            collections.deque(maxlen=self.capacity) for _ in range(n_strategies)
        ]
        self.similarity = strategy_similarity_matrix(cfg)
        self._rng = np.random.default_rng(int(cfg["seed"] if seed is None else seed))
        self.legacy_global_rng = bool(legacy_global_rng)

    # ------------------------------------------------------------------ #
    def partition_ids(self, ids):
        return [LABEL_TO_ID.get(str(i).lower(), int(i)) for i in ids]

    def add(self, transition, strategy_id):
        pid = self.partition_ids([strategy_id])[0]
        self.partitions[pid].append(transition)

    def _partition_weights(self, current_strategy_id, prioritized):
        cfg_rl = self.cfg["rl"]
        sizes = np.asarray([len(p) for p in self.partitions], dtype=np.float64)
        if not prioritized:
            return np.where(sizes > 0, 1.0, 0.0)
        stale = float(cfg_rl.get("stale_decay", 0.5))
        power = float(cfg_rl.get("prior_power", 1.6))
        sim = self.similarity[current_strategy_id]
        # current strategy -> weight 1; others -> stale * sim^power (>=sizes>=1)
        w = np.full(self.n_strategies, stale, dtype=np.float64)
        w *= np.power(np.clip(sim, 0.0, 1.0), power)
        w[current_strategy_id] = 1.0
        w = np.where(sizes > 0, w, 0.0)
        return w

    def sample(self, batch_size, current_strategy_id=None, prioritized=None, **kwargs):
        cfg_rl = self.cfg["rl"]
        if prioritized is None:
            prioritized = cfg_rl["replay"] != "partitioned_noprior"
        current_strategy_id = int(current_strategy_id or 0)
        sizes = np.asarray([len(p) for p in self.partitions], dtype=np.int64)
        if sizes.sum() == 0:
            return []
        w = self._partition_weights(current_strategy_id, prioritized)
        if w.sum() <= 0:
            w = (sizes > 0).astype(np.float64)
        # allocate counts proportional to weights
        probs = w / w.sum()
        rng = np.random if self.legacy_global_rng else self._rng
        counts = rng.multinomial(min(batch_size, int(sizes.sum())), probs)
        if counts.sum() < batch_size and int(sizes.sum()) < batch_size:
            counts = np.minimum(counts, sizes)
        transitions = []
        for pi, cnt in enumerate(counts):
            cnt = min(int(cnt), sizes[pi])
            if cnt <= 0:
                continue
            idx = rng.choice(sizes[pi], cnt, replace=False)
            transitions += [self.partitions[pi][i] for i in idx]
        if len(transitions) < batch_size and int(sizes.sum()) >= batch_size:
            # topping up from current-strategy partition if under-filled
            pi, missing = current_strategy_id, batch_size - len(transitions)
            missing = min(missing, sizes[pi])
            if missing > 0:
                idx = rng.choice(sizes[pi], missing, replace=False)
                transitions += [self.partitions[pi][i] for i in idx]
        return transitions[:batch_size]

    def strategy_size(self, strategy_id):
        return len(self.partitions[strategy_id])

    def __len__(self):
        return sum(len(p) for p in self.partitions)