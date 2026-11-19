"""F7: strategy-classifier validation in the deployment distribution."""

import json
import os

import numpy as np
import torch

from ..env.chase_env import ChaseEnv
from ..env.features import FeatureWindow
from ..env.strategy import LABEL_TO_ID
from ..strategy.detector import FeatureHistory, StrategyClassifier
from ..strategy.train_detector import _generation_action, train_model
from .config import LABELS, audit_config
from .runner import load_agent, observed


def load_classifier(cfg, path):
    model = StrategyClassifier(cfg)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def _episode_rows(cfg, policy, classifier, agent=None, policy_cfg=None,
                  episodes_per_label=6, ep_len=200, seed=2000):
    rows = []
    rng = np.random.default_rng(seed)
    for label_id, label in enumerate(LABELS):
        for ep in range(episodes_per_label):
            env = ChaseEnv(cfg, seed=int(rng.integers(0, 2 ** 31)))
            env.player.max_hp = env.player.hp = 1e9
            env.npc.max_hp = env.npc.hp = 1e9
            env.player_max_hp = env.npc_max_hp = 1e9
            env.reset(player_strategy=label)
            env.player.max_hp = env.player.hp = 1e9
            env.npc.max_hp = env.npc.hp = 1e9
            fw = FeatureWindow(cfg["strategy"]["feature_window"])
            fh = FeatureHistory(cfg["detector"]["context_len"])
            prev_npc = env.npc_pos.copy()
            prev_player = env.player.pos.copy()
            prev_info = {}
            ep_rows = []
            for step in range(int(ep_len)):
                if policy == "randomwalk":
                    action = _generation_action(env, "gentle")
                elif policy == "competent":
                    base = env._obs()
                    obs = observed(policy_cfg, env, base, label_id)
                    action = agent.greedy(obs)
                else:
                    raise ValueError(policy)
                _, _, _, truncated, info = env.step(action)
                fw.update(FeatureWindow.extract_step(
                    env.npc_pos, prev_npc, env.player.pos, prev_player,
                    combat={
                        "player_attack": info.get("player_attacked", False),
                        "player_hit": info.get("player_damage_dealt", 0.0) > 0,
                        "damage_dealt": info.get("player_damage_dealt", 0.0),
                        "player_hp_frac": info.get("player_hp", 1.0)
                        / (env.player.max_hp or 1.0),
                    },
                ))
                feat = fw.features()
                if feat is not None:
                    fh.add(feat)
                    if fh.ready():
                        pred, probs = classifier.predict(fh.as_context())
                        pred = int(pred)
                        ep_rows.append({
                            "episode_label": label,
                            "episode_label_id": int(label_id),
                            "episode_index": int(ep),
                            "step": int(step),
                            "pred": int(pred),
                            "probs": [float(x) for x in probs],
                            "max_prob": float(np.max(probs)),
                            "correct": int(pred == label_id),
                        })
                prev_npc = env.npc_pos.copy()
                prev_player = env.player.pos.copy()
                prev_info = info
                if truncated:
                    break
            rows.extend(ep_rows)
    return rows


def _summarize(rows):
    if not rows:
        return {}
    y = np.asarray([r["episode_label_id"] for r in rows], dtype=int)
    p = np.asarray([r["pred"] for r in rows], dtype=int)
    correct = (y == p).astype(np.float64)
    n_classes = len(LABELS)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1

    per_class = {}
    recalls = []
    f1s = []
    for c, name in enumerate(LABELS):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": f1, "support": int(cm[c, :].sum())}
        recalls.append(recall)
        f1s.append(f1)

    ep_acc = {}
    for label_id, name in enumerate(LABELS):
        vals = []
        for ep in sorted({r["episode_index"] for r in rows
                          if r["episode_label_id"] == label_id}):
            rr = [r for r in rows
                  if r["episode_label_id"] == label_id and r["episode_index"] == ep]
            if rr:
                vals.append(float(np.mean([r["correct"] for r in rr])))
        ep_acc[name] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "median": float(np.median(vals)) if vals else float("nan"),
            "values": vals,
        }

    conf = np.asarray([r["max_prob"] for r in rows], dtype=np.float64)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if mask.sum():
            ece += float(mask.mean()) * abs(float(correct[mask].mean())
                                           - float(conf[mask].mean()))

    return {
        "n_windows": int(len(rows)),
        "window_accuracy": float(correct.mean()),
        "macro_f1": float(np.nanmean(f1s)),
        "macro_recall": float(np.nanmean(recalls)),
        "per_class": per_class,
        "episode_accuracy": ep_acc,
        "confusion": cm.tolist(),
        "ece_10bin": float(ece),
        "mean_confidence": float(conf.mean()),
    }


def evaluate_classifier(cfg, classifier, policy, agent=None, policy_cfg=None,
                        episodes_per_label=6, ep_len=200, seed=2000):
    rows = _episode_rows(cfg, policy, classifier, agent=agent,
                         policy_cfg=policy_cfg,
                         episodes_per_label=episodes_per_label,
                         ep_len=ep_len, seed=seed)
    return _summarize(rows), rows


def build_training_dataset(cfg, policy_cfg, checkpoint, seed=3000,
                           episodes_per_label=40, ep_len=200):
    """Return (X, y) telemetry windows for retraining on a competent chaser."""
    _, agent = load_agent(policy_cfg, checkpoint)
    X, y = [], []
    rng = np.random.default_rng(seed)
    for label_id, label in enumerate(LABELS):
        for _ in range(episodes_per_label):
            env = ChaseEnv(cfg, seed=int(rng.integers(0, 2 ** 31)))
            env.player.max_hp = env.player.hp = 1e9
            env.npc.max_hp = env.npc.hp = 1e9
            env.player_max_hp = env.npc_max_hp = 1e9
            env.reset(player_strategy=label)
            env.player.max_hp = env.player.hp = 1e9
            env.npc.max_hp = env.npc.hp = 1e9
            fw = FeatureWindow(cfg["strategy"]["feature_window"])
            fh = FeatureHistory(cfg["detector"]["context_len"])
            prev_npc = env.npc_pos.copy()
            prev_player = env.player.pos.copy()
            prev_info = {}
            for _step in range(int(ep_len)):
                base = env._obs()
                obs = observed(policy_cfg, env, base, label_id)
                action = agent.greedy(obs)
                _, _, _, truncated, info = env.step(action)
                fw.update(FeatureWindow.extract_step(
                    env.npc_pos, prev_npc, env.player.pos, prev_player,
                    combat={
                        "player_attack": info.get("player_attacked", False),
                        "player_hit": info.get("player_damage_dealt", 0.0) > 0,
                        "damage_dealt": info.get("player_damage_dealt", 0.0),
                        "player_hp_frac": info.get("player_hp", 1.0)
                        / (env.player.max_hp or 1.0),
                    },
                ))
                feat = fw.features()
                if feat is not None:
                    fh.add(feat)
                    if fh.ready():
                        X.append(np.concatenate(fh.as_context()))
                        y.append(label_id)
                prev_npc = env.npc_pos.copy()
                prev_player = env.player.pos.copy()
                prev_info = info
                if truncated:
                    break
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
