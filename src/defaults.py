"""Canonical default configuration as plain Python (no yaml dependency).

This mirrors config/default.yaml. Keeping a Python copy guarantees the project
runs even if PyYAML is not installed.
"""

DEFAULTS = {
    "seed": 0,
    "env": {
        "world_size": 20.0,
        "dt": 0.1,
        "npc_max_speed": 3.0,
        "npc_sensor_range": 20.0,
        "episode_len": 400,
        "catch_dist": 1.1,
        "player_cfg": {
            "aggression": 0.6,
            "evasion_distance": 0.0,
            "clinging": 0.0,
        },
    },
    "strategy": {
        "params": {
            # (aggression, retreat_rate, evasion_distance, max_speed_mult,
            #  engagement_dist, attack_frequency, attack_range, retreat_health_threshold)
            "aggressive": [0.95, 0.05, 0.0, 1.0, 1.2, 0.85, 1.5, 0.20],
            "defensive": [0.10, 0.85, 0.2, 0.7, 16.0, 0.35, 1.4, 0.60],
            "evasive": [0.35, 0.25, 1.6, 1.20, 5.0, 0.55, 1.5, 0.35],
            "balanced": [0.65, 0.30, 0.3, 1.0, 2.2, 0.65, 1.5, 0.30],
        },
        "labels": ["aggressive", "defensive", "evasive", "balanced"],
        "feature_window": 15,
        "dt": 0.1,
    },
    "combat": {
        "player_hp": 20,
        "npc_hp": 20,
        "attack_range": 1.6,
        "cone": 70,           # facing cone half-angle (deg) for a hit
        "attack_cooldown": 4, # steps between attacks
        "damage": 2,          # HP removed per hit
        "reward_win": 150.0,
        "reward_lose": 50.0,
        "reward_hit": 8.0,
        "reward_heading": 0.6,
        "reward_approach": 1.2,
        "reward_inrange": 1.0,
        "reward_penalty": 0.005,
    },
    "detector": {
        "hidden_sizes": [64, 64],
        "epochs": 60,
        "lr": 0.001,
        "batch_size": 128,
        "context_len": 8,  # number of consecutive 5-feature vectors fed to MLP
        "rollout_episodes_per_strategy": 40,
        "rollout_episode_len": 200,
        "model_path": "data/detector.pt",
    },
    "transitions": {
        "cusum_k": 0.5,
        "cusum_h": 6.0,
        "min_segment": 10,
        "confirm": 10,
        "score_smoothing": 4,
    },
    "rl": {
        "algorithm": "dqn",
        "hidden_sizes": [128, 128],
        "lr": 0.0005,
        "gamma": 0.99,
        "tau": 0.005,
        "target_update_freq": 500,
        "buffer_capacity": 50000,
        "batch_size": 256,
        "learn_every": 2,  # run a learning update every N steps (latency-bound)
        "epsilon_start": 0.4,
        "epsilon_end": 0.02,
        "epsilon_decay_steps": 15000,
        "replay": "partitioned",  # single | partitioned | partitioned_noprior
        "cond_on_strategy": True,  # append detected-strategy one-hot to state
        "prior_power": 1.6,
        "stale_decay": 0.5,
        "episodes": 400,
        "switch_every": 40,
        "eval_switch_every": 60,
        "seed_demos": 30,
        "params_jitter": 0.2,  # 0..~0.3 domain randomization on player params
        "in_episode_switch": True,  # train ON mid-episode personality transitions
        "switch_step": 100,          # step at which the player changes strategy
    },
}
