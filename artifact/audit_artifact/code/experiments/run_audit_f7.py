"""F7: cross-opponent classifier evaluation and competent-chaser retraining."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit.config import AUDIT_ROOT, audit_config
from src.audit.f7 import (build_training_dataset, evaluate_classifier,
                          load_classifier, write_json)
from src.strategy.train_detector import save_model, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-checkpoint", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "audit", "checkpoints",
        "single200k-gt-off-ep400-sw40-seed0.pt",
    ))
    parser.add_argument("--episodes-per-label", type=int, default=6)
    parser.add_argument("--episode-len", type=int, default=200)
    parser.add_argument("--retrain-episodes-per-label", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--skip-retrain", action="store_true")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    detector_dir = os.path.join(root, "data", "audit")
    randomwalk_path = os.path.join(detector_dir, "detector_hp60_randomwalk.pt")
    competent_path = os.path.join(detector_dir, "detector_hp60_competent.pt")
    out_dir = os.path.join(AUDIT_ROOT, "f7")
    os.makedirs(out_dir, exist_ok=True)

    base = audit_config(
        seed=0, arm="single200k", label_source="detector", cond=False,
        detector_path=randomwalk_path, tag="f7-base",
    )
    randomwalk_model = load_classifier(base, randomwalk_path)

    policy_cfg = audit_config(
        seed=0, arm="single200k", label_source="groundtruth", cond=False,
        tag="f7-policy",
    )

    results = {"randomwalk_model": {}, "competent_model": {}}
    for policy in ("randomwalk", "competent"):
        print(f"evaluating randomwalk-trained classifier on {policy} ...", flush=True)
        agent = None
        if policy == "competent":
            from src.audit.runner import load_agent
            _, agent = load_agent(policy_cfg, args.policy_checkpoint)
        summary, rows = evaluate_classifier(
            base, randomwalk_model, policy, agent=agent,
            policy_cfg=policy_cfg,
            episodes_per_label=args.episodes_per_label,
            ep_len=args.episode_len,
            seed=2000,
        )
        results["randomwalk_model"][policy] = summary
        with open(os.path.join(out_dir, f"randomwalk-model-{policy}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    if not args.skip_retrain:
        print("building competent-chaser training set ...", flush=True)
        cfg_comp = audit_config(
            seed=1, arm="partitioned", label_source="detector", cond=False,
            detector_path=competent_path, tag="f7-competent-train",
        )
        cfg_comp["detector"]["epochs"] = args.epochs
        X, y = build_training_dataset(
            cfg_comp, policy_cfg, args.policy_checkpoint, seed=3000,
            episodes_per_label=args.retrain_episodes_per_label,
            ep_len=args.episode_len,
        )
        print(f"competent dataset: {X.shape}, counts={[int((y==i).sum()) for i in range(4)]}",
              flush=True)
        model, acc, per = train_model(cfg_comp, X, y)
        save_model(model, cfg_comp)
        write_json(os.path.join(out_dir, "competent_train_metrics.json"), {
            "validation_accuracy": acc, "per_class_recall": per,
            "dataset_size": int(len(y)), "model_path": competent_path,
        })
        competent_model = load_classifier(cfg_comp, competent_path)
        for policy in ("randomwalk", "competent"):
            print(f"evaluating competent-trained classifier on {policy} ...", flush=True)
            if policy == "competent":
                from src.audit.runner import load_agent
                _, agent = load_agent(policy_cfg, args.policy_checkpoint)
            else:
                agent = None
            summary, rows = evaluate_classifier(
                cfg_comp, competent_model, policy, agent=agent,
                policy_cfg=policy_cfg,
                episodes_per_label=args.episodes_per_label,
                ep_len=args.episode_len, seed=2000,
            )
            results["competent_model"][policy] = summary
            with open(os.path.join(out_dir, f"competent-model-{policy}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

    write_json(os.path.join(out_dir, "f7_summary.json"), results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
