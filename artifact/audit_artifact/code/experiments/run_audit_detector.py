"""Train the HP-60 random-walk strategy classifier for the audit."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit.config import AUDIT_ROOT, audit_config
from src.strategy.train_detector import generate_dataset, save_model, train_model


def main():
    cfg = audit_config(
        seed=0, arm="partitioned", label_source="detector", cond=True,
        detector_path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "audit", "detector_hp60_randomwalk.pt",
        ),
        tag="detector-hp60-randomwalk",
    )
    out_dir = os.path.join(AUDIT_ROOT, "detector")
    os.makedirs(out_dir, exist_ok=True)
    print("generating random-walk telemetry ...", flush=True)
    X, y = generate_dataset(cfg, seed=0, npc_behavior="gentle")
    print(f"dataset: {X.shape} classes={y.tolist()[:0]} counts={[int((y==i).sum()) for i in range(4)]}",
          flush=True)
    model, acc, per = train_model(cfg, X, y)
    save_model(model, cfg)
    metrics = {
        "detector": "hp60-randomwalk",
        "dataset_size": int(len(y)),
        "validation_accuracy": acc,
        "per_class_recall": per,
        "model_path": cfg["detector"]["model_path"],
    }
    with open(os.path.join(out_dir, "randomwalk_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
