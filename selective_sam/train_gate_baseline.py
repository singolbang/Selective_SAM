"""Train a gate with frozen baseline mask evidence and no fusion candidate input."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .dataset_baseline import GateDatasetBaselineSAM
from .gate_baseline import GateBaselineSAM
from .gate_sam import SAM_FEATURE_DIM
from .label_generation import sha256


def choose_damage_policy(conditions, baseline, fusion, probabilities):
    """Choose by the raw mean IoU of all damaged target-condition pairs."""
    damaged = np.asarray(conditions) != "clean"
    if not damaged.any():
        raise ValueError("No damaged validation inputs")
    baseline = np.asarray(baseline)
    fusion = np.asarray(fusion)
    probabilities = np.asarray(probabilities)
    candidates = [("off", 1.0), ("on", 0.0)]
    candidates += [("selective", float(tau)) for tau in np.arange(.05, 1.0, .025)]
    scored = []
    for mode, tau in candidates:
        if mode == "off":
            use = np.zeros(len(probabilities), dtype=bool)
        elif mode == "on":
            use = np.ones(len(probabilities), dtype=bool)
        else:
            use = probabilities >= tau
        chosen = np.where(use[damaged], fusion[damaged], baseline[damaged])
        scored.append({"mode": mode, "tau": tau, "damaged_iou": float(chosen.mean()),
                       "damaged_pairs": int(damaged.sum()),
                       "adapter_use_rate": float(use[damaged].mean())})
    scored.sort(key=lambda row: (-round(row["damaged_iou"], 8),
                                 row["adapter_use_rate"], row["mode"]))
    return scored[0], scored


def predict_sam(model, dataset, device, batch_size):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    scores = []
    with torch.inference_mode():
        for global_input, local_input, geometry, sam_vector, _ in loader:
            logits = model(global_input.to(device), local_input.to(device),
                           geometry.to(device), sam_vector.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--gate-config", default="configs/gate_baseline.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    gate_config = yaml.safe_load(Path(args.gate_config).read_text())
    if (gate_config["architecture"] != "GateBaselineSAM_v1" or
            gate_config["sam_feature_dim"] != SAM_FEATURE_DIM or
            gate_config["baseline_hint"] != "sigmoid_logits_area_resized_uint8" or
            gate_config["objective"] != "binary_cross_entropy" or
            gate_config["class_weight"] != "none" or gate_config["sample_weight"] != "none" or
            gate_config["validation_scope"] != "damaged_observed_pairs" or
            gate_config["validation_aggregation"] != "unweighted_target_mean"):
        raise ValueError("Expected baseline hint, plain BCE and damaged validation")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.set_num_threads(4)
    adapter_hash = sha256(Path("checkpoints/adapter_best.pt"))
    train = GateDatasetBaselineSAM("gate_train", config, adapter_hash, gate_config)
    val = GateDatasetBaselineSAM("val", config, adapter_hash, gate_config)
    device = config["sam"]["device"]
    model = GateBaselineSAM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=gate_config["learning_rate"])
    warmup, total = gate_config["warmup_epochs"], gate_config["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: (.1 + .9 * (epoch + 1) / warmup) if epoch < warmup
        else .5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, total - warmup))))
    log_dir = Path("runs/gate")
    log_dir.mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    best_score, stale = -1.0, 0
    for epoch in range(total):
        model.train()
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        loader = DataLoader(train, batch_size=gate_config["batch_size"], shuffle=True,
                            generator=generator, num_workers=0, pin_memory=True)
        losses = []
        for global_input, local_input, geometry, sam_vector, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(global_input.to(device, non_blocking=True),
                           local_input.to(device, non_blocking=True),
                           geometry.to(device, non_blocking=True),
                           sam_vector.to(device, non_blocking=True))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        probabilities = predict_sam(model, val, device, gate_config["batch_size"])
        selected, candidates = choose_damage_policy(val.conditions, val.baseline,
                                                     val.fusion, probabilities)
        improved = selected["damaged_iou"] > best_score + 1e-8
        stale = 0 if improved else stale + 1
        if improved:
            best_score = selected["damaged_iou"]
            torch.save({"gate": model.state_dict(), "architecture": gate_config["architecture"],
                        "gate_baseline_source_sha256": sha256(Path("selective_sam/gate_baseline.py")),
                        "dataset_baseline_source_sha256": sha256(Path("selective_sam/dataset_baseline.py")),
                        "adapter_sha256": adapter_hash,
                        "sam_sha256": config["sam"]["checkpoint_sha256"],
                        "sam_source_commit": config["sam"]["official_code_commit"],
                        "epoch": epoch + 1, "validation_policy": selected,
                        "gate_config": gate_config,
                        "positive_train": int(train.labels.sum()),
                        "negative_train": int(len(train) - train.labels.sum())},
                       "checkpoints/gate_best.pt")
            Path("checkpoints/selection_policy.json").write_text(json.dumps({
                **selected, "epsilon": config["gate"]["epsilon"],
                "validation_scope": gate_config["validation_scope"],
                "validation_aggregation": gate_config["validation_aggregation"],
                "adapter_sha256": adapter_hash,
                "sam_sha256": config["sam"]["checkpoint_sha256"],
                "gate_baseline_source_sha256": sha256(Path("selective_sam/gate_baseline.py")),
                "selected_epoch": epoch + 1}, indent=2))
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
                  "selected": selected, "candidates": candidates, "best": improved}
        with (log_dir / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({key: value for key, value in record.items()
                          if key != "candidates"}), flush=True)
        if epoch + 1 >= gate_config["min_epochs"] and stale >= gate_config["patience"]:
            break


if __name__ == "__main__":
    main()
