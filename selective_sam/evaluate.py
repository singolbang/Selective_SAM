"""Evaluate the published selector against baseline and always-fusion."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

from .corruption import corrupt_depth
from .dataset_baseline import GateDatasetBaselineSAM
from .gate_baseline import GateBaselineSAM
from .infer_baseline import SelectivePredictorBaseline
from .label_generation import sha256
from .samples import adapter_depth, load_sample
from .train_gate_baseline import predict_sam


def aggregate(rows: list[dict], name: str, image_count: int):
    if not rows:
        raise ValueError(name)
    output = {"condition": name, "images": image_count, "targets": len(set(row["target_id"] for row in rows)),
              "target_condition_pairs": len(rows)}
    for mode in ("baseline", "fusion", "selective", "oracle"):
        output[f"{mode}_iou"] = float(np.mean([row[f"{mode}_iou"] for row in rows]))
    output["fusion_delta"] = output["fusion_iou"] - output["baseline_iou"]
    output["selective_delta"] = output["selective_iou"] - output["baseline_iou"]
    output["adapter_use_rate"] = float(np.mean([row["adapter_used"] for row in rows]))
    harms = [row["baseline_iou"] - row["fusion_iou"] for row in rows if row["adapter_used"] and row["fusion_iou"] < row["baseline_iou"]]
    output["harmful_use_fraction"] = len(harms) / len(rows)
    output["mean_harm_when_used"] = float(np.mean(harms)) if harms else 0.0
    return output


def _montage(label_row, sample_row, config, predictor, destination):
    sample = load_sample(sample_row, Path(config["data"]["root"]))
    target = label_row["target"]
    condition = label_row["condition"]
    cache_row = label_row["cache_row"]
    depth, valid = corrupt_depth(sample["depth"], sample["valid"], condition,
                                 cache_row["strength"], cache_row["seed"])
    box = target["box"]
    with torch.inference_mode():
        features = predictor.model.encode_rgb(sample["rgb"])
        d, v = adapter_depth(
            {"depth": depth, "valid": valid}, predictor.model.sam.image_size, predictor.model.device_name)
        base = predictor.model.segment(features, np.array([box], np.float32), sample["rgb"].shape[:2])[0]
        fused = predictor.model.segment(features, np.array([box], np.float32), sample["rgb"].shape[:2],
                                        d, v, adapter_on=True)[0]
    original = next(i for i, t in enumerate(sample_row["targets"]) if t["id"] == target["id"])
    gt = sample["masks"][original]
    selected = fused > 0 if label_row["adapter_used"] else base > 0
    masks = [gt, (base > 0).cpu().numpy(), (fused > 0).cpu().numpy(),
             selected.cpu().numpy() if torch.is_tensor(selected) else selected]
    h, w = sample["rgb"].shape[:2]
    size = (320, 320)
    rgb = Image.fromarray(sample["rgb"]).resize(size)
    depth_visual = Image.fromarray(np.uint8(np.clip(depth, 0, 1) * 255)).convert("RGB").resize(size)
    panels = [rgb, depth_visual] + [Image.fromarray(np.uint8(mask) * 255).convert("RGB").resize(size)
                                       for mask in masks]
    canvas = Image.new("RGB", (size[0] * 3, size[1] * 2 + 48), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ["RGB", "Depth", "GT", "Baseline", "Fusion", "Selective"]
    for i, (panel, title) in enumerate(zip(panels, titles)):
        x, y = (i % 3) * size[0], (i // 3) * size[1]
        canvas.paste(panel, (x, y))
        draw.text((x + 5, y + 5), title, fill="red")
    draw.text((5, size[1] * 2 + 8),
              f"{condition} score={label_row['gate_score']:.3f} adapter={label_row['adapter_used']} "
              f"IoU B/F/S={label_row['baseline_iou']:.3f}/{label_row['fusion_iou']:.3f}/{label_row['selective_iou']:.3f}",
              fill="black")
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--gate-config", default="configs/gate_baseline.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    gate_config = yaml.safe_load(Path(args.gate_config).read_text())
    policy = json.loads(Path("checkpoints/selection_policy.json").read_text())
    adapter_hash = sha256(Path("checkpoints/adapter_best.pt"))
    saved = torch.load("checkpoints/gate_best.pt", map_location="cpu", weights_only=False)
    if (policy["adapter_sha256"] != adapter_hash or saved["adapter_sha256"] != adapter_hash
            or saved.get("architecture") != "GateBaselineSAM_v1"
            or saved["gate_baseline_source_sha256"] != sha256(Path("selective_sam/gate_baseline.py"))
            or saved["gate_config"] != gate_config):
        raise ValueError("Adapter, selector, source or config mismatch")
    dataset = GateDatasetBaselineSAM("test", config, adapter_hash, gate_config)
    gate = GateBaselineSAM().to(config["sam"]["device"])
    gate.load_state_dict(saved["gate"])
    probabilities = predict_sam(gate, dataset, config["sam"]["device"], gate_config["batch_size"])
    predictor = SelectivePredictorBaseline(args.config, args.gate_config)
    records = []
    for i, (row_index, target_index) in enumerate(dataset.entries):
        cache_row = dataset.rows[row_index]
        target = cache_row["targets"][target_index]
        used = (policy["mode"] == "on" or
                (policy["mode"] == "selective" and probabilities[i] >= policy["tau"]))
        b, f = target["baseline_iou"], target["fusion_iou"]
        records.append({"image_id": cache_row["image_id"], "target_id": target["id"],
                        "condition": cache_row["condition"], "gate_score": float(probabilities[i]),
                        "adapter_used": bool(used), "baseline_iou": b, "fusion_iou": f,
                        "selective_iou": f if used else b, "oracle_iou": max(b, f),
                        "delta_iou": f - b, "harmful_use": bool(used and f < b),
                        "cache_row": cache_row, "target": target})
    path = Path("results")
    path.mkdir(exist_ok=True)
    public = [{k: v for k, v in row.items() if k not in ("cache_row", "target")} for row in records]
    with (path / "per_sample_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(public[0]))
        writer.writeheader()
        writer.writerows(public)
    conditions = ["clean"] + [condition for condition in sorted(set(dataset.conditions)) if condition != "clean"]
    condition_metrics = [aggregate([row for row in records if row["condition"] == condition], condition,
                                   sum(cache_row["condition"] == condition for cache_row in dataset.rows))
                         for condition in conditions]
    metrics = list(condition_metrics)
    metrics.append(aggregate([row for row in records if row["condition"] != "clean"],
                             "damaged_observed_pairs", len({row["image_id"] for row in records})))
    metrics.append(aggregate(records, "all_observed_pairs", len({row["image_id"] for row in records})))
    equal = {key: float(np.mean([row[key] for row in condition_metrics])) for key in
             ("baseline_iou", "fusion_iou", "selective_iou", "oracle_iou", "adapter_use_rate",
              "harmful_use_fraction", "mean_harm_when_used")}
    weighted = {key: .5 * condition_metrics[0][key] + .5 * float(np.mean([row[key] for row in condition_metrics[1:]]))
                for key in equal}
    for name, values in (("equal_10_conditions", equal), ("weighted_clean50_damage50", weighted)):
        metrics.append({"condition": name, "images": len({row["image_id"] for row in records}),
                        "targets": len({row["target_id"] for row in records}),
                        "target_condition_pairs": len(records), **values,
                        "fusion_delta": values["fusion_iou"] - values["baseline_iou"],
                        "selective_delta": values["selective_iou"] - values["baseline_iou"]})
    with (path / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    qualitative = path / "qualitative"
    qualitative.mkdir(exist_ok=True)
    choices = []
    gains = [row for row in records if row["adapter_used"] and row["delta_iou"] > 0]
    harms = [row for row in records if row["delta_iou"] < 0]
    misses = [row for row in records if not row["adapter_used"] and row["delta_iou"] > 0]
    wrong_uses = [row for row in records if row["adapter_used"] and row["delta_iou"] < 0]
    for name, candidates, reverse in (("gain", gains, True), ("harm", harms, False),
                                      ("miss", misses, True), ("wrong_use", wrong_uses, False)):
        if candidates:
            choices.append((name, sorted(candidates, key=lambda row: row["delta_iou"], reverse=reverse)[0]))
    for name, row in choices:
        _montage(row, dataset.manifest[row["image_id"]], config, predictor, qualitative / f"{name}.png")
    print(json.dumps({"test_images": len({row["image_id"] for row in records}), "pairs": len(records),
                      "policy": policy, "weighted": weighted, "qualitative": [name for name, _ in choices]}, indent=2))



if __name__ == "__main__":
    main()
