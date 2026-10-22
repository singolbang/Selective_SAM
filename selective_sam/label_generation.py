"""Stage B: cache actual per-target IoU differences from fixed SAM and adapter."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .corruption import corrupt_depth
from .model import DepthSAM, iou_from_logits
from .samples import adapter_depth, load_sample, read_manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bytes(path: Path) -> bytes:
    """Hash Python source identically after LF or CRLF checkout."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def source_sha256(path: Path) -> str:
    return hashlib.sha256(_source_bytes(path)).hexdigest()


def preprocessing_signature(config: dict) -> str:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for path in ("selective_sam/data.py", "selective_sam/samples.py",
                 "selective_sam/corruption.py", "selective_sam/model.py",
                 "selective_sam/gate.py", "selective_sam/label_generation.py",
                 "selective_sam/gate_dataset.py"):
        digest.update(path.encode())
        digest.update(_source_bytes(Path(path)))
    return digest.hexdigest()


def load_adapter(config: dict, checkpoint: Path) -> DepthSAM:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["sam_sha256"] != config["sam"]["checkpoint_sha256"]:
        raise ValueError("Adapter checkpoint refers to another SAM checkpoint")
    if saved.get("model_source_sha256") != source_sha256(Path("selective_sam/model.py")):
        raise ValueError("Adapter architecture source changed since checkpoint creation")
    if sha256(Path(config["sam"]["checkpoint"])) != config["sam"]["checkpoint_sha256"]:
        raise ValueError("Actual SAM checkpoint does not match configured SHA256")
    model = DepthSAM(config["sam"]["config"], Path(config["sam"]["checkpoint"]), config["sam"]["device"])
    model.adapter.load_state_dict(saved["adapter"])
    model.eval()
    return model


def make_labels(config: dict, split: str, adapter_checkpoint: Path, gate_config: dict):
    root = Path(config["data"]["root"])
    rows = read_manifest(split)
    model = load_adapter(config, adapter_checkpoint)
    output = Path("cache/gate_labels")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{split}.jsonl"
    temp = destination.with_suffix(".jsonl.tmp")
    adapter_hash = sha256(adapter_checkpoint)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    preprocessing_hash = preprocessing_signature(config)
    manifest_hash = sha256(Path("manifests") / f"{split}.jsonl")
    input_config = {key: gate_config[key] for key in
                    ("global_size", "local_size", "roi_margin_fraction", "roi_min_margin_px")}
    roi_source_hash = source_sha256(Path("selective_sam/gate_roi.py"))
    roi_config_hash = hashlib.sha256(json.dumps(input_config, sort_keys=True).encode()).hexdigest()
    began = time.time()
    with temp.open("w") as stream, torch.inference_mode():
        for index, row in enumerate(rows):
            sample = load_sample(row, root, check_hash=True)
            features = model.encode_rgb(sample["rgb"])
            shape = sample["rgb"].shape[:2]
            truth = torch.from_numpy(sample["masks"]).to(model.device_name)
            valid_gt = torch.from_numpy(sample["gt_valid"]).to(model.device_name).expand_as(truth)
            base = model.segment(features, sample["boxes"], shape)
            baseline_ious = iou_from_logits(base, truth, valid_gt).cpu().tolist()
            for variant_config in row["depth_variants"]:
                name, strength, seed = (variant_config["condition"], variant_config["strength"],
                                        variant_config["seed"])
                corrupted, validity = corrupt_depth(sample["depth"], sample["valid"], name, strength, seed)
                variant = dict(sample, depth=corrupted, valid=validity)
                depth, valid = adapter_depth(variant, model.sam.image_size, model.device_name)
                fused = model.segment(features, sample["boxes"], shape, depth, valid, adapter_on=True)
                fusion_ious = iou_from_logits(fused, truth, valid_gt).cpu().tolist()
                targets = []
                for target, b, f in zip(row["targets"], baseline_ious, fusion_ious):
                    delta = f - b
                    targets.append({"id": target["id"], "box": target["box"], "baseline_iou": b,
                                    "fusion_iou": f, "delta_iou": delta,
                                    "label": int(delta > config["gate"]["epsilon"])})
                stream.write(json.dumps({"split": split, "image_id": row["id"], "group": row["group"],
                                         "condition": name, "strength": strength, "seed": seed,
                                         "targets": targets, "sam_sha256": config["sam"]["checkpoint_sha256"],
                                         "adapter_sha256": adapter_hash, "config_sha256": config_hash,
                                         "preprocessing_sha256": preprocessing_hash,
                                         "manifest_sha256": manifest_hash,
                                         "gate_roi_source_sha256": roi_source_hash,
                                         "gate_roi_config_sha256": roi_config_hash}) + "\n")
            if (index + 1) % 25 == 0:
                stream.flush()
                print(json.dumps({"split": split, "images": index + 1, "total": len(rows),
                                  "elapsed_sec": round(time.time() - began, 1)}), flush=True)
    temp.replace(destination)
    print(json.dumps({"complete": str(destination), "images": len(rows), "elapsed_sec": time.time() - began}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--split", choices=("gate_train", "val", "test"), required=True)
    parser.add_argument("--adapter", default="checkpoints/adapter_best.pt")
    parser.add_argument("--gate-config", default="configs/gate_baseline.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    gate_config = yaml.safe_load(Path(args.gate_config).read_text())
    make_labels(config, args.split, Path(args.adapter), gate_config)


if __name__ == "__main__":
    main()
