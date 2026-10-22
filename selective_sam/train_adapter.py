"""Stage A: train only the depth encoder and residual adapter on clean depth."""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .model import DepthSAM, binary_loss, iou_from_logits
from .label_generation import source_sha256
from .samples import (adapter_depth, load_sample, lowres_truth, pack_clean_sample,
                      read_manifest, unpack_clean_sample, load_rgb)


def _state_digest(module):
    digest = hashlib.sha256()
    for key, tensor in module.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def cached_rgb_features(model, image_id, rgb, cache, cacheable):
    # The cache exists only for this process and this immutable SAM instance.
    if image_id not in cache:
        if rgb is None:
            raise ValueError(f"RGB needed for initial feature encoding: {image_id}")
        encoded = model.encode_rgb(rgb)
        if image_id in cacheable:
            cache[image_id] = tuple(t.detach().cpu().contiguous() for t in encoded)
        return encoded
    return tuple(t.to(model.device_name, non_blocking=True) for t in cache[image_id])


def evaluate_clean(model, rows, root, baseline_cache=None, feature_cache=None,
                   sample_cache=None, cacheable=None):
    model.eval()
    baseline_cache = {} if baseline_cache is None else baseline_cache
    feature_cache = {} if feature_cache is None else feature_cache
    sample_cache = {} if sample_cache is None else sample_cache
    cacheable = set() if cacheable is None else cacheable
    baseline_scores, fusion_scores, improved, worsened = [], [], 0, 0
    with torch.inference_mode():
        for index, row in enumerate(rows):
            if row["id"] in sample_cache:
                sample = unpack_clean_sample(sample_cache[row["id"]])
            else:
                sample = load_sample(row, root, check_hash=True)
                sample_cache[row["id"]] = pack_clean_sample(sample)
            rgb = sample["rgb"] if sample["rgb"] is not None else (
                None if row["id"] in feature_cache else load_rgb(row, root))
            features = cached_rgb_features(model, row["id"], rgb, feature_cache, cacheable)
            shape = sample["shape"]
            truth = torch.from_numpy(sample["masks"]).to(model.device_name)
            valid_gt = torch.from_numpy(sample["gt_valid"]).to(model.device_name).expand_as(truth)
            if row["id"] not in baseline_cache:
                base = model.segment(features, sample["boxes"], shape)
                baseline_cache[row["id"]] = iou_from_logits(base, truth, valid_gt).cpu().tolist()
            depth, valid = adapter_depth(sample, model.sam.image_size, model.device_name)
            fused = model.segment(features, sample["boxes"], shape, depth, valid, adapter_on=True)
            f_scores = iou_from_logits(fused, truth, valid_gt).cpu().tolist()
            b_scores = baseline_cache[row["id"]]
            baseline_scores.extend(b_scores)
            fusion_scores.extend(f_scores)
            improved += sum(f > b for f, b in zip(f_scores, b_scores))
            worsened += sum(f < b for f, b in zip(f_scores, b_scores))
    n = len(fusion_scores)
    return {"images": len(rows), "targets": n, "baseline_iou": float(np.mean(baseline_scores)),
            "fusion_iou": float(np.mean(fusion_scores)), "delta_iou": float(np.mean(fusion_scores) - np.mean(baseline_scores)),
            "improved_fraction": improved / n, "worsened_fraction": worsened / n}, baseline_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    device = config["sam"]["device"]
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.set_num_threads(4)
    root = Path(config["data"]["root"])
    train_rows = read_manifest("adapter_train")
    val_rows = read_manifest("val")
    run_signature = hashlib.sha256(
        Path(args.config).read_bytes() + Path("manifests/adapter_train.jsonl").read_bytes()
        + Path("manifests/val.jsonl").read_bytes()
        + Path("selective_sam/train_adapter.py").read_bytes()
        + Path("selective_sam/samples.py").read_bytes()
        + Path("selective_sam/model.py").read_bytes()).hexdigest()
    if hashlib.sha256(Path(config["sam"]["checkpoint"]).read_bytes()).hexdigest() != config["sam"]["checkpoint_sha256"]:
        raise ValueError("Actual SAM checkpoint does not match configured SHA256")
    model = DepthSAM(config["sam"]["config"], Path(config["sam"]["checkpoint"]), device)
    sam_before = _state_digest(model.sam)
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=config["adapter"]["learning_rate"],
                                  weight_decay=config["adapter"]["weight_decay"])
    warmup = config["adapter"]["warmup_epochs"]
    total = config["adapter"]["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: (.1 + .9 * (epoch + 1) / warmup) if epoch < warmup
        else .5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, total - warmup))))
    start_epoch, best_iou, stale = 0, -1.0, 0
    run_dir = Path("runs/adapter")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("run_signature") != run_signature:
            raise ValueError("Resume checkpoint does not match this config and manifests")
        model.adapter.load_state_dict(saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch, best_iou, stale = saved["epoch"], saved["best_iou"], saved["stale"]
    baseline_cache = None
    feature_cache = {}
    sample_cache = {}
    def fixed_cache_ids(rows, count):
        ordered = sorted(rows, key=lambda row: hashlib.sha256(row["id"].encode()).digest())
        return {row["id"] for row in ordered[:count]}
    cacheable = fixed_cache_ids(train_rows, config["adapter"]["feature_cache_train_images"])
    cacheable |= fixed_cache_ids(val_rows, config["adapter"]["feature_cache_val_images"])
    for epoch in range(start_epoch, config["adapter"]["epochs"]):
        model.train()
        indices = list(range(len(train_rows)))
        random.Random(config["seed"] + epoch).shuffle(indices)
        losses = []
        began = time.time()
        for step, index in enumerate(indices):
            row = train_rows[index]
            if row["id"] in sample_cache:
                sample = unpack_clean_sample(sample_cache[row["id"]])
            else:
                sample = load_sample(row, root, check_hash=True)
                sample_cache[row["id"]] = pack_clean_sample(sample)
            rgb = sample["rgb"] if sample["rgb"] is not None else (
                None if row["id"] in feature_cache else load_rgb(row, root))
            features = cached_rgb_features(model, row["id"], rgb, feature_cache, cacheable)
            depth, valid = adapter_depth(sample, model.sam.image_size, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.segment(features, sample["boxes"], sample["shape"],
                                   depth, valid, adapter_on=True, low_res=True)
            truth, gt_valid = lowres_truth(sample, logits.shape[-2:], device)
            if bool((truth.flatten(1).sum(1) == 0).any()):
                logits = F.interpolate(logits[:, None], size=sample["shape"],
                                       mode="bilinear", align_corners=False)[:, 0]
                truth = torch.from_numpy(sample["masks"]).to(device)
                gt_valid = torch.from_numpy(sample["gt_valid"]).to(device).expand_as(truth)
            loss = binary_loss(logits, truth, gt_valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite adapter loss: {row['id']}")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (step + 1) % 100 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step + 1,
                                  "total": len(indices), "loss_100": float(np.mean(losses[-100:])),
                                  "elapsed_sec": round(time.time() - began, 1)}), flush=True)
        scheduler.step()
        metrics, baseline_cache = evaluate_clean(model, val_rows, root, baseline_cache,
                                                 feature_cache, sample_cache, cacheable)
        improved = metrics["fusion_iou"] > best_iou + 1e-6
        stale = 0 if improved else stale + 1
        if improved:
            best_iou = metrics["fusion_iou"]
            best_path = Path("checkpoints/adapter_best.pt")
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"adapter": model.adapter.state_dict(), "architecture": "DepthAdapter_v1",
                        "sam_checkpoint": config["sam"]["checkpoint"],
                        "sam_sha256": config["sam"]["checkpoint_sha256"],
                        "sam_source_commit": config["sam"]["official_code_commit"],
                        "deliver_source_commit": config["data"]["official_code_commit"],
                        "model_source_sha256": source_sha256(Path("selective_sam/model.py")),
                        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                        "epoch": epoch + 1, "validation": metrics}, best_path)
        torch.save({"adapter": model.adapter.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch + 1,
                    "best_iou": best_iou, "stale": stale,
                    "run_signature": run_signature}, run_dir / "resume.pt")
        record = {"epoch": epoch + 1, "train_images": len(train_rows), "train_targets": sum(len(r["targets"]) for r in train_rows),
                  "train_loss": float(np.mean(losses)), "epoch_sec": time.time() - began, **metrics,
                  "cached_rgb_images": len(feature_cache),
                  "best": improved, "sam_state_unchanged": _state_digest(model.sam) == sam_before}
        with (run_dir / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if not record["sam_state_unchanged"]:
            raise RuntimeError("Frozen SAM state changed")
        if epoch + 1 >= config["adapter"]["min_epochs"] and stale >= config["adapter"]["patience"]:
            print(f"Early stopping at epoch {epoch + 1}", flush=True)
            break


if __name__ == "__main__":
    main()
