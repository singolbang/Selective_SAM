"""Create immutable scene-disjoint sample manifests before model evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .corruption import conditions_for_image, corruption_seed
from .data import decode_semantic, official_full_records, scene_group_from_path, targets_from_labels


def discover(root: Path):
    source = {split: [] for split in ("train", "val", "test")}
    for rgb in sorted((root / "img").glob("*/*/*/*_rgb_front.png")):
        parts = rgb.relative_to(root).parts
        weather, split = parts[1:3]
        if split not in source:
            continue
        depth = root / "depth" / weather / split / parts[3] / rgb.name.replace("_rgb_", "_depth_")
        label = root / "semantic" / weather / split / parts[3] / rgb.name.replace("_rgb_", "_semantic_")
        if not depth.is_file() or not label.is_file():
            raise FileNotFoundError(f"Unpaired RGB: {rgb}; depth={depth}; semantic={label}")
        source[split].append({
            "id": str(rgb.relative_to(root)), "group": scene_group_from_path(rgb, root),
            "condition": weather, "official_split": split,
            "rgb_sha256": hashlib.sha256(rgb.read_bytes()).hexdigest(),
            "rgb": str(rgb.relative_to(root)), "depth": str(depth.relative_to(root)),
            "semantic": str(label.relative_to(root)),
        })
    return source


def materialize_targets(root: Path, rows: list[dict], min_area: int, max_targets: int,
                        small_fraction: float, medium_fraction: float, reuse: dict[str, dict]):
    output = []
    reused = 0
    target_rule = {"min_area": min_area, "max_targets": max_targets,
                   "small_fraction": small_fraction, "medium_fraction": medium_fraction,
                   "method": "size_bin_distinct_class_v1"}
    legacy_rule = (min_area == 256 and max_targets == 3 and
                   small_fraction == .005 and medium_fraction == .05)
    for index, row in enumerate(rows):
        cached = reuse.get(row["id"])
        if (cached and cached.get("rgb_sha256") == row["rgb_sha256"] and cached.get("targets")
                and (cached.get("target_rule") == target_rule or
                     ("target_rule" not in cached and legacy_rule))):
            cached["target_rule"] = target_rule
            output.append(cached)
            reused += 1
            continue
        with Image.open(root / row["rgb"]) as image:
            rgb_hw = image.size[::-1]
        with Image.open(root / row["depth"]) as image:
            depth_hw = image.size[::-1]
            depth_mode = image.mode
        label = decode_semantic(root / row["semantic"])
        if rgb_hw != depth_hw or rgb_hw != label.shape or depth_mode != "L":
            raise ValueError(f"Unaligned/unsupported input: {row['id']} {rgb_hw} {depth_hw} {label.shape} {depth_mode}")
        targets = targets_from_labels(label, min_area, max_targets, small_fraction, medium_fraction)
        if not targets:
            raise ValueError(f"No valid target for official image {row['id']}; do not silently shrink full split")
        clean = dict(row)
        clean["shape"] = list(rgb_hw)
        clean["target_rule"] = target_rule
        clean["targets"] = [
            {"id": f"{row['id']}:{t['class_id']}:{t['component_id']}",
             "class_id": t["class_id"], "component_id": t["component_id"],
             "area": t["area"], "box": t["box"], "mask_sha256": t["mask_sha256"]}
            for t in targets
        ]
        output.append(clean)
        if (index + 1) % 100 == 0:
            print(f"targets {index + 1}/{len(rows)}", flush=True)
    return output, reused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(config["data"]["root"])
    source = discover(root)
    if not any(source.values()):
        raise FileNotFoundError(f"No DeLiVER front RGB at {root}")
    selected, audit = official_full_records(source, config["seed"])
    path = Path("manifests")
    path.mkdir(exist_ok=True)
    reuse = {}
    for existing in (path / "adapter_train.jsonl", path / "gate_train.jsonl",
                     path / "val.jsonl", path / "test.jsonl"):
        if existing.is_file():
            for line in existing.read_text().splitlines():
                item = json.loads(line)
                reuse[item["id"]] = item
    for split, rows in selected.items():
        materialized, reused = materialize_targets(root, rows, config["data"]["min_target_area"],
                                           config["data"]["max_targets_per_image"],
                                           config["data"]["small_area_fraction_lt"],
                                           config["data"]["medium_area_fraction_lt"], reuse)
        for row in materialized:
            row["depth_variants"] = [
                {"condition": condition, "strength": strength,
                 "seed": corruption_seed(config["seed"], row["id"], condition)}
                for condition, strength in conditions_for_image(config, row["id"])
            ]
        with (path / f"{split}.jsonl").open("w") as stream:
            for row in materialized:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        audit.setdefault("materialized_images", {})[split] = len(materialized)
        audit.setdefault("reused_target_rows", {})[split] = reused
        audit.setdefault("materialized_targets", {})[split] = sum(len(row["targets"]) for row in materialized)
        audit.setdefault("selected_conditions", {})[split] = {
            condition: sum(row["condition"] == condition for row in materialized)
            for condition in sorted({row["condition"] for row in materialized})
        }
        audit.setdefault("depth_variant_conditions", {})[split] = {
            condition: sum(variant["condition"] == condition
                           for row in materialized for variant in row["depth_variants"])
            for condition in sorted({variant["condition"] for row in materialized
                                     for variant in row["depth_variants"]})
        }
        group_damage = {}
        for row in materialized:
            group_damage.setdefault(row["group"], set()).update(
                variant["condition"] for variant in row["depth_variants"] if variant["condition"] != "clean")
        audit.setdefault("groups_with_multiple_damage_conditions", {})[split] = sum(
            len(names) >= 2 for names in group_damage.values())
        audit.setdefault("selected_classes", {})[split] = {
            str(class_id): sum(target["class_id"] == class_id for row in materialized for target in row["targets"])
            for class_id in sorted({target["class_id"] for row in materialized for target in row["targets"]})
        }
        reuse.update({row["id"]: row for row in materialized})
    audit["manifest_sha256"] = {
        split: hashlib.sha256((path / f"{split}.jsonl").read_bytes()).hexdigest()
        for split in selected
    }
    (path / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
