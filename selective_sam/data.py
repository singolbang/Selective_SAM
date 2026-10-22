from pathlib import Path
import hashlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def official_full_records(source: dict[str, list[dict]], seed: int):
    """Use every official image, reusing train for both stages; audit split leakage."""
    groups = {split: {row["group"] for row in source[split]} for split in ("train", "val", "test")}
    overlaps = sorted((groups["train"] & groups["val"]) |
                      (groups["train"] & groups["test"]) |
                      (groups["val"] & groups["test"]))
    hashes = {split: {row["rgb_sha256"] for row in source[split] if "rgb_sha256" in row}
              for split in ("train", "val", "test")}
    duplicate_hashes = sorted((hashes["train"] & hashes["val"]) |
                              (hashes["train"] & hashes["test"]) |
                              (hashes["val"] & hashes["test"]))
    exclude_val = {row["group"] for row in source["val"]
                   if row.get("rgb_sha256") in hashes["test"]}
    exclude_train = {row["group"] for row in source["train"]
                     if row.get("rgb_sha256") in hashes["val"] | hashes["test"]}
    # Resolve before inspecting model outputs: test > validation > train.
    eligible = {
        "test": source["test"],
        "val": [row for row in source["val"] if row["group"] not in groups["test"] | exclude_val],
        "train": [row for row in source["train"]
                  if row["group"] not in groups["test"] | groups["val"] | exclude_train],
    }
    selected = {
        "adapter_train": sorted(eligible["train"], key=lambda row: row["id"]),
        "gate_train": sorted(eligible["train"], key=lambda row: row["id"]),
        "val": sorted(eligible["val"], key=lambda row: row["id"]),
        "test": sorted(eligible["test"], key=lambda row: row["id"]),
    }
    audit = {
        "seed": seed,
        "official_images": {key: len(rows) for key, rows in source.items()},
        "official_cross_split_groups": overlaps,
        "cross_split_identical_rgb_hashes": duplicate_hashes,
        "resolution": "test > val > train; exclude lower-priority entire scenes sharing group or exact RGB bytes",
        "intentional_train_reuse": True,
        "selected_groups": {key: sorted({row["group"] for row in rows}) for key, rows in selected.items()},
        "selected_images": {key: len(rows) for key, rows in selected.items()},
    }
    return selected, audit


def decode_depth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read DeLiVER's distributed one-channel uint8 depth frame, never HHA."""
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"{path}: expected one-channel uint8 depth, got {image.mode}")
        raw = np.asarray(image, dtype=np.uint8)
    # The release documents 0..255 log-scale depth frames and no missing sentinel.
    # Preserve all source pixels as valid; synthetic dropout changes this mask later.
    return raw.astype(np.float32) / 255.0, np.ones(raw.shape, dtype=np.bool_)


def decode_semantic(path: Path) -> np.ndarray:
    """Follow the official DeLiVER loader: semantic ID is PNG channel zero."""
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim not in (2, 3):
        raise ValueError(f"{path}: unsupported semantic encoding {array.shape} {array.dtype}")
    return array if array.ndim == 2 else array[:, :, 0]


def resize_depth_valid(depth: np.ndarray, valid: np.ndarray, size: tuple[int, int]):
    if depth.shape != valid.shape or depth.ndim != 2:
        raise ValueError("depth and validity must be aligned H,W arrays")
    if not np.isfinite(depth).all():
        raise ValueError("non-finite depth")
    d = torch.from_numpy(depth.astype(np.float32))[None, None]
    v = torch.from_numpy(valid.astype(np.float32))[None, None]
    numerator = F.interpolate(d * v, size=size, mode="bilinear", align_corners=False)
    denominator = F.interpolate(v, size=size, mode="bilinear", align_corners=False)
    output_valid = denominator > 1e-6
    output = torch.where(output_valid, numerator / denominator.clamp_min(1e-6), 0)
    return output[0, 0].numpy(), output_valid[0, 0].numpy()


def scene_group_from_path(rgb_path: Path, root: Path) -> str:
    parts = Path(rgb_path).relative_to(root).parts
    if len(parts) != 5 or parts[0] != "img" or parts[2] not in ("train", "val", "test"):
        raise ValueError(f"Unexpected DeLiVER RGB path: {rgb_path}")
    return parts[3]


def targets_from_labels(label: np.ndarray, min_area: int, max_targets: int,
                        small_fraction: float = .005, medium_fraction: float = .05) -> list[dict]:
    if label.ndim != 2 or not (((label >= 0) & (label <= 25)) | (label == 255)).all():
        raise ValueError("Expected semantic IDs 0..25 or 255 in H,W label")
    if min_area < 1 or max_targets < 1 or not 0 < small_fraction < medium_fraction < 1:
        raise ValueError("min_area and max_targets must be positive")
    by_class = {}
    for class_id in (int(value) for value in np.unique(label) if 1 <= value <= 25):
        binary = (label == class_id).astype(np.uint8)
        if not binary.any():
            continue
        count, components, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        entries = []
        for index in range(1, count):
            x, y, width, height, area = (int(v) for v in stats[index])
            if area < min_area:
                continue
            mask = components == index
            entries.append(
                {
                    "class_id": class_id,
                    "component_id": index,
                    "area": area,
                    "box": [x, y, x + width, y + height],
                    "mask": mask,
                    "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                }
            )
        if entries:
            by_class[class_id] = sorted(entries, key=lambda t: (-t["area"], t["box"]))
    all_entries = [entry for entries in by_class.values() for entry in entries]
    total_pixels = label.size
    buckets = {"small": [], "medium": [], "large": []}
    for entry in all_entries:
        fraction = entry["area"] / total_pixels
        size_bin = "small" if fraction < small_fraction else "medium" if fraction < medium_fraction else "large"
        buckets[size_bin].append(entry)
    for entries in buckets.values():
        entries.sort(key=lambda item: (-item["area"], item["class_id"], item["box"]))
    chosen = []
    used_classes = set()
    for size_bin in ("medium", "small", "large"):
        candidates = buckets[size_bin]
        if not candidates or len(chosen) >= max_targets:
            continue
        entry = next((item for item in candidates if item["class_id"] not in used_classes), candidates[0])
        chosen.append(entry)
        used_classes.add(entry["class_id"])
    remaining = [entry for entry in all_entries if not any(entry is selected for selected in chosen)]
    remaining.sort(key=lambda item: (item["class_id"] in used_classes, -item["area"], item["class_id"]))
    chosen.extend(remaining[:max_targets - len(chosen)])
    return chosen
