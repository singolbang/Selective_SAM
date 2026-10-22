import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .data import decode_depth, decode_semantic, resize_depth_valid


def read_manifest(split: str) -> list[dict]:
    return [json.loads(line) for line in (Path("manifests") / f"{split}.jsonl").read_text().splitlines()]


def load_rgb(row: dict, root: Path) -> np.ndarray:
    with Image.open(root / row["rgb"]) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    if list(rgb.shape[:2]) != row["shape"]:
        raise ValueError(f"RGB shape changed: {row['id']}")
    return rgb


def load_sample(row: dict, root: Path, with_targets: bool = True, check_hash: bool = False) -> dict:
    rgb = load_rgb(row, root)
    depth, valid = decode_depth(root / row["depth"])
    label = decode_semantic(root / row["semantic"])
    h, w = rgb.shape[:2]
    if depth.shape != (h, w) or label.shape != (h, w) or row["shape"] != [h, w]:
        raise ValueError(f"Sample alignment changed: {row['id']}")
    sample = {"rgb": rgb, "depth": depth, "valid": valid, "gt_valid": (label != 0) & (label != 255),
              "boxes": np.asarray([t["box"] for t in row["targets"]], dtype=np.float32),
              "shape": (h, w)}
    if with_targets:
        components = {}
        masks = []
        for target in row["targets"]:
            class_id = target["class_id"]
            if class_id not in components:
                _, components[class_id] = cv2.connectedComponents((label == class_id).astype(np.uint8), 8)
            mask = components[class_id] == target["component_id"]
            if check_hash and hashlib.sha256(mask.tobytes()).hexdigest() != target["mask_sha256"]:
                raise ValueError(f"Target identity changed: {target['id']}")
            masks.append(mask)
        sample["masks"] = np.stack(masks)
    return sample


def pack_clean_sample(sample: dict) -> dict:
    """Lossless compact CPU cache for repeatedly used clean Stage A samples."""
    h, w = sample["shape"]
    n = len(sample["masks"])
    if not np.allclose(sample["depth"] * 255, np.round(sample["depth"] * 255), atol=1e-4):
        raise ValueError("Only released uint8 depth may use this cache")
    return {"depth_u8": np.round(sample["depth"] * 255).astype(np.uint8),
            "valid_bits": np.packbits(sample["valid"].reshape(-1)),
            "gt_valid_bits": np.packbits(sample["gt_valid"].reshape(-1)),
            "mask_bits": np.packbits(sample["masks"].reshape(-1)),
            "shape": (h, w), "targets": n, "boxes": sample["boxes"]}


def unpack_clean_sample(packed: dict) -> dict:
    h, w = packed["shape"]
    pixels = h * w
    return {"rgb": None, "depth": packed["depth_u8"].astype(np.float32) / 255.0,
            "valid": np.unpackbits(packed["valid_bits"], count=pixels).reshape(h, w).astype(bool),
            "gt_valid": np.unpackbits(packed["gt_valid_bits"], count=pixels).reshape(h, w).astype(bool),
            "masks": np.unpackbits(packed["mask_bits"], count=packed["targets"] * pixels)
            .reshape(packed["targets"], h, w).astype(bool),
            "boxes": packed["boxes"], "shape": (h, w)}


def adapter_depth(sample: dict, size: int, device: str):
    depth, valid = resize_depth_valid(sample["depth"], sample["valid"], (size, size))
    return (torch.as_tensor(depth, dtype=torch.float32, device=device),
            torch.as_tensor(valid, dtype=torch.bool, device=device))


def lowres_truth(sample: dict, shape: tuple[int, int], device: str):
    import torch.nn.functional as F
    masks = torch.from_numpy(sample["masks"].astype(np.float32))[:, None].to(device)
    valid = torch.from_numpy(sample["gt_valid"].astype(np.float32))[None, None].to(device)
    masks = F.interpolate(masks, shape, mode="nearest")[:, 0] > .5
    valid = F.interpolate(valid, shape, mode="nearest")[:, 0] > .5
    return masks, valid.expand_as(masks)
