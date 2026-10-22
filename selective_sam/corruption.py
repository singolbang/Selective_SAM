"""Deterministic depth-only perturbations on the fixed 0..1 released scale."""

import hashlib
import numpy as np


def corruption_seed(global_seed: int, image_id: str, condition: str) -> int:
    key = f"{global_seed}|{image_id}|{condition}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def conditions(config: dict) -> list[tuple[str, float]]:
    output = [("clean", 0.0)]
    for kind in ("noise", "pixel_dropout", "region_dropout"):
        output.extend((f"{kind}_{level}", float(strength))
                      for level, strength in zip(("weak", "medium", "strong"), config["corruptions"][kind]))
    return output


def conditions_for_image(config: dict, image_id: str) -> list[tuple[str, float]]:
    """Every image has clean plus one balanced, image-keyed damage variant."""
    all_conditions = conditions(config)
    if config["corruptions"]["damaged_variants_per_image"] != 1:
        raise ValueError("This run implements exactly one damaged variant per image")
    index = corruption_seed(config["seed"], image_id, "condition_assignment") % 9
    return [all_conditions[0], all_conditions[1 + index]]


def corrupt_depth(depth: np.ndarray, valid: np.ndarray, condition: str, strength: float, seed: int):
    if depth.shape != valid.shape or depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError("Expected aligned, finite H,W depth/validity")
    d = depth.copy().astype(np.float32)
    v = valid.copy().astype(np.bool_)
    rng = np.random.default_rng(seed)
    if condition == "clean":
        return d, v
    if condition.startswith("noise_"):
        d[v] = np.clip(d[v] + rng.normal(0, strength, size=int(v.sum())), 0, 1)
    elif condition.startswith("pixel_dropout_"):
        v &= rng.random(v.shape) >= strength
        d[~v] = 0
    elif condition.startswith("region_dropout_"):
        h, w = d.shape
        edge = np.sqrt(strength)
        rh, rw = max(1, round(h * edge)), max(1, round(w * edge))
        top = int(rng.integers(0, h - rh + 1))
        left = int(rng.integers(0, w - rw + 1))
        v[top:top + rh, left:left + rw] = False
        d[~v] = 0
    else:
        raise ValueError(f"Unknown depth corruption: {condition}")
    return d, v
