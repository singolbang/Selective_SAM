"""Small RGB/depth/validity/box selector; no candidate masks as input."""

import cv2
import numpy as np
import torch

from .data import resize_depth_valid


def gate_base(rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    if rgb.shape[:2] != depth.shape or depth.shape != valid.shape:
        raise ValueError("Gate input alignment mismatch")
    rgb_small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    d, v = resize_depth_valid(depth, valid, (size, size))
    output = np.empty((5, size, size), dtype=np.uint8)
    output[:3] = rgb_small.transpose(2, 0, 1)
    output[3] = np.round(np.clip(d, 0, 1) * 255).astype(np.uint8)
    output[4] = (v.astype(np.uint8) * 255)
    return output


def gate_input(base: np.ndarray, box: list[float], original_hw: tuple[int, int]) -> torch.Tensor:
    size = base.shape[-1]
    h, w = original_hw
    x1, y1, x2, y2 = box
    x1, x2 = int(np.floor(x1 * size / w)), int(np.ceil(x2 * size / w))
    y1, y2 = int(np.floor(y1 * size / h)), int(np.ceil(y2 * size / h))
    mask = np.zeros((1, size, size), dtype=np.float32)
    mask[:, max(0, y1):min(size, y2), max(0, x1):min(size, x2)] = 1
    return torch.from_numpy(np.concatenate((base.astype(np.float32) / 255.0, mask), axis=0))
