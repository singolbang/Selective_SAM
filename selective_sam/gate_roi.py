"""Target-focused RGB/depth gate without candidate masks or experiment metadata."""

import cv2
import numpy as np
import torch

from .data import resize_depth_valid


def roi_window(box, shape, margin_fraction=0.15, min_margin_px=16):
    h, w = shape
    x1, y1, x2, y2 = (float(value) for value in box)
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("Box lies outside image")
    mx = max(min_margin_px, (x2 - x1) * margin_fraction)
    my = max(min_margin_px, (y2 - y1) * margin_fraction)
    left = max(0, int(np.floor(x1 - mx)))
    top = max(0, int(np.floor(y1 - my)))
    right = min(w, int(np.ceil(x2 + mx)))
    bottom = min(h, int(np.ceil(y2 + my)))
    return left, top, right, bottom


def box_geometry(box, shape):
    h, w = shape
    x1, y1, x2, y2 = (float(value) for value in box)
    return np.array([x1 / w, y1 / h, x2 / w, y2 / h,
                     np.sqrt((x2 - x1) * (y2 - y1) / (w * h))], dtype=np.float32)


def roi_input(rgb, depth, valid, box, size=96, margin_fraction=0.15,
              min_margin_px=16):
    """Return 6 uint8 crop channels and normalized box geometry."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected uint8 H,W,3 RGB")
    if rgb.shape[:2] != depth.shape or depth.shape != valid.shape:
        raise ValueError("RGB, depth and validity are not aligned")
    left, top, right, bottom = roi_window(box, rgb.shape[:2], margin_fraction,
                                           min_margin_px)
    r = rgb[top:bottom, left:right]
    d = depth[top:bottom, left:right]
    v = valid[top:bottom, left:right]
    output = np.empty((6, size, size), dtype=np.uint8)
    output[:3] = cv2.resize(r, (size, size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    resized_depth, resized_valid = resize_depth_valid(d, v, (size, size))
    output[3] = np.round(np.clip(resized_depth, 0, 1) * 255).astype(np.uint8)
    # Area interpolation retains the fraction of valid support after downsampling.
    output[4] = cv2.resize(v.astype(np.uint8) * 255, (size, size),
                           interpolation=cv2.INTER_AREA)
    output[3][~resized_valid | (output[4] == 0)] = 0
    output[5] = 0
    x1, y1, x2, y2 = (float(value) for value in box)
    sx = size / (right - left)
    sy = size / (bottom - top)
    px1 = max(0, int(np.floor((x1 - left) * sx)))
    py1 = max(0, int(np.floor((y1 - top) * sy)))
    px2 = min(size, int(np.ceil((x2 - left) * sx)))
    py2 = min(size, int(np.ceil((y2 - top) * sy)))
    output[5, py1:py2, px1:px2] = 255
    return output, box_geometry(box, rgb.shape[:2])
