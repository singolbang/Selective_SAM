"""Frozen SAM RGB feature summaries for the final selector."""

import numpy as np
import torch



SAM_FEATURE_DIM = 640


def _pooled_box(feature: torch.Tensor, box, original_hw):
    h, w = original_hw
    _, channels, feature_h, feature_w = feature.shape
    x1, y1, x2, y2 = (float(value) for value in box)
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("Box lies outside the original RGB image")
    left = max(0, min(feature_w - 1, int(np.floor(x1 * feature_w / w))))
    top = max(0, min(feature_h - 1, int(np.floor(y1 * feature_h / h))))
    right = max(left + 1, min(feature_w, int(np.ceil(x2 * feature_w / w))))
    bottom = max(top + 1, min(feature_h, int(np.ceil(y2 * feature_h / h))))
    return feature[0, :, top:bottom, left:right].mean(dim=(-2, -1))


def sam_box_feature_tensor(features, box, original_hw):
    """Return global/box means at /16 then /8, all from the frozen RGB encoder."""
    high, low = features[1], features[2]
    if (high.ndim != 4 or low.ndim != 4 or high.shape[:2] != (1, 64)
            or low.shape[:2] != (1, 256)):
        raise ValueError("Expected frozen SAM2.1 Tiny /8 and /16 feature maps")
    return torch.cat((low[0].mean(dim=(-2, -1)), _pooled_box(low, box, original_hw),
                      high[0].mean(dim=(-2, -1)), _pooled_box(high, box, original_hw)))


def sam_box_feature(features, box, original_hw):
    return sam_box_feature_tensor(features, box, original_hw).detach().float().cpu().numpy()
