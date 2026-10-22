"""Use the frozen baseline SAM mask as a hint before deciding on fusion."""

import cv2
import numpy as np
import torch
from torch import nn

from .gate_roi import roi_window
from .gate_sam import SAM_FEATURE_DIM


def baseline_hint(probability, box, original_hw, global_size=128, local_size=96,
                  margin_fraction=0.15, min_margin_px=16):
    if probability.shape != tuple(original_hw) or not np.isfinite(probability).all():
        raise ValueError("Baseline mask probability must align with original RGB")
    left, top, right, bottom = roi_window(box, original_hw, margin_fraction, min_margin_px)
    global_map = cv2.resize(probability, (global_size, global_size),
                            interpolation=cv2.INTER_AREA)
    local_map = cv2.resize(probability[top:bottom, left:right], (local_size, local_size),
                           interpolation=cv2.INTER_AREA)
    return (np.rint(np.clip(global_map, 0, 1) * 255).astype(np.uint8),
            np.rint(np.clip(local_map, 0, 1) * 255).astype(np.uint8))


class _Stream7(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (7, 24, 48, 96, 128)
        layers = []
        for left, right in zip(channels[:-1], channels[1:]):
            layers.extend((nn.Conv2d(left, right, 3, stride=2, padding=1),
                           nn.BatchNorm2d(right), nn.GELU()))
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        feature = self.features(x)
        return torch.cat((feature.mean(dim=(-2, -1)),
                          feature.amax(dim=(-2, -1))), dim=1)


class GateBaselineSAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_stream = _Stream7()
        self.local_stream = _Stream7()
        self.sam_norm = nn.LayerNorm(SAM_FEATURE_DIM)
        self.head = nn.Sequential(nn.Linear(256 * 2 + 5 + SAM_FEATURE_DIM, 128),
                                  nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 1))

    def forward(self, global_input, local_input, geometry, sam_vector):
        if global_input.shape[1] != 7 or local_input.shape[1] != 7 or sam_vector.shape[1] != SAM_FEATURE_DIM:
            raise ValueError("Expected 7-channel maps and B,640 SAM features")
        combined = torch.cat((self.global_stream(global_input),
                              self.local_stream(local_input), geometry,
                              self.sam_norm(sam_vector)), dim=1)
        return self.head(combined).flatten()
