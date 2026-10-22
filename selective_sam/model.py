from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.GELU(),
    )


class DepthAdapter(nn.Module):
    """A small depth encoder with RGB-conditioned residuals at two SAM scales."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            _block(2, 32),
            _block(32, 64),
            _block(64, 96),
            _block(96, 128),
        )
        self.fuse_8 = nn.Sequential(nn.Conv2d(64 + 96, 96, 1), nn.GELU(), nn.Conv2d(96, 64, 1))
        self.fuse_16 = nn.Sequential(nn.Conv2d(256 + 128, 128, 1), nn.GELU(), nn.Conv2d(128, 256, 1))
        self.scale_8 = nn.Parameter(torch.tensor(0.02))
        self.scale_16 = nn.Parameter(torch.tensor(0.02))

    def forward(self, features: tuple[torch.Tensor, ...], depth: torch.Tensor, valid: torch.Tensor):
        if depth.ndim != 2 or valid.shape != depth.shape:
            raise ValueError("depth and validity must be H,W tensors")
        x = torch.stack((depth, valid.to(depth.dtype)), dim=0).unsqueeze(0)
        x = self.encoder[0](x)
        x = self.encoder[1](x)
        x_8 = self.encoder[2](x)
        x_16 = self.encoder[3](x_8)
        if x_8.shape[-2:] != features[1].shape[-2:] or x_16.shape[-2:] != features[2].shape[-2:]:
            raise ValueError("depth and RGB feature scales are not aligned")
        high = features[1] + self.scale_8 * self.fuse_8(torch.cat((features[1], x_8), dim=1))
        low = features[2] + self.scale_16 * self.fuse_16(torch.cat((features[2], x_16), dim=1))
        return (features[0], high, low)


class DepthSAM(nn.Module):
    def __init__(self, config_name: str, checkpoint: Path, device: str = "cuda:0"):
        super().__init__()
        from sam2.build_sam import build_sam2
        from sam2.utils.transforms import SAM2Transforms

        self.sam = build_sam2(config_name, str(checkpoint), device=device, apply_postprocessing=False)
        self.sam.requires_grad_(False).eval()
        self.adapter = DepthAdapter().to(device)
        self.transforms = SAM2Transforms(self.sam.image_size, 0.0)
        self.device_name = device

    def train(self, mode: bool = True):
        super().train(mode)
        self.sam.eval()
        return self

    def encode_rgb(self, rgb: np.ndarray) -> tuple[torch.Tensor, ...]:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("RGB must be uint8 H,W,3")
        input_image = self.transforms(rgb)[None].to(self.device_name)
        with torch.no_grad():
            backbone = self.sam.forward_image(input_image)
            _, flattened, _, sizes = self.sam._prepare_backbone_features(backbone)
            if self.sam.directly_add_no_mem_embed:
                flattened[-1] = flattened[-1] + self.sam.no_mem_embed
            features = tuple(
                feature.permute(1, 2, 0).view(1, -1, *size)
                for feature, size in zip(flattened, sizes)
            )
        return features

    def segment(
        self,
        features: tuple[torch.Tensor, ...],
        boxes: np.ndarray,
        original_hw: tuple[int, int],
        depth: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        adapter_on: bool = False,
        low_res: bool = False,
    ) -> torch.Tensor:
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=self.device_name)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] < 1:
            raise ValueError("boxes must be N,4 XYXY")
        if adapter_on and depth is not None and valid is not None and bool(valid.any()):
            if not torch.isfinite(depth).all():
                raise ValueError("non-finite depth")
            features = self.adapter(features, depth, valid)
        with torch.no_grad():
            transformed = self.transforms.transform_boxes(boxes, normalize=True, orig_hw=original_hw)
            labels = torch.tensor([[2, 3]], device=boxes.device, dtype=torch.int).repeat(boxes.shape[0], 1)
            sparse, dense = self.sam.sam_prompt_encoder(
                points=(transformed, labels), boxes=None, masks=None
            )
            image_pe = self.sam.sam_prompt_encoder.get_dense_pe()
        masks, _, _, _ = self.sam.sam_mask_decoder(
            image_embeddings=features[2],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=boxes.shape[0] > 1,
            high_res_features=[features[0], features[1]],
        )
        if low_res:
            return masks[:, 0]
        return self.transforms.postprocess_masks(masks, original_hw)[:, 0]


def iou_from_logits(logits: torch.Tensor, target: torch.Tensor, valid_gt: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != valid_gt.shape:
        raise ValueError("IoU inputs must have identical shapes")
    pred = logits > 0
    intersection = ((pred & target) & valid_gt).flatten(1).sum(1)
    union = ((pred | target) & valid_gt).flatten(1).sum(1)
    return torch.where(union > 0, intersection.float() / union.clamp_min(1), torch.ones_like(union, dtype=torch.float32))


def binary_loss(logits: torch.Tensor, target: torch.Tensor, valid_gt: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != valid_gt.shape:
        raise ValueError("loss inputs must have identical shapes")
    loss_map = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    mask = valid_gt.float()
    bce = (loss_map * mask).sum() / mask.sum().clamp_min(1)
    probabilities = torch.sigmoid(logits.float()) * mask
    truth = target.float() * mask
    dice = 1 - (2 * (probabilities * truth).sum() + 1) / (probabilities.sum() + truth.sum() + 1)
    return bce + dice
