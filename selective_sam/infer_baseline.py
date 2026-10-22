"""Online selector using one frozen baseline mask as an input hint."""

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .gate import gate_base, gate_input
from .gate_baseline import GateBaselineSAM, baseline_hint
from .gate_roi import roi_input
from .gate_sam import sam_box_feature_tensor
from .label_generation import load_adapter, sha256
from .samples import adapter_depth


class SelectivePredictorBaseline:
    def __init__(self, config_path="configs/final.yaml", gate_config_path="configs/gate_baseline.yaml",
                 adapter_path="checkpoints/adapter_best.pt", gate_path="checkpoints/gate_best.pt",
                 policy_path="checkpoints/selection_policy.json"):
        self.config = yaml.safe_load(Path(config_path).read_text())
        self.gate_config = yaml.safe_load(Path(gate_config_path).read_text())
        self.model = load_adapter(self.config, Path(adapter_path))
        saved = torch.load(gate_path, map_location="cpu", weights_only=False)
        if (saved.get("architecture") != "GateBaselineSAM_v1" or
                saved["gate_config"] != self.gate_config):
            raise ValueError("Baseline-hint gate architecture/config mismatch")
        gate_source_sha = sha256(Path("selective_sam/gate_baseline.py"))
        if saved["gate_baseline_source_sha256"] != gate_source_sha:
            raise ValueError("Baseline-hint gate source changed since training")
        self.policy = json.loads(Path(policy_path).read_text())
        adapter_hash = sha256(Path(adapter_path))
        if (saved["adapter_sha256"] != adapter_hash or
                self.policy["adapter_sha256"] != adapter_hash or
                saved["sam_sha256"] != self.config["sam"]["checkpoint_sha256"] or
                self.policy["sam_sha256"] != self.config["sam"]["checkpoint_sha256"] or
                self.policy["gate_baseline_source_sha256"] != gate_source_sha or
                self.policy["selected_epoch"] != saved["epoch"]):
            raise ValueError("SAM, adapter, gate and policy identifiers do not match")
        if self.policy["mode"] not in ("off", "on", "selective"):
            raise ValueError("Unknown selection policy mode")
        self.gate = GateBaselineSAM().to(self.config["sam"]["device"])
        self.gate.load_state_dict(saved["gate"])
        self.gate.eval()

    def predict(self, rgb, depth, valid, box, mode="selective"):
        if mode not in ("baseline", "fusion", "selective"):
            raise ValueError(mode)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected uint8 RGB H,W,3")
        available = depth is not None and valid is not None and bool(valid.any())
        if available and (depth.shape != rgb.shape[:2] or valid.shape != depth.shape):
            raise ValueError("RGB and depth are misaligned")
        device = self.config["sam"]["device"]
        score = None
        use_adapter = False
        prompt = np.asarray([box], dtype=np.float32)
        with torch.inference_mode():
            features = self.model.encode_rgb(rgb)
            forced_on = available and (mode == "fusion" or
                                        (mode == "selective" and self.policy["mode"] == "on"))
            baseline_logits = None
            # A selective decision needs this mask. Reuse it directly when OFF.
            if not forced_on:
                baseline_logits = self.model.segment(features, prompt, rgb.shape[:2])
            if forced_on:
                use_adapter = True
            elif available and mode == "selective" and self.policy["mode"] == "selective":
                global_base = gate_base(rgb, depth, valid, self.gate_config["global_size"])
                global_input = gate_input(global_base, box, rgb.shape[:2])[None].to(device)
                local_base, geometry = roi_input(
                    rgb, depth, valid, box, self.gate_config["local_size"],
                    self.gate_config["roi_margin_fraction"],
                    self.gate_config["roi_min_margin_px"])
                local_input = torch.from_numpy(local_base.astype(np.float32) / 255.0)[None].to(device)
                geometry = torch.from_numpy(geometry)[None].to(device)
                probability = torch.sigmoid(baseline_logits[0]).cpu().numpy()
                global_hint, local_hint = baseline_hint(
                    probability, box, rgb.shape[:2],
                    self.gate_config["global_size"], self.gate_config["local_size"],
                    self.gate_config["roi_margin_fraction"],
                    self.gate_config["roi_min_margin_px"])
                global_hint = torch.from_numpy(global_hint.astype(np.float32) / 255)[None, None].to(device)
                local_hint = torch.from_numpy(local_hint.astype(np.float32) / 255)[None, None].to(device)
                sam_vector = sam_box_feature_tensor(features, box, rgb.shape[:2])[None].float()
                score = float(torch.sigmoid(self.gate(
                    torch.cat((global_input, global_hint), dim=1),
                    torch.cat((local_input, local_hint), dim=1),
                    geometry, sam_vector)).item())
                use_adapter = score >= self.policy["tau"]
            if use_adapter:
                d, v = adapter_depth({"depth": depth, "valid": valid},
                                     self.model.sam.image_size, device)
                logits = self.model.segment(features, prompt, rgb.shape[:2],
                                            d, v, adapter_on=True)
            else:
                logits = baseline_logits
        return {"mask": (logits[0] > 0).cpu().numpy().astype(np.uint8),
                "gate_score": score, "adapter_used": use_adapter, "mode": mode}
