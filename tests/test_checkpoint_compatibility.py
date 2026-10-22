"""Checkpoint loading across line endings and optional inference paths."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from selective_sam.infer_baseline import SelectivePredictorBaseline
from selective_sam.label_generation import load_adapter, preprocessing_signature


class FakeModel:
    def __init__(self, *args):
        self.sam = type("SAM", (), {"image_size": 16})()
        self.adapter = self

    def load_state_dict(self, state):
        if state != {}:
            raise AssertionError("Unexpected adapter state")

    def eval(self):
        return self

    def encode_rgb(self, rgb):
        return ()

    def segment(self, features, boxes, shape, *args, adapter_on=False):
        return torch.full((1, *shape), 1.0 if adapter_on else -1.0)


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_preprocessing_signature_is_stable_across_source_line_endings(self):
        names = ("data.py", "samples.py", "corruption.py", "model.py", "gate.py",
                 "label_generation.py", "gate_dataset.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "selective_sam").mkdir()
            for name in names:
                (root / "selective_sam" / name).write_bytes(b"first\nsecond\n")
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                linux_signature = preprocessing_signature({"seed": 1})
                for name in names:
                    (root / "selective_sam" / name).write_bytes(b"first\r\nsecond\r\n")
                self.assertEqual(preprocessing_signature({"seed": 1}), linux_signature)
            finally:
                os.chdir(old_cwd)

    def test_adapter_accepts_crlf_checkout_of_unchanged_model_source(self):
        source = (Path(__file__).resolve().parents[1] / "selective_sam/model.py").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "selective_sam").mkdir()
            (root / "selective_sam/model.py").write_bytes(source.replace(b"\n", b"\r\n"))
            sam = root / "sam.pt"
            sam.write_bytes(b"sam checkpoint")
            adapter = root / "adapter.pt"
            torch.save({"sam_sha256": hashlib.sha256(sam.read_bytes()).hexdigest(),
                        "model_source_sha256": hashlib.sha256(source).hexdigest(),
                        "adapter": {}}, adapter)
            config = {"sam": {"checkpoint": str(sam), "config": "unused", "device": "cpu",
                              "checkpoint_sha256": hashlib.sha256(sam.read_bytes()).hexdigest()}}
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch("selective_sam.label_generation.DepthSAM", FakeModel):
                    self.assertIsInstance(load_adapter(config, adapter), FakeModel)
            finally:
                os.chdir(old_cwd)

    def test_baseline_and_fusion_do_not_require_gate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "final.yaml"
            config.write_text(yaml.safe_dump({"sam": {"device": "cpu"}}))
            with patch("selective_sam.infer_baseline.load_adapter", return_value=FakeModel()):
                predictor = SelectivePredictorBaseline(
                    config_path=config, gate_config_path=root / "missing_gate.yaml",
                    adapter_path=root / "missing_adapter.pt", gate_path=root / "missing_gate.pt",
                    policy_path=root / "missing_policy.json")
            rgb = np.zeros((16, 16, 3), dtype=np.uint8)
            depth = np.ones((16, 16), dtype=np.float32)
            valid = np.ones((16, 16), dtype=bool)
            baseline = predictor.predict(rgb, depth, valid, [1, 1, 8, 8], "baseline")
            fusion = predictor.predict(rgb, depth, valid, [1, 1, 8, 8], "fusion")
            self.assertFalse(baseline["adapter_used"])
            self.assertTrue(fusion["adapter_used"])
            self.assertEqual(int(baseline["mask"].sum()), 0)
            self.assertEqual(int(fusion["mask"].sum()), 16 * 16)

    def test_selective_names_incompatible_gate_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "final.yaml"
            config.write_text(yaml.safe_dump({"sam": {"device": "cpu"}}))
            gate_config = root / "gate.yaml"
            gate_config.write_text(yaml.safe_dump({"architecture": "GateBaselineSAM_v1"}))
            gate = root / "gate.pt"
            torch.save({"architecture": "GateCNN_v1", "gate_config": {"architecture": "GateBaselineSAM_v1"}}, gate)
            with patch("selective_sam.infer_baseline.load_adapter", return_value=FakeModel()):
                predictor = SelectivePredictorBaseline(
                    config_path=config, gate_config_path=gate_config,
                    adapter_path=root / "adapter.pt", gate_path=gate,
                    policy_path=root / "policy.json")
            rgb = np.zeros((16, 16, 3), dtype=np.uint8)
            depth = np.ones((16, 16), dtype=np.float32)
            valid = np.ones((16, 16), dtype=bool)
            with self.assertRaisesRegex(ValueError, "GateCNN_v1.*GateBaselineSAM_v1"):
                predictor.predict(rgb, depth, valid, [1, 1, 8, 8], "selective")

    def test_selective_accepts_crlf_checkout_of_unchanged_gate_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = (Path(__file__).resolve().parents[1] / "selective_sam/gate_baseline.py").read_bytes()
            (root / "selective_sam").mkdir()
            (root / "selective_sam/gate_baseline.py").write_bytes(source.replace(b"\n", b"\r\n"))
            config = root / "final.yaml"
            config.write_text(yaml.safe_dump({"sam": {"device": "cpu", "checkpoint_sha256": "samhash"}}))
            gate_config = root / "gate.yaml"
            gate_config.write_text(yaml.safe_dump({"architecture": "GateBaselineSAM_v1"}))
            adapter = root / "adapter.pt"
            adapter.write_bytes(b"adapter checkpoint")
            adapter_hash = hashlib.sha256(adapter.read_bytes()).hexdigest()
            source_hash = hashlib.sha256(source).hexdigest()
            gate = root / "gate.pt"
            torch.save({"architecture": "GateBaselineSAM_v1", "gate_config": {"architecture": "GateBaselineSAM_v1"},
                        "gate_baseline_source_sha256": source_hash, "adapter_sha256": adapter_hash,
                        "sam_sha256": "samhash", "epoch": 3, "gate": {}}, gate)
            policy = root / "policy.json"
            policy.write_text(json.dumps({"mode": "off", "selected_epoch": 3,
                                          "gate_baseline_source_sha256": source_hash,
                                          "adapter_sha256": adapter_hash, "sam_sha256": "samhash"}))
            class FakeGate:
                def to(self, device):
                    return self
                def load_state_dict(self, state):
                    if state != {}:
                        raise AssertionError("Unexpected gate state")
                def eval(self):
                    return self
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch("selective_sam.infer_baseline.load_adapter", return_value=FakeModel()), \
                     patch("selective_sam.infer_baseline.GateBaselineSAM", FakeGate):
                    predictor = SelectivePredictorBaseline(config_path=config, gate_config_path=gate_config,
                                                           adapter_path=adapter, gate_path=gate, policy_path=policy)
                    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
                    depth = np.ones((16, 16), dtype=np.float32)
                    valid = np.ones((16, 16), dtype=bool)
                    result = predictor.predict(rgb, depth, valid, [1, 1, 8, 8], "selective")
                    self.assertFalse(result["adapter_used"])
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
