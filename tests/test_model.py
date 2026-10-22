import unittest
from pathlib import Path

import numpy as np
import torch

from selective_sam.model import DepthSAM


CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT = Path("checkpoints/sam2.1_hiera_tiny.pt")


@unittest.skipUnless(torch.cuda.is_available() and CHECKPOINT.exists(), "SAM2 GPU assets")
class SAMIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = DepthSAM(CONFIG, CHECKPOINT, "cuda:0")

    def test_off_matches_official_box_predictor(self):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        rgb = np.random.default_rng(4).integers(0, 256, (1042, 1042, 3), np.uint8)
        box = np.array([[250, 240, 700, 750]], dtype=np.float32)
        official = SAM2ImagePredictor(self.model.sam)
        with torch.inference_mode():
            official.set_image(rgb)
            expected, _, _ = official.predict(box=box[0], multimask_output=False, return_logits=True)
            features = self.model.encode_rgb(rgb)
            actual = self.model.segment(features, box, rgb.shape[:2], adapter_on=False)
        self.assertEqual(tuple(actual.shape), (1, 1042, 1042))
        self.assertTrue(np.allclose(actual.cpu().numpy(), expected, atol=1e-4))

    def test_train_step_updates_adapter_and_preserves_every_sam_tensor(self):
        rgb = np.zeros((1042, 1042, 3), dtype=np.uint8)
        depth = torch.ones((1024, 1024), device="cuda:0") * 0.4
        valid = torch.ones_like(depth, dtype=torch.bool)
        box = np.array([[200, 200, 700, 700]], dtype=np.float32)
        before_sam = {k: v.detach().clone() for k, v in self.model.sam.state_dict().items()}
        before_adapter = {k: v.detach().clone() for k, v in self.model.adapter.state_dict().items()}
        optimizer = torch.optim.AdamW(self.model.adapter.parameters(), lr=1e-3)
        with torch.autocast("cuda", dtype=torch.float16):
            features = self.model.encode_rgb(rgb)
            logits = self.model.segment(features, box, rgb.shape[:2], depth, valid, True, low_res=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), torch.ones_like(logits.float())
            )
        optimizer.zero_grad()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.adapter.parameters()))
        optimizer.step()
        self.assertTrue(all(torch.equal(v, before_sam[k]) for k, v in self.model.sam.state_dict().items()))
        self.assertTrue(any(not torch.equal(v, before_adapter[k]) for k, v in self.model.adapter.state_dict().items()))
        self.assertFalse(torch.equal(self.model.adapter.encoder[0][0].weight,
                                     before_adapter["encoder.0.0.weight"]))
        self.assertFalse(torch.equal(self.model.adapter.fuse_16[0].weight,
                                     before_adapter["fuse_16.0.weight"]))
        self.assertTrue(all(p.grad is None for p in self.model.sam.parameters()))

    def test_all_invalid_depth_falls_back_to_off_path(self):
        rgb = np.zeros((1042, 1042, 3), dtype=np.uint8)
        box = np.array([[200, 200, 700, 700]], dtype=np.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            features = self.model.encode_rgb(rgb)
            off = self.model.segment(features, box, rgb.shape[:2], adapter_on=False, low_res=True)
            fallback = self.model.segment(
                features,
                box,
                rgb.shape[:2],
                torch.zeros((1024, 1024), device="cuda:0"),
                torch.zeros((1024, 1024), dtype=torch.bool, device="cuda:0"),
                True,
                low_res=True,
            )
        self.assertTrue(torch.equal(off, fallback))


if __name__ == "__main__":
    unittest.main()
