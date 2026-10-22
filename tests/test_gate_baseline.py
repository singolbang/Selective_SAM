import unittest

import numpy as np
import torch

from selective_sam.gate_baseline import GateBaselineSAM, baseline_hint


class BaselineHintTests(unittest.TestCase):
    def test_hint_preserves_fixed_global_and_box_crop_channels(self):
        probability = np.arange(16, dtype=np.float32).reshape(4, 4) / 15
        global_hint, local_hint = baseline_hint(
            probability, [1, 1, 3, 3], (4, 4), global_size=4, local_size=2,
            margin_fraction=0, min_margin_px=0)
        np.testing.assert_array_equal(global_hint, np.arange(16, dtype=np.uint8).reshape(4, 4) * 17)
        np.testing.assert_array_equal(local_hint,
                                      np.array([[5, 6], [9, 10]], dtype=np.uint8) * 17)

    def test_mask_and_sam_feature_both_reach_gate_head(self):
        model = GateBaselineSAM()
        global_input = torch.rand(2, 7, 128, 128)
        local_input = torch.rand(2, 7, 96, 96)
        geometry = torch.rand(2, 5)
        sam_vector = torch.rand(2, 640)
        output = model(global_input, local_input, geometry, sam_vector)
        self.assertEqual(output.shape, (2,))
        output.square().mean().backward()
        self.assertGreater(float(model.global_stream.features[0].weight.grad[:, 6].abs().sum()), 0)
        self.assertGreater(float(model.local_stream.features[0].weight.grad[:, 6].abs().sum()), 0)
        self.assertGreater(float(model.head[0].weight.grad[:, -640:].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
