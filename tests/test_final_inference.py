import unittest

import numpy as np
import torch

from selective_sam.infer_baseline import SelectivePredictorBaseline


class FakeModel:
    def __init__(self):
        self.sam = type("SAM", (), {"image_size": 1024})()
        self.device_name = "cpu"
        self.adapter_calls = 0

    def encode_rgb(self, rgb):
        return (None,)

    def segment(self, features, boxes, shape, *args, adapter_on=False):
        if adapter_on:
            self.adapter_calls += 1
        return torch.zeros((1, *shape))


class FinalInferenceRoutingTests(unittest.TestCase):
    def test_baseline_hint_reuses_off_mask_and_only_runs_adapter_for_on(self):
        class HintModel(FakeModel):
            def __init__(self):
                super().__init__()
                self.encode_calls = 0
                self.segment_calls = 0

            def encode_rgb(self, rgb):
                self.encode_calls += 1
                return (torch.empty(0), torch.ones((1, 64, 4, 4)),
                        torch.ones((1, 256, 2, 2)))

            def segment(self, features, boxes, shape, *args, adapter_on=False):
                self.segment_calls += 1
                if adapter_on:
                    self.adapter_calls += 1
                return torch.zeros((1, *shape))

        predictor = SelectivePredictorBaseline.__new__(SelectivePredictorBaseline)
        predictor.config = {"sam": {"device": "cpu"}}
        predictor.gate_config = {"global_size": 8, "local_size": 8,
                                 "roi_margin_fraction": .15, "roi_min_margin_px": 1}
        predictor.model = HintModel()
        predictor.policy = {"mode": "selective", "tau": .5}
        calls = []

        def gate(*inputs):
            calls.append((inputs[0].shape, inputs[1].shape))
            return torch.tensor([-10.0])

        predictor.gate = gate
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        depth = np.ones((16, 16), dtype=np.float32)
        valid = np.ones((16, 16), dtype=bool)
        result = predictor.predict(rgb, depth, valid, [4, 4, 12, 12])
        self.assertFalse(result["adapter_used"])
        self.assertEqual((predictor.model.encode_calls, predictor.model.segment_calls,
                          predictor.model.adapter_calls), (1, 1, 0))
        self.assertEqual(calls, [(torch.Size([1, 7, 8, 8]), torch.Size([1, 7, 8, 8]))])
        result = predictor.predict(rgb, None, None, [4, 4, 12, 12])
        self.assertFalse(result["adapter_used"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(predictor.model.segment_calls, 2)
        result = predictor.predict(rgb, depth, np.zeros_like(valid), [4, 4, 12, 12])
        self.assertFalse(result["adapter_used"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(predictor.model.segment_calls, 3)
        predictor.gate = lambda *inputs: torch.tensor([10.0])
        result = predictor.predict(rgb, depth, valid, [4, 4, 12, 12])
        self.assertTrue(result["adapter_used"])
        self.assertEqual((predictor.model.encode_calls, predictor.model.segment_calls,
                          predictor.model.adapter_calls), (4, 5, 1))


if __name__ == "__main__":
    unittest.main()
