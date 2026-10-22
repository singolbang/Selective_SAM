import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from selective_sam.data import (
    decode_depth,
    decode_semantic,
    resize_depth_valid,
    scene_group_from_path,
    targets_from_labels,
)
from selective_sam.samples import pack_clean_sample, unpack_clean_sample


class RawDepthTests(unittest.TestCase):
    def test_compact_training_cache_preserves_depth_and_masks(self):
        sample = {"rgb": np.zeros((3, 4, 3), dtype=np.uint8),
                  "depth": np.arange(12, dtype=np.uint8).reshape(3, 4).astype(np.float32) / 255,
                  "valid": np.ones((3, 4), dtype=bool),
                  "gt_valid": np.array([[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]], dtype=bool),
                  "masks": np.array([np.eye(3, 4, dtype=bool), np.fliplr(np.eye(3, 4, dtype=bool))]),
                  "boxes": np.array([[0, 0, 3, 3], [1, 0, 4, 3]], dtype=np.float32),
                  "shape": (3, 4)}
        restored = unpack_clean_sample(pack_clean_sample(sample))
        for key in ("depth", "valid", "gt_valid", "masks", "boxes"):
            np.testing.assert_array_equal(restored[key], sample[key])

    def test_one_channel_depth_keeps_values_and_rejects_hha(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.png"
            Image.fromarray(np.array([[0, 127, 255]], dtype=np.uint8)).save(path)
            depth, valid = decode_depth(path)
            np.testing.assert_allclose(depth[0], [0, 127 / 255, 1])
            self.assertTrue(valid.all())
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(path)
            with self.assertRaises(ValueError):
                decode_depth(path)

    def test_rgba_semantic_uses_official_first_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantic.png"
            array = np.zeros((2, 2, 4), dtype=np.uint8)
            array[:, :, 0] = [[0, 7], [13, 255]]
            array[:, :, 3] = 255
            Image.fromarray(array, "RGBA").save(path)
            np.testing.assert_array_equal(decode_semantic(path), array[:, :, 0])

    def test_invalid_depth_does_not_blend_into_valid_depth(self):
        depth = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        valid = np.array([[True, False], [True, False]])
        resized, resized_valid = resize_depth_valid(depth, valid, (1, 1))
        self.assertAlmostEqual(float(resized[0, 0]), 1.0)
        self.assertTrue(bool(resized_valid[0, 0]))


class TargetTests(unittest.TestCase):
    def test_components_exclude_ignore_and_choose_largest_classes(self):
        label = np.array(
            [[0, 1, 1, 0, 2, 2], [255, 1, 1, 0, 2, 2], [3, 3, 0, 0, 0, 0]],
            dtype=np.uint8,
        )
        targets = targets_from_labels(label, min_area=2, max_targets=2)
        self.assertEqual([t["class_id"] for t in targets], [1, 2])
        self.assertEqual([t["box"] for t in targets], [[1, 0, 3, 2], [4, 0, 6, 2]])
        self.assertEqual([int(t["mask"].sum()) for t in targets], [4, 4])
        self.assertFalse(any(t["mask"][0, 0] or t["mask"][1, 0] for t in targets))

    def test_scene_group_uses_path_scene_folder_across_weather(self):
        root = Path("/dataset/DELIVER")
        cloud = root / "img/cloud/train/MAP_10_point72/042350_rgb_front.png"
        fog = root / "img/fog/train/MAP_10_point72/042350_rgb_front.png"
        self.assertEqual(scene_group_from_path(cloud, root), scene_group_from_path(fog, root))
        with self.assertRaises(ValueError):
            scene_group_from_path(root / "hha/cloud/train/MAP_10_point72/a.png", root)

    def test_target_selection_covers_size_bins_without_model_scores(self):
        label = np.zeros((100, 100), dtype=np.uint8)
        label[0:40, 0:40] = 1   # large
        label[50:60, 0:20] = 2  # medium
        label[50:54, 50:55] = 3 # small
        targets = targets_from_labels(label, min_area=10, max_targets=3)
        self.assertEqual({t["class_id"] for t in targets}, {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
