import unittest
import numpy as np
from selective_sam.corruption import corrupt_depth, corruption_seed, conditions_for_image
from selective_sam.infer import assigned_manifest_damage


class CorruptionTests(unittest.TestCase):
    def test_single_sample_damage_replays_manifest_variant(self):
        depth = np.arange(16, dtype=np.float32).reshape(4, 4) / 15
        valid = np.ones((4, 4), dtype=bool)
        variant = {"condition": "region_dropout_weak", "strength": 0.64, "seed": 123}
        row = {"depth_variants": [{"condition": "clean", "strength": 0.0, "seed": 1}, variant]}
        actual_depth, actual_valid, actual_name = assigned_manifest_damage(depth, valid, row)
        expected_depth, expected_valid = corrupt_depth(depth, valid, **variant)
        self.assertEqual(actual_name, variant["condition"])
        np.testing.assert_array_equal(actual_depth, expected_depth)
        np.testing.assert_array_equal(actual_valid, expected_valid)
        np.testing.assert_array_equal(depth, np.arange(16, dtype=np.float32).reshape(4, 4) / 15)

    def test_each_image_keeps_clean_and_one_of_nine_damage_variants(self):
        config = {"seed": 42, "corruptions": {"damaged_variants_per_image": 1,
                  "noise": [.015, .04, .1], "pixel_dropout": [.05, .2, .5],
                  "region_dropout": [.04, .16, .36]}}
        assignments = [conditions_for_image(config, f"scene/frame/{i}") for i in range(100)]
        self.assertTrue(all(len(items) == 2 and items[0] == ("clean", 0.0) for items in assignments))
        self.assertEqual(len({items[1][0] for items in assignments}), 9)

    def test_reproduces_exact_depth_and_keeps_original_unmodified(self):
        original = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
        valid = np.ones((10, 10), dtype=bool)
        for name, strength in (("clean", 0), ("noise_medium", .04),
                               ("pixel_dropout_medium", .2), ("region_dropout_medium", .16)):
            seed = corruption_seed(42, "image-1", name)
            a = corrupt_depth(original, valid, name, strength, seed)
            b = corrupt_depth(original, valid, name, strength, seed)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
        self.assertTrue(valid.all())
        self.assertEqual(original[0, 1], np.float32(1 / 99))


if __name__ == "__main__":
    unittest.main()
