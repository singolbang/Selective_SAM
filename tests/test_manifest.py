import unittest

from selective_sam.data import official_full_records


def records(split, groups):
    return [
        {"id": f"{split}/{weather}/{group}/{index}", "group": group, "condition": weather}
        for group in groups
        for weather in ("cloud", "fog", "night", "rain", "sun")
        for index in range(2)
    ]


class ManifestTests(unittest.TestCase):
    def test_every_official_train_image_is_reused_for_both_training_stages(self):
        source = {"train": records("train", [f"g{i}" for i in range(8)]),
                  "val": records("val", ["v0", "v1"]),
                  "test": records("test", ["t0", "t1"])}
        selected, audit = official_full_records(source, 42)
        self.assertEqual([len(selected[key]) for key in ("adapter_train", "gate_train", "val", "test")],
                         [80, 80, 20, 20])
        self.assertEqual([r["id"] for r in selected["adapter_train"]],
                         [r["id"] for r in selected["gate_train"]])
        self.assertTrue(audit["intentional_train_reuse"])
        train_groups = {r["group"] for r in selected["adapter_train"]}
        self.assertFalse(train_groups & {r["group"] for r in selected["val"] + selected["test"]})
        self.assertEqual(audit["official_cross_split_groups"], [])

    def test_cross_official_split_scene_priority_prevents_leakage(self):
        source = {"train": records("train", ["shared", "train_only"]),
                  "val": records("val", ["shared", "val_only"]),
                  "test": records("test", ["shared", "test_only"])}
        selected, audit = official_full_records(source, 42)
        self.assertEqual(audit["official_cross_split_groups"], ["shared"])
        self.assertFalse(any(row["group"] == "shared" for row in selected["val"] + selected["adapter_train"]))

    def test_identical_rgb_in_different_named_groups_removes_lower_priority_group(self):
        source = {
            "train": [{"id": "a", "group": "train_group", "condition": "sun", "rgb_sha256": "same"},
                      {"id": "b", "group": "safe", "condition": "sun", "rgb_sha256": "other"}],
            "val": [],
            "test": [{"id": "c", "group": "test_group", "condition": "sun", "rgb_sha256": "same"}],
        }
        selected, audit = official_full_records(source, 42)
        self.assertFalse(any(row["group"] == "train_group" for row in selected["adapter_train"]))
        self.assertEqual(audit["cross_split_identical_rgb_hashes"], ["same"])


if __name__ == "__main__":
    unittest.main()
