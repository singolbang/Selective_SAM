"""Cached target crops paired with the fixed-IoU label rows."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .corruption import corrupt_depth
from .gate_roi import box_geometry, roi_input
from .label_generation import sha256, source_sha256
from .samples import load_sample
from .gate_dataset import GateDataset


class GateDatasetROI(GateDataset):
    def __init__(self, split, config, adapter_hash, roi_config):
        super().__init__(split, config, adapter_hash)
        self.roi_config = roi_config
        expected_source = source_sha256(Path("selective_sam/gate_roi.py"))
        input_config = {key: roi_config[key] for key in
                        ("global_size", "local_size", "roi_margin_fraction", "roi_min_margin_px")}
        expected_config = hashlib.sha256(
            json.dumps(input_config, sort_keys=True).encode()).hexdigest()
        if any(row.get("gate_roi_source_sha256") != expected_source or
               row.get("gate_roi_config_sha256") != expected_config for row in self.rows):
            raise ValueError(f"ROI label cache is stale for {split}")
        if config["gate"]["input_size"] != roi_config["global_size"]:
            raise ValueError("Global cache and ROI model disagree on input size")
        self.local_inputs = self._local_inputs()

    def _local_inputs(self):
        directory = Path("cache/gate_roi_inputs")
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / f"{self.split}.npy"
        metadata = directory / f"{self.split}.json"
        size = self.roi_config["local_size"]
        expected = {
            "label_sha256": sha256(Path("cache/gate_labels") / f"{self.split}.jsonl"),
            "gate_roi_source_sha256": source_sha256(Path("selective_sam/gate_roi.py")),
            "dataset_roi_source_sha256": source_sha256(Path("selective_sam/dataset_roi.py")),
            "local_size": size,
            "margin_fraction": self.roi_config["roi_margin_fraction"],
            "min_margin_px": self.roi_config["roi_min_margin_px"],
            "shape": [len(self.entries), 6, size, size],
        }
        if file.is_file() and metadata.is_file() and json.loads(metadata.read_text()) == expected:
            return np.load(file, mmap_mode="r")
        array = np.lib.format.open_memmap(file.with_suffix(".npy.tmp"), mode="w+",
                                           dtype=np.uint8, shape=tuple(expected["shape"]))
        root = Path(self.config["data"]["root"])
        current_image_id = None
        current_row_index = None
        sample = depth = valid = None
        for i, (row_index, target_index) in enumerate(self.entries):
            row = self.rows[row_index]
            if row["image_id"] != current_image_id:
                sample = load_sample(self.manifest[row["image_id"]], root, with_targets=False)
                current_image_id = row["image_id"]
            if row_index != current_row_index:
                depth, valid = corrupt_depth(sample["depth"], sample["valid"],
                                             row["condition"], row["strength"], row["seed"])
                current_row_index = row_index
            box = row["targets"][target_index]["box"]
            array[i], _ = roi_input(sample["rgb"], depth, valid, box, size,
                                    self.roi_config["roi_margin_fraction"],
                                    self.roi_config["roi_min_margin_px"])
            if (i + 1) % 1000 == 0:
                print(f"ROI input cache {self.split}: {i + 1}/{len(self.entries)}", flush=True)
        array.flush()
        del array
        file.with_suffix(".npy.tmp").replace(file)
        metadata.write_text(json.dumps(expected, indent=2))
        return np.load(file, mmap_mode="r")

    def __getitem__(self, index):
        global_input, label = super().__getitem__(index)
        row_index, target_index = self.entries[index]
        row = self.rows[row_index]
        box = row["targets"][target_index]["box"]
        shape = self.manifest[row["image_id"]]["shape"]
        local_input = torch.from_numpy(np.asarray(self.local_inputs[index], dtype=np.float32) / 255.0)
        geometry = torch.from_numpy(box_geometry(box, shape))
        return global_input, local_input, geometry, label
