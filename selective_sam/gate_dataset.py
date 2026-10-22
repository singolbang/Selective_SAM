"""Fixed IoU labels and RGB/depth inputs for final selector training."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .corruption import corrupt_depth
from .gate import gate_base, gate_input
from .label_generation import preprocessing_signature, sha256
from .samples import load_sample, read_manifest


class GateDataset(Dataset):
    def __init__(self, split: str, config: dict, adapter_hash: str):
        self.split, self.config = split, config
        self.rows = [json.loads(line) for line in (Path("cache/gate_labels") / f"{split}.jsonl").read_text().splitlines()]
        manifest = {row["id"]: row for row in read_manifest(split)}
        expected_config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        expected_preprocessing_hash = preprocessing_signature(config)
        expected_manifest_hash = sha256(Path("manifests") / f"{split}.jsonl")
        for row in self.rows:
            if (row["adapter_sha256"] != adapter_hash or row["config_sha256"] != expected_config_hash
                    or row.get("preprocessing_sha256") != expected_preprocessing_hash
                    or row["manifest_sha256"] != expected_manifest_hash or row["sam_sha256"] != config["sam"]["checkpoint_sha256"]):
                raise ValueError(f"Stale label cache in {split}: {row['image_id']}")
        self.manifest = manifest
        self.entries = [(i, t) for i, row in enumerate(self.rows) for t in range(len(row["targets"]))]
        self.targets = [self.rows[i]["targets"][t] for i, t in self.entries]
        self.conditions = np.array([self.rows[i]["condition"] for i, _ in self.entries])
        self.labels = np.array([target["label"] for target in self.targets], dtype=np.float32)
        self.baseline = np.array([target["baseline_iou"] for target in self.targets], dtype=np.float32)
        self.fusion = np.array([target["fusion_iou"] for target in self.targets], dtype=np.float32)
        self.inputs = self._get_inputs()

    def _get_inputs(self):
        path = Path("cache/gate_inputs")
        path.mkdir(parents=True, exist_ok=True)
        file = path / f"{self.split}.npy"
        metadata = path / f"{self.split}.json"
        expected = {"label_sha256": sha256(Path("cache/gate_labels") / f"{self.split}.jsonl"),
                    "size": self.config["gate"]["input_size"], "shape": [len(self.rows), 5,
                                                                            self.config["gate"]["input_size"],
                                                                            self.config["gate"]["input_size"]]}
        if file.is_file() and metadata.is_file() and json.loads(metadata.read_text()) == expected:
            return np.load(file, mmap_mode="r")
        array = np.lib.format.open_memmap(file.with_suffix(".npy.tmp"), mode="w+", dtype=np.uint8,
                                           shape=tuple(expected["shape"]))
        root = Path(self.config["data"]["root"])
        current_image_id, sample = None, None
        for i, row in enumerate(self.rows):
            if row["image_id"] != current_image_id:
                sample = load_sample(self.manifest[row["image_id"]], root, with_targets=False)
                current_image_id = row["image_id"]
            depth, valid = corrupt_depth(sample["depth"], sample["valid"], row["condition"],
                                         row["strength"], row["seed"])
            array[i] = gate_base(sample["rgb"], depth, valid, expected["size"])
            if (i + 1) % 500 == 0:
                print(f"gate input cache {self.split}: {i + 1}/{len(self.rows)}", flush=True)
        array.flush()
        del array
        file.with_suffix(".npy.tmp").replace(file)
        metadata.write_text(json.dumps(expected, indent=2))
        return np.load(file, mmap_mode="r")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row_index, target_index = self.entries[index]
        row = self.rows[row_index]
        target = row["targets"][target_index]
        original_hw = self.manifest[row["image_id"]]["shape"]
        x = gate_input(self.inputs[row_index], target["box"], original_hw)
        return x, torch.tensor(target["label"], dtype=torch.float32)
