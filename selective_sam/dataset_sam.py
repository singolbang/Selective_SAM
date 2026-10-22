"""Pair fixed IoU labels with frozen SAM RGB feature summaries."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .dataset_roi import GateDatasetROI
from .gate_sam import SAM_FEATURE_DIM, sam_box_feature
from .label_generation import load_adapter, sha256, source_sha256
from .samples import load_sample


def sam_cache_digest(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


class GateDatasetSAMROI(GateDatasetROI):
    def __init__(self, split, config, adapter_hash, roi_config):
        super().__init__(split, config, adapter_hash, roi_config)
        self.sam_features = self._sam_features(adapter_hash)

    def _sam_features(self, adapter_hash):
        directory = Path("cache/gate_sam_features")
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / f"{self.split}.npy"
        metadata_path = directory / f"{self.split}.json"
        identity = {
            "sam_sha256": self.config["sam"]["checkpoint_sha256"],
            "sam_config": self.config["sam"]["config"],
            "adapter_sha256": adapter_hash,
            "manifest_sha256": sha256(Path("manifests") / f"{self.split}.jsonl"),
            "label_sha256": sha256(Path("cache/gate_labels") / f"{self.split}.jsonl"),
            "model_source_sha256": source_sha256(Path("selective_sam/model.py")),
            "feature_source_sha256": source_sha256(Path("selective_sam/gate_sam.py")),
            "dataset_source_sha256": source_sha256(Path("selective_sam/dataset_sam.py")),
            "shape": [len(self.entries), SAM_FEATURE_DIM],
        }
        expected = {"identity": identity, "digest": sam_cache_digest(identity)}
        if file.is_file() and metadata_path.is_file() and json.loads(metadata_path.read_text()) == expected:
            return np.load(file, mmap_mode="r")
        model = load_adapter(self.config, Path("checkpoints/adapter_best.pt"))
        temp = file.with_suffix(".npy.tmp")
        array = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32,
                                          shape=tuple(identity["shape"]))
        root = Path(self.config["data"]["root"])
        current_image_id = None
        features = None
        image_count = 0
        with torch.inference_mode():
            for i, (row_index, target_index) in enumerate(self.entries):
                row = self.rows[row_index]
                if row["image_id"] != current_image_id:
                    sample_row = self.manifest[row["image_id"]]
                    sample = load_sample(sample_row, root, with_targets=False)
                    features = model.encode_rgb(sample["rgb"])
                    current_image_id = row["image_id"]
                    image_count += 1
                    if image_count % 100 == 0:
                        print(f"SAM feature cache {self.split}: {image_count} images", flush=True)
                box = row["targets"][target_index]["box"]
                array[i] = sam_box_feature(features, box, sample["rgb"].shape[:2])
        array.flush()
        del array, model
        temp.replace(file)
        metadata_path.write_text(json.dumps(expected, indent=2) + "\n")
        return np.load(file, mmap_mode="r")

    def __getitem__(self, index):
        global_input, local_input, geometry, label = super().__getitem__(index)
        sam_vector = torch.from_numpy(np.array(self.sam_features[index], dtype=np.float32))
        return global_input, local_input, geometry, sam_vector, label
