"""Fixed baseline mask hints paired with the same IoU-labelled depth inputs."""

import json
from pathlib import Path

import numpy as np
import torch

from .dataset_sam import GateDatasetSAMROI, sam_cache_digest
from .gate_baseline import baseline_hint
from .label_generation import load_adapter, sha256
from .samples import load_sample


class GateDatasetBaselineSAM(GateDatasetSAMROI):
    def __init__(self, split, config, adapter_hash, gate_config):
        super().__init__(split, config, adapter_hash, gate_config)
        self.baseline_global, self.baseline_local = self._mask_inputs(adapter_hash)

    def _mask_inputs(self, adapter_hash):
        directory = Path("cache/gate_baseline_masks")
        directory.mkdir(parents=True, exist_ok=True)
        global_file = directory / f"{self.split}_global.npy"
        local_file = directory / f"{self.split}_local.npy"
        metadata_file = directory / f"{self.split}.json"
        size_global = self.roi_config["global_size"]
        size_local = self.roi_config["local_size"]
        identity = {
            "sam_sha256": self.config["sam"]["checkpoint_sha256"],
            "sam_config": self.config["sam"]["config"],
            "adapter_sha256": adapter_hash,
            "manifest_sha256": sha256(Path("manifests") / f"{self.split}.jsonl"),
            "label_sha256": sha256(Path("cache/gate_labels") / f"{self.split}.jsonl"),
            "model_source_sha256": sha256(Path("selective_sam/model.py")),
            "hint_source_sha256": sha256(Path("selective_sam/gate_baseline.py")),
            "dataset_source_sha256": sha256(Path("selective_sam/dataset_baseline.py")),
            "global_shape": [len(self.entries), size_global, size_global],
            "local_shape": [len(self.entries), size_local, size_local],
            "roi_margin_fraction": self.roi_config["roi_margin_fraction"],
            "roi_min_margin_px": self.roi_config["roi_min_margin_px"],
        }
        expected = {"identity": identity, "digest": sam_cache_digest(identity)}
        if (global_file.is_file() and local_file.is_file() and metadata_file.is_file()
                and json.loads(metadata_file.read_text()) == expected):
            return np.load(global_file, mmap_mode="r"), np.load(local_file, mmap_mode="r")
        model = load_adapter(self.config, Path("checkpoints/adapter_best.pt"))
        temp_global = global_file.with_suffix(".npy.tmp")
        temp_local = local_file.with_suffix(".npy.tmp")
        global_array = np.lib.format.open_memmap(temp_global, mode="w+", dtype=np.uint8,
                                                  shape=tuple(identity["global_shape"]))
        local_array = np.lib.format.open_memmap(temp_local, mode="w+", dtype=np.uint8,
                                                shape=tuple(identity["local_shape"]))
        root = Path(self.config["data"]["root"])
        current_image_id = None
        image_count = 0
        with torch.inference_mode():
            for i, (row_index, target_index) in enumerate(self.entries):
                row = self.rows[row_index]
                if row["image_id"] != current_image_id:
                    sample_row = self.manifest[row["image_id"]]
                    sample = load_sample(sample_row, root, with_targets=False)
                    features = model.encode_rgb(sample["rgb"])
                    targets = sample_row["targets"]
                    boxes = np.asarray([target["box"] for target in targets], dtype=np.float32)
                    logits = model.segment(features, boxes, sample["rgb"].shape[:2])
                    probabilities = torch.sigmoid(logits).cpu().numpy()
                    masks = {target["id"]: probabilities[j] for j, target in enumerate(targets)}
                    current_image_id = row["image_id"]
                    image_count += 1
                    if image_count % 100 == 0:
                        print(f"Baseline mask cache {self.split}: {image_count} images", flush=True)
                target = row["targets"][target_index]
                global_array[i], local_array[i] = baseline_hint(
                    masks[target["id"]], target["box"], sample["rgb"].shape[:2],
                    size_global, size_local, self.roi_config["roi_margin_fraction"],
                    self.roi_config["roi_min_margin_px"])
        global_array.flush()
        local_array.flush()
        del global_array, local_array, model
        temp_global.replace(global_file)
        temp_local.replace(local_file)
        metadata_file.write_text(json.dumps(expected, indent=2) + "\n")
        return np.load(global_file, mmap_mode="r"), np.load(local_file, mmap_mode="r")

    def __getitem__(self, index):
        global_input, local_input, geometry, sam_vector, label = super().__getitem__(index)
        global_hint = torch.from_numpy(np.asarray(self.baseline_global[index], dtype=np.float32) / 255)
        local_hint = torch.from_numpy(np.asarray(self.baseline_local[index], dtype=np.float32) / 255)
        return (torch.cat((global_input, global_hint[None]), dim=0),
                torch.cat((local_input, local_hint[None]), dim=0),
                geometry, sam_vector, label)
